from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

from routefinder.models.encoder import RouteFinderEncoder
from routefinder.models.env_embeddings.mtvrp import (
    MTVRPContextEmbeddingRouteFinder,
    MTVRPInitEmbeddingRouteFinder,
)
import abc

from typing import Any, Callable, Optional, Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger
from routefinder.models.decoder import RouteFinderDecoder
import torch
log = get_pylogger(__name__)


class RouteFinderPolicy(AttentionModelPolicy):
    """
    Main RouteFinder policy based on the Transformer Architecture.
    We use the base AttentionModelPolicy for decoding (i.e. masked attention + pointer network)
    and our new RouteFinderEncoder for the encoder.
    """

    def __init__(
            self,
            embed_dim: int = 128,
            num_encoder_layers: int = 6,
            num_heads: int = 8,
            normalization: str = "instance",
            feedforward_hidden: int = 512,
            parallel_gated_kwargs: dict = None,
            encoder_use_post_layers_norm: bool = False,
            encoder_use_prenorm: bool = False,
            env_name: str = "mtvrp",
            use_graph_context: bool = False,
            init_embedding: MTVRPInitEmbeddingRouteFinder = None,
            context_embedding: MTVRPContextEmbeddingRouteFinder = None,
            extra_encoder_kwargs: dict = {},
            **kwargs,
    ):

        encoder = RouteFinderEncoder(
            init_embedding=init_embedding,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            **extra_encoder_kwargs,
        )
        decoder = RouteFinderDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            use_graph_context=use_graph_context,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            **extra_encoder_kwargs,
        )

        if context_embedding is None:
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        super(RouteFinderPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            **kwargs,
        )

    def forward(
            self,
            td: TensorDict,
            env: Optional[Union[str, RL4COEnvBase]] = None,
            phase: str = "train",
            calc_reward: bool = True,
            return_actions: bool = False,
            return_entropy: bool = False,
            return_hidden: bool = False,
            return_init_embeds: bool = False,
            return_sum_log_likelihood: bool = True,
            actions=None,
            max_steps=1_000_000,
            **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)
        labels_O = td["open_route"].squeeze(-1).to(dtype=torch.int)
        labels_TW = (td["time_windows"][..., 1] != float("inf")).any(-1).to(dtype=torch.int)
        labels_L = (td["distance_limit"] != float("inf")).squeeze(-1).to(dtype=torch.int)
        labels_B = (td["demand_backhaul"] != 0).any(-1).to(dtype=torch.int)
        td["task_label"] = torch.stack([labels_B, labels_L, labels_O, labels_TW], dim=-1)
        max_steps=2000
        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask, hidden = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict
