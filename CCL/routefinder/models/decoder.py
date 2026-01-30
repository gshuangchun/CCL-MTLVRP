import random
from dataclasses import dataclass, fields
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbedding
from typing import Tuple, Union
from routefinder.models.nn.transformer import Normalization, TransformerBlock
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import PointerAttention, PointerAttnMoE
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger
from routefinder.models.nn.transformer import Normalization, ParallelGatedMLP
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor
import math
from rl4co.utils.ops import gather_by_index, get_distance

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


def scaled_dot_product_attention_simple(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention ."
        "Using custom implementation of scaled_dot_product_attention without Flash Attention. "
    )
    scaled_dot_product_attention = scaled_dot_product_attention_simple


class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            attention_dropout: float = 0.0,
            causal: bool = False,
            device: str = None,
            dtype: torch.dtype = None,
            sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
                self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, q, k, v, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        # q = self.Wq(q)
        # k = self.Wq(k)
        # v = self.Wq(v)
        q = rearrange(
            self.Wq(q), "b s (one h d) -> one b h s d", one=1, h=self.num_heads
        ).unbind(dim=0)[0]
        k = rearrange(
            self.Wk(k), "b s (one h d) -> one b h s d", one=1, h=self.num_heads
        ).unbind(dim=0)[0]
        v = rearrange(
            self.Wv(v), "b s (one h d) -> one b h s d", one=1, h=self.num_heads
        ).unbind(dim=0)[0]
        # q, k, v = rearrange(
        #     self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        # ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class CrossTransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
            normalization: Optional[str] = "instance",
            use_prenorm: bool = False,
            bias: bool = True,
            sdpa_fn: Optional[Callable] = None,
            moe_kwargs: Optional[dict] = None,
            parallel_gated_kwargs: Optional[dict] = None,
    ):
        super(CrossTransformerBlock, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        elif parallel_gated_kwargs is not None:
            ffn = ParallelGatedMLP(embed_dim, **parallel_gated_kwargs)
        else:
            ffn = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
                hidden_act="ReLU",
            )

        self.norm_attn_q = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm_attn_k = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm_attn_v = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.ffn = ffn
        self.use_prenorm = use_prenorm

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = q + self.attention(self.norm_attn_q(q), self.norm_attn_k(k), self.norm_attn_v(v), mask)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_attn(q + self.attention(q, k, v, mask))
            h = self.norm_ffn(h + self.ffn(h))
        return h


class RouteFinderDecoder(AttentionModelDecoder):
    """
    TODO
    Note that the real change is the pointer attention
    """

    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            env_name: str = "tsp",
            context_embedding: nn.Module = None,
            dynamic_embedding: nn.Module = None,
            mask_inner: bool = True,
            out_bias_pointer_attn: bool = False,
            linear_bias: bool = False,
            use_graph_context: bool = True,
            check_nan: bool = True,
            sdpa_fn: callable = None,
            pointer: nn.Module = None,
            normalization: str = "instance",
            use_prenorm: bool = False,
            feedforward_hidden: int = 512,
            num_layers: int = 1,
            use_post_layers_norm: bool = False,
            parallel_gated_kwargs: dict = None,
            moe_kwargs: dict = None,
            **transformer_kwargs,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        if context_embedding is None:
            log.info("Using default MTVRPContextEmbedding")
            context_embedding = MTVRPContextEmbedding(embed_dim)
        self.context_embedding = context_embedding

        if dynamic_embedding is None:
            log.info("Using default StaticEmbedding")
            self.dynamic_embedding = StaticEmbedding()
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

        if pointer is None:
            # MHA with Pointer mechanism (https://arxiv.org/abs/1506.03134)
            pointer_attn_class = (
                ReLDPointerAttention if moe_kwargs is None else PointerAttnMoE
            )
            pointer = pointer_attn_class(
                embed_dim,
                num_heads,
                mask_inner=mask_inner,
                out_bias=out_bias_pointer_attn,
                check_nan=check_nan,
                sdpa_fn=sdpa_fn,
                normalization=normalization,
                parallel_gated_kwargs=parallel_gated_kwargs,
                moe_kwargs=moe_kwargs,
            )

        self.pointer = pointer

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

        self.layers = nn.Sequential(
            *(
                CrossTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(1)
            )
        )
        # self.proj_q = nn.Linear(embed_dim, embed_dim)
        # self.proj_node = nn.Linear(embed_dim, embed_dim)

    def forward(
            self,
            td: TensorDict,
            cached: PrecomputedCache,
            num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)

        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        mask = td["action_mask"]
        prev_node = td["current_node"][:, :, None].expand(mask.shape)
        prev_loc = torch.gather(td["locs"], dim=2, index=prev_node.unsqueeze(-1).expand(-1, -1, -1, 2))
        # prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = td["locs"]
        distance = get_distance(prev_loc, curr_loc)
        logdis = -1 * torch.nan_to_num(torch.log(distance), nan=0.0, posinf=0.0, neginf=0.0)

        # glimpse_q, B_embedding, L_embedding, O_embedding, TW_embedding = self._compute_q(cached, td)
        glimpse = self._compute_q(cached, td)
        glimpse_q = glimpse[0]
        prob = 0.75
        if glimpse_q.shape[1] > 50:
            prob = 0.25
        if not self.training:
            prob = 0.15
            # prob = 1
            if glimpse_q.shape[1] > 50:
                prob = 0.02
                # prob = 0.5
        nodes = cached.node_embeddings
        if random.random() < prob:
            q_in = glimpse_q
            locs_x = td["locs"][:, 0, :, :].unsqueeze(2)
            locs_y = td["locs"][:, 0, :, :].unsqueeze(1)
            distance_xy = get_distance(locs_x, locs_y)
            logdis_xy = -1 * torch.nan_to_num(torch.log(distance_xy), nan=0.0, posinf=0.0, neginf=0.0)
            input = torch.cat((q_in, nodes), dim=1)

            first_mask = torch.zeros(input.shape[0], nodes.shape[1], input.shape[1]).cuda()
            first_mask[:, :, :logdis.shape[1]] = logdis.permute(0, 2, 1)
            first_mask[:, :, logdis.shape[1]:] = logdis_xy
            # outputs = self.layers[0](nodes, input, input)
            outputs = self.layers[0](nodes, input, input, mask=first_mask)
            # glimpse_q = outputs[:, :q_in.shape[1], :]
            node_embeddings = outputs
            cached = self._precompute_cache(node_embeddings, num_starts=num_starts)
            # nodes = node_embeddings + nodes

        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # for layer in self.layers:
        #     glimpse_q = layer(glimpse_q, glimpse_k, glimpse_v, mask)

        # Compute logits

        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        logits = logits

        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        # cached = self._precompute_cache(nodes, num_starts=num_starts)
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask, cached


class ReLDPointerAttention(nn.Module):
    """Calculate logits given query, key and value and logit key.
    This follows the pointer mechanism of Vinyals et al. (2015) (https://arxiv.org/abs/1506.03134).

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mask_inner: bool = True,
            out_bias: bool = False,
            check_nan: bool = True,
            sdpa_fn: Optional[Callable] = None,
            normalization: str = "instance",
            parallel_gated_kwargs: dict = None,
            **kwargs,

    ):
        super(ReLDPointerAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_inner = mask_inner

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention
        self.check_nan = check_nan
        # self.ffn = ParallelGatedMLP(embed_dim, **parallel_gated_kwargs)
        # self.norm_ffn = (
        #     Normalization(embed_dim, normalization)
        #     if normalization is not None
        #     else lambda x: x
        # )

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        # q_idt = query[-1]
        # query = query[0]
        heads = query + self._inner_mha(query, key, value, attn_mask)
        # heads = heads + self.ffn(self.norm_ffn(heads))
        glimpse = self._project_out(heads, attn_mask)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(self, query, key, value, attn_mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if self.mask_inner:
            # make mask the same number of dimensions as q
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )
        else:
            attn_mask = None
        heads = self.sdpa_fn(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)

    def _project_out(self, out, *kwargs):
        return self.project_out(out)
