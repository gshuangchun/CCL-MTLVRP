from typing import Tuple, Union

import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder
from routefinder.models.nn.transformer import Normalization, TransformerBlock

log = get_pylogger(__name__)
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

import itertools
import math
import warnings

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.moe import MoE
from rl4co.utils import get_pylogger


class RouteFinderEncoder(nn.Module):
    """
    Encoder for RouteFinder model based on the Transformer Architecture.
    Here we include additional embedding from raw to embedding space, as
    well as more modern architecture options compared to the usual Attention Models
    based on POMO (including multi-task VRP ones).
    """

    def __init__(
            self,
            init_embedding: nn.Module = None,
            num_heads: int = 8,
            embed_dim: int = 128,
            num_layers: int = 6,
            feedforward_hidden: int = 512,
            normalization: str = "instance",
            use_prenorm: bool = False,
            use_post_layers_norm: bool = False,
            parallel_gated_kwargs: dict = None,
            **transformer_kwargs,
    ):
        super(RouteFinderEncoder, self).__init__()

        if init_embedding is None:
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
        else:
            log.warning("Using custom init_embedding")
        self.init_embedding = init_embedding

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )
        self.layers_mlp = nn.Sequential(
            *(
                nn.Linear(embed_dim, embed_dim)
                for _ in range(num_layers)
            )
        )

        self.layers_spa = nn.Sequential(
            *(
                SparseTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )
        self.layers_spa_mlp = nn.Sequential(
            *(
                nn.Linear(embed_dim, embed_dim)
                for _ in range(num_layers)
            )
        )
        self.prompt_mlp = nn.Linear(4, embed_dim)
        self.proj_mlp = nn.Linear(embed_dim * 2, embed_dim)

        self.post_layers_norm = (
            Normalization(embed_dim, normalization) if use_post_layers_norm else None
        )
        self.ln=nn.LayerNorm(embed_dim*2)

    def forward(
            self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]
        has_open = td["open_route"].squeeze(-1)
        has_tw = (td["time_windows"][..., 1] != float("inf")).any(-1)
        has_limit = (td["distance_limit"] != float("inf")).squeeze(-1)
        has_backhaul = (td["demand_backhaul"] != 0).any(-1)
        # backhaul_class = td.get("backhaul_class", torch.full_like(has_open, 1))
        prompt = torch.cat((has_open[:, None], has_tw[:, None], has_limit[:, None], has_backhaul[:, None]), dim=-1)
        prompt = self.prompt_mlp(prompt[:, None, :].to(dtype=torch.float)).expand(init_h.shape)

        # Process embedding
        h = self.proj_mlp(self.ln(torch.cat((init_h, prompt), dim=-1)))
        h1 = init_h
        for layer, layer1, mlp, mlp1 in zip(self.layers, self.layers_spa, self.layers_mlp, self.layers_spa_mlp):
            h = layer(h, mask)
            h1 = layer1(h1, mask)
            h = h + mlp1(h1)
            h1 = h1 + mlp(h)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h, init_h  # [B, N, H]


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
                "rms": RMSNorm,
            }.get(normalization, None)
            self.normalizer = (
                normalizer_class(embed_dim, affine=True)
                if normalizer_class is not None
                else None
            )
        else:
            self.normalizer = "layer"
        if self.normalizer is None:
            log.error(
                "Normalization type {} not found. Skipping normalization.".format(
                    normalization
                )
            )

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif isinstance(self.normalizer, RMSNorm):
            return self.normalizer(x)
        else:
            assert self.normalizer is None, "Unknown normalizer type {}".format(
                self.normalizer
            )
            return x


class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
            self,
            hidden_size: int = 128,
            inner_size_multiple_of: int = 256,
            mlp_activation: str = "silu",
            model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
                (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)


class SparseTransformerBlock(nn.Module):
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
        super(SparseTransformerBlock, self).__init__()
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

        self.norm_attn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.attention = SparseMultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.ffn = ffn
        self.use_prenorm = use_prenorm

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = x + self.attention(self.norm_attn(x), mask)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_attn(x + self.attention(x, mask))
            h = self.norm_ffn(h + self.ffn(h))
        return h


import math


def scaled_dot_product_attention_sparse(
        q, k, v, attn_mask=None, dropout_p=0.0):
    """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
    # Check for causal and attn_mask conflict

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    # 计算每行（51, 51）的前 25 大的值和其索引
    topk_values, topk_indices = torch.topk(scores, k=(scores.shape[-1] - 1) // 2, dim=-1)

    # 创建 mask，初始化为全 False
    attn_mask = torch.zeros_like(scores, dtype=torch.bool)

    # 直接使用 scatter_ 来设置前 25 个最大值的位置为 True
    attn_mask.scatter_(dim=-1, index=topk_indices, value=True)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


class SparseMultiHeadAttention(nn.Module):
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
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention_sparse

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
                self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

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
