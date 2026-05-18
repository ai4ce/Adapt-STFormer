from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, xavier_init
from mmengine.registry import MODELS


class DropPath(nn.Module):
    """Stochastic Depth (identity if drop_prob=0)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class CrossAttention(nn.Module):
    """
    Dense cross-attention (spatial only) with absolute 2D pos embed for Q/K.
    - Accepts distinct Q/K/V.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        projection_dropout: float = 0.1,
        spatial_size: int = 24,  # H=W assumed
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.spatial_size = spatial_size

        # Absolute positional embedding
        self.pos_embed_q = nn.Parameter(torch.zeros(1, spatial_size * spatial_size, dim))
        self.pos_embed_k = nn.Parameter(torch.zeros(1, spatial_size * spatial_size, dim))
        nn.init.trunc_normal_(self.pos_embed_q, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_k, std=0.02)

        # Separate projections for cross-attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

        self._init_weights()

    def _init_weights(self):
        xavier_init(self.q_proj, distribution="uniform", bias=0.0)
        xavier_init(self.k_proj, distribution="uniform", bias=0.0)
        xavier_init(self.v_proj, distribution="uniform", bias=0.0)
        xavier_init(self.proj, distribution="uniform", bias=0.0)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, C) -> (B, h, N, d)
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,  # (B, Nq, C)
        key: torch.Tensor,    # (B, Nk, C)
        value: torch.Tensor,  # (B, Nk, C)
        query_pos: Optional[torch.Tensor] = None, # extra pos, optional
        **kwargs,
    ) -> torch.Tensor:
        B, Nq, _ = query.shape
        _, Nk, _ = key.shape

        # Add absolute positional embeddings
        if Nq == self.spatial_size * self.spatial_size:
            query = query + self.pos_embed_q
        if Nk == self.spatial_size * self.spatial_size:
            key = key + self.pos_embed_k

        if query_pos is not None:
            query = query + query_pos

        # Projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Heads: (B, h, N, d)
        q = self._shape_heads(q)
        k = self._shape_heads(k)
        v = self._shape_heads(v)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)  # (B, h, Nq, d)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, Nq, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@MODELS.register_module()
class DenseCrossAttention2D(BaseModule):
    """
    Dense cross-attention block with absolute 2D pos embed, pre-norm + residual.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        projection_dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        spatial_size: int = 24,
        batch_first: bool = True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        assert batch_first, "This implementation expects (B, N, C) tensors"

        self.pre_norm = nn.LayerNorm(embed_dims)
        self.attn = CrossAttention(
            dim=embed_dims,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            spatial_size=spatial_size,
        )
        self.drop_path = DropPath(drop_path_rate)
        self.post_norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        k = key if key is not None else query
        v = value if value is not None else k
        residual = query if identity is None else identity

        x = self.pre_norm(query)
        x = self.attn(x, k, v, query_pos=query_pos)
        x = residual + self.drop_path(x)
        x = self.post_norm(x)
        return x
