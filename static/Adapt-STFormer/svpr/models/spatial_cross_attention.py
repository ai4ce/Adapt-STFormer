from svpr.models.modified_multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import math
from mmengine.model import BaseModule, xavier_init, constant_init
from mmengine.registry import MODELS
import time


from mmcv.utils import ext_loader

@MODELS.register_module()
class SpatialCrossAttention2D(BaseModule):
    """2D-only Multi‑Scale Deformable Cross‑Attention (no camera intrinsics/3D)."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=2,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        head_dim = embed_dims // num_heads
        assert embed_dims % num_heads == 0, "embed_dims must divide num_heads"

        self.embed_dims     = embed_dims
        self.num_heads      = num_heads
        self.num_levels     = num_levels
        self.num_points     = num_points
        self.im2col_step    = im2col_step
        self.batch_first    = batch_first
        self.dropout        = nn.Dropout(dropout)

        # linear layers to predict offsets & attention weights
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )

        # project value and final output
        self.value_proj  = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.register_buffer("normalizer", torch.stack([torch.tensor([24.0]), torch.tensor([24.0])], dim=-1))

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers similarly to Deformable DETR."""
        constant_init(self.sampling_offsets, 0.0)
        # initialize offsets in a circular pattern
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  \
                        .view(self.num_heads, 1, 1, 2)             \
                        .repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[..., i, :] *= (i + 1)
        self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj,  distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
    ):

        if identity is None:
            identity = query

        bs, num_query, C = query.shape
        _, num_value, _ = value.shape

        value = self.value_proj(value) 

        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = (
            self.sampling_offsets(query)
            .view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        attention_weights = (
            self.attention_weights(query)
            .view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
            .softmax(-1)
            .view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        normalizer = self.normalizer
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / normalizer[None, None, None, :, None, :]
        )

        # breakpoint()

        if torch.cuda.is_available() and value.is_cuda:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        # breakpoint()
        output = self.output_proj(output)
        output = self.dropout(output) + identity

        return output