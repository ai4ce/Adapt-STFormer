import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)

class MultiScaleDeformableAttnFunction_fp32(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step

        B, Q, H, L_real, P, _ = sampling_locations.shape

        sampling_locations_fake = (
            sampling_locations.contiguous()
            .view(B, -1)[:, : Q * H * 1 * P * 2]
            .view(B, Q, H, 1, P, 2)
            .contiguous()
        )

        attention_weights_fake = (
            attention_weights.contiguous()
            .view(B, -1)[:, : Q * H * 1 * P]
            .view(B, Q, H, 1, P)
            .contiguous()
        )


        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations_fake,
            attention_weights_fake,
            im2col_step=ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors

        B, Q, H, L_real, P, _ = sampling_locations.shape
        sampling_locations_fake = (
            sampling_locations.contiguous()
            .view(B, -1)[:, : Q * H * 1 * P * 2]
            .view(B, Q, H, 1, P, 2)
            .contiguous()
        )

        attention_weights_fake = (
            attention_weights.contiguous()
            .view(B, -1)[:, : Q * H * 1 * P]
            .view(B, Q, H, 1, P)
            .contiguous()
        )

        grad_value = torch.zeros_like(value)

        grad_sampling_fake = torch.zeros_like(sampling_locations_fake)
        grad_attn_fake = torch.zeros_like(attention_weights_fake)

        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations_fake,
            attention_weights_fake,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_fake,
            grad_attn_fake,
            im2col_step=ctx.im2col_step,
        )
        grad_sampling_loc.view(B, -1)[:, :Q * H * P * 2] = grad_sampling_fake.reshape(B, -1)
        grad_attn.view(B, -1)[:, :Q * H * P] = grad_attn_fake.reshape(B, -1)

        return (
            grad_value,
            None,
            None,
            grad_sampling_loc,
            grad_attn,
            None,
        )