import math
import torch
import torch.nn as nn
from svpr.models.spatial_cross_attention import SpatialCrossAttention2D
from svpr.models.dense_spatial_cross_attention import DenseCrossAttention2D
import time
from einops import rearrange

def ref_2d(bs,H, W):
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, device="cpu"),
        torch.linspace(0.5, W - 0.5, W, device="cpu"),
    )
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref_2d = torch.stack((ref_x, ref_y), -1)
    return ref_2d.repeat(bs, 1, 1).unsqueeze(2)

def flatten_frame(feat,num_tokens,embed_size): # feat shape (E, H, W) or (tokens, E)
    H0 = W0 = int(num_tokens**0.5)
    if feat.ndim == 2:                       # already (tokens, E)
        return feat.unsqueeze(0)             # → (1, tokens, E)
    return feat.view(embed_size, -1).permute(1, 0).unsqueeze(0)

class Detr(nn.Module):
    def __init__(self,
                 opt,
                 num_tokens: int,
                 embed_size: int,
                 heads: int,
                 use_positional_embedding: bool = False):
        super().__init__()
        self.opt = opt
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.heads = heads
        self.use_positional_embedding = use_positional_embedding

        # Pre-compute static tensors (device will be set later in forward())
        H0 = W0 = int(num_tokens**0.5)
        self.register_buffer('reference_points', ref_2d(1,H0, W0))  # Pre-compute grid
        self.register_buffer('spatial_shapes', torch.tensor([[H0, W0]], dtype=torch.long))
        self.register_buffer('level_start_index', torch.tensor([0]))

        # Learnable query offset
        self.query_offset = nn.Parameter(torch.zeros(1, num_tokens, embed_size))
        nn.init.xavier_uniform_(self.query_offset)

        # Attention modules
        if opt.spatial_ablation:
            self.spatial_cross_attention = DenseCrossAttention2D(
                embed_dims=embed_size,
                num_heads=heads,
                dropout=0.1,
                batch_first=True,
            )
        else:
            self.spatial_cross_attention = SpatialCrossAttention2D(
                embed_dims=embed_size,
                num_heads=heads,
                num_levels=opt.num_levels,  
                num_points=opt.k_points,
                dropout=0.1,
                batch_first=True,
            )
            
        self.cross_attn_norm = nn.LayerNorm(embed_size)

    def forward(self, img_ft, device):        
        # Move pre-computed tensors to device once
        ref_points        = self.reference_points.to(device)
        spatial_shapes    = self.spatial_shapes.to(device)
        level_start_index = self.level_start_index.to(device)
        img_ft = rearrange(img_ft, '(b s) n d -> b s n d', s=self.opt.seqL)

        H0 = W0 = int(self.num_tokens ** 0.5)
        B, S, N, D = img_ft.shape
        device = img_ft.device

        # Initial query from first frame
        feat0 = img_ft[:, 0].view(B, self.embed_size, H0, W0)
        flat0 = feat0.view(B, self.embed_size, -1).permute(0, 2, 1).contiguous()
        query = flat0 + self.query_offset.to(device)
        results = []

        for t in range(S):
            feat = img_ft[:, t].view(B, self.embed_size, H0, W0)
            flat = feat.view(B, self.embed_size, -1).permute(0, 2, 1).contiguous()

            cross = self.spatial_cross_attention(
                query=query,
                key=flat,
                value=flat,
                reference_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

            query = self.cross_attn_norm(cross)
            results.append(query)

        output = torch.stack(results, dim=1)

        return output


class DetrIdentity(nn.Module):
    def forward(self, x, device=None):
        return x