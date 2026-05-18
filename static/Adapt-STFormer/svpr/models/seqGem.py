import torch
from torch import nn
import torch.nn.functional as F
import einops

def seq_gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    B, D, SL = x.shape
    return F.avg_pool1d(x.clamp(min=eps).pow(p), SL).pow(1./p)

class SeqGeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        B, SL, D = x.shape
        x = einops.rearrange(x, "b sl d -> b d sl")
        x = seq_gem(x, p=self.p, eps=self.eps)
        assert x.shape == torch.Size([B, D, 1]), f"{x.shape}"
        return x[:, :, 0]
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

def seq_gem_modified(x, p=torch.ones(1) * 3, eps: float = 1e-6):
    # x: [..., S]
    # return x.clamp(min=eps).pow(p).mean(dim=-1, keepdim=True).pow(1.0 / p)
    B, N, D, SL = x.shape
    return F.avg_pool1d(x.reshape(B * N, D, SL).clamp(min=eps).pow(p), SL).pow(1. / p).reshape(B, N, D, 1)

class SeqGeM_modified(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: [B, S, N, D]
        B, S, N, D = x.shape
        # move sequence dimension to the end because seq_gem pools over last dim
        # [B, S, N, D] -> [B, N, D, S]
        x = einops.rearrange(x, "b s n d -> b n d s")

        x = seq_gem_modified(x, p=self.p, eps=self.eps)
        return x[..., 0]   # [B, N, D]

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"