from svpr.models.detr import Detr,DetrIdentity
import torch
import torch.nn as nn
from svpr.models.encoder import SeqVladModel, dinov2_encoder, crica_encoder, vit_cct_encoder
from svpr.models import pooling
from svpr.utils.utils import print_free_memory
from svpr.models.seqGem import SeqGeM
from svpr.models.seqGem import SeqGeM_modified
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA
import math
import time
from einops import rearrange

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
class Bev_RGB_Fusion(nn.Module):
    def __init__(self, opt, num_tokens, embed_size, heads):
        super(Bev_RGB_Fusion, self).__init__()
        self.opt = opt
        self.encoder = get_encoder(opt)

        original_seq_len = opt.seq_length

        if opt.use_temporal_encoder or opt.no_seqgem:
            self.aggregator = get_aggregator(opt)
        else:
            opt.seq_length = 1
            self.aggregator = get_aggregator(opt)
            opt.seq_length = original_seq_len
        if opt.no_recurrent_dte:
            self.detr = DetrIdentity()
        else:
            self.detr = Detr(
                num_tokens=num_tokens, 
                embed_size=embed_size, 
                heads=heads,
                use_positional_embedding=False,
                opt=opt  
            )
        if opt.use_temporal_encoder:
            temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1)
            self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=2)
        elif opt.no_seqgem:
            pass
        else:
            self.seqgem = SeqGeM()
            self.seqgem_modified = SeqGeM_modified()

    def forward(self, x, device="cuda"):
        x = self.encoder(x)
        
        x = self.detr(x, device)
        # print("x after detr: ", x[:4,:], x.shape)

        if self.opt.use_temporal_encoder:
            x = x.unsqueeze(0)
            B, S, N, E = x.shape
            x = rearrange(x, 'b s n e -> b (s n) e')  # → (B, S*N, E)
            x = self.temporal_encoder(x)
            x = rearrange(x, 'b (s n) e -> b s n e', s=S)
            x = x.squeeze(0)
        
        elif self.opt.no_seqgem:
            pass
            
        else:
            x = self.seqgem_modified(x)

        # -- aggregator --
        x = self.aggregator(x)

        return x

def get_encoder(args):
    if args.backbone == "cct384":
        encoder = SeqVladModel(args)
        args.features_dim = 384 

    elif args.backbone == "vit_cct":
        encoder = vit_cct_encoder(args)
        args.features_dim = 384 
        
    elif "dinov2" in args.backbone:
        encoder = dinov2_encoder(args)
        args.features_dim = 768
        if args.backbone == "dinov2_vits14":
            args.features_dim = 384
    elif args.backbone == "ResNet18":
        encoder = resnet_encoder(args)
        args.features_dim = 512
    elif args.backbone == "crica_big":
        encoder = crica_encoder(args)
        args.features_dim = 768
    return encoder

class ConvPooling(nn.Module):
    def __init__(self, in_dim=384):
        super(ConvPooling, self).__init__()
        # input expected as [B, L, D] where L = 576 (seq_len), D = in_dim
        # reshape to [B, D, 24, 24] → conv → [B, D, 8, 8] → flatten
        self.in_dim = in_dim
        self.conv = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=7, stride=3, padding=2)

    def forward(self, x):
        B, L, D = x.shape
        H = W = int(L**0.5)  # assume square
        x = x.view(B, H, W, D).permute(0, 3, 1, 2)   # [B, D, H, W]
        x = self.conv(x)                             # [B, D, 8, 8]
        x = x.contiguous().view(B, -1)   
        x = x.mean(dim=0)       
        x = F.normalize(x, p=2, dim=-1)  
        return x


def get_aggregator(args):
    aggregator = pooling.SeqVLAD(
        seq_length=args.seq_length, dim=args.features_dim, clusters_num=args.clusters,
        backbone_name = args.backbone
    )
    print("args.clusters: ",args.clusters)
    print("args.features_dim: ",args.features_dim)
    args.features_dim *= args.clusters
    return aggregator
