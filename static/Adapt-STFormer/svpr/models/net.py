import torch
from torch import nn
from svpr.models import pooling
from svpr.models.encoder import STEcoder, SeqVladModel, dinov2_encoder, crica_encoder, vit_cct_encoder
from svpr.models.JIST import SeqGeM
from svpr.utils.utils import print_free_memory
import time

class Net(nn.Module):
    """Network for VG applied to sequences.The used networks are composed of
    an encoder, a pooling layer and an aggregator."""

    def __init__(self, args):
        super().__init__()
        self.encoder = get_encoder(args)
        self.aggregator = get_aggregator(args)
        self.meta = {"outputdim": args.features_dim}
        self.args = args

        self.enc_s = torch.cuda.Event(enable_timing=True)
        self.enc_e = torch.cuda.Event(enable_timing=True)
        self.agg_s = torch.cuda.Event(enable_timing=True)
        self.agg_e = torch.cuda.Event(enable_timing=True)

    def forward(self, x):
        timings = {}

        # -- encode (seqvlad or stformer) --
        self.enc_s.record()
        if self.args.arch == "seqvlad":
            x = self.encoder(x)

        elif self.args.arch == "stformer":
            spatial, temporal = self.encoder(x)
            if   self.args.part == "only_spatial":
                x = spatial
            elif self.args.part == "only_temporal":
                x = temporal
            else:
                x = spatial + temporal

        self.enc_e.record()

        # -- aggregator/pooling --
        # print(" shape:        ", x.shape)
        # print(" dtype:        ", x.dtype)
        # print(" is_contig:    ", x.is_contiguous())
        # print(" strides:      ", x.stride())
        # print(" storage_ptr:  ", x.storage().data_ptr())
        # print(" data_ptr:     ", x.data_ptr())
        # print(" storage_off:  ", x.storage_offset())
        self.agg_s.record()
        x = self.aggregator(x) 
        self.agg_e.record()
        # synchronize once to flush all events
        torch.cuda.synchronize()

        # compute elapsed times (seconds)
        timings['encoder']    = self.enc_s.elapsed_time(self.enc_e)    / 1000.0
        timings['aggregator'] = self.agg_s.elapsed_time(self.agg_e)    / 1000.0
        print("timings:", timings)
        return x

def get_encoder(args):
    if args.arch == "seqvlad":
        if args.backbone == "cct384":
            encoder = SeqVladModel(args)
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
        elif args.backbone == "vit_cct":
            encoder = vit_cct_encoder(args)
            args.features_dim = 384 
    elif args.arch == "stformer":
        encoder = STEcoder(
            layer_s=args.trunc_te,
            layer_t=args.trunc_te_tatt,
            freeze_te=args.freeze_te,
            freeze_te_tatt=args.freeze_te_tatt,
            rel_pos_temporal=args.rel_pos_temporal,
            rel_pos_spatial=args.rel_pos_spatial,
        )
        args.features_dim = 384
    return encoder


def get_aggregator(args):
    aggregator = pooling.SeqVLAD(
        seq_length=args.seq_length, dim=args.features_dim, clusters_num=args.clusters,
        backbone_name = args.backbone
    )
    print("args.clusters: ",args.clusters)
    print("args.features_dim: ",args.features_dim)
    args.features_dim *= args.clusters
    return aggregator


def get_output_channels_dim(model, img_size):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, img_size[0], img_size[1]])).shape[1]


def get_output_tensor_dim(model, img_size):
    """Return the tensor shape in the output of a model."""
    return model(torch.ones([1, 3, img_size[0], img_size[1]])).shape[1:]
