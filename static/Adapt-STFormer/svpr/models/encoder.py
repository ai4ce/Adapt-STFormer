import torch
import torch.nn as nn
import logging
from svpr.models.cct import cct_14_7x2_384, cct_14_7x2_224
from svpr.utils.utils import print_free_memory
from timm import create_model
import time
from svpr.models.JIST import SeqGeM
import einops

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


class STEcoder(nn.Module):
    def __init__(
        self,
        layer_s=None,
        layer_t=None,
        freeze_te=None,
        freeze_te_tatt=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.trunc_te = layer_s
        self.trunc_te_t = layer_t
        self.freeze_te = freeze_te
        self.freeze_te_t = freeze_te_tatt
        # self.seqlen = 5
        self.is_inference = False
        self.rel_pos_temporal = kwargs.get("rel_pos_temporal", False)
        self.rel_pos_spatial = kwargs.get("rel_pos_spatial", False)

        self.spatial = cct_14_7x2_384(
            pretrained=True,
            progress=True,
            use_all_tokens=False,
            rel_pos_spatial=self.rel_pos_spatial,
            abs_pos_embed=kwargs.get("abs_pos_embed", False),
        )
        if self.trunc_te:
            logging.debug(
                f"Truncate CCT at spatial transformers encoder {self.trunc_te}"
            )
            self.spatial.classifier.blocks = torch.nn.ModuleList(
                self.spatial.classifier.blocks[: self.trunc_te].children()
            )
        if self.freeze_te:
            logging.debug(
                f"Freeze all the layers up to spatial tranformer encoder {self.freeze_te}"
            )
            for p in self.spatial.parameters():
                p.requires_grad = False
            for (
                name,
                child,
            ) in (
                self.spatial.classifier.blocks.named_children()
            ):  # name from 0 to args.trunc_te-1
                if int(name) > self.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True

        self.temporal = cct_14_7x2_384(
            pretrained=True,
            progress=True,
            use_all_token=False,
            use_for_t=True,
            rel_pos_temporal=self.rel_pos_temporal,
            abs_pos_embed=kwargs.get("abs_pos_embed", False),
        ).classifier  # 对应到区域时间注意力

        if self.trunc_te_t:
            logging.debug(
                f"Truncate CCT at temporal transformers encoder {self.trunc_te_t}"
            )
            self.temporal.blocks = torch.nn.ModuleList(
                self.temporal.blocks[: self.trunc_te_t].children()
            )
        if self.freeze_te_t:
            logging.debug(
                f"Freeze all the layers up to temporal tranformer encoder {self.freeze_te_t}"
            )
            for p in self.temporal.parameters():
                p.requires_grad = False
            for (
                name,
                child,
            ) in (
                self.temporal.blocks.named_children()
            ):  # name from 0 to args.trunc_te-1
                if int(name) > self.freeze_te_t:  # if freeze_te=1 freeze layer1h&layer2
                    for params in child.parameters():
                        params.requires_grad = True

        # self.tk_s  = torch.cuda.Event(enable_timing=True)
        # self.tk_e  = torch.cuda.Event(enable_timing=True)
        self.sp_s  = torch.cuda.Event(enable_timing=True)
        self.sp_e  = torch.cuda.Event(enable_timing=True)
        # self.tm_s  = torch.cuda.Event(enable_timing=True)
        # self.tm_e  = torch.cuda.Event(enable_timing=True)
        
    def forward(self, x):
        patches = self.spatial.tokenizer(x)
        self.sp_s.record()
        spatial_x = self.spatial.classifier(patches)
        self.sp_e.record()
        torch.cuda.synchronize()
        print("frame attn: ",self.sp_s.elapsed_time(self.sp_e) / 1000.0)

        # temporal_x = self.temporal(patches)
        
        return spatial_x, spatial_x

    def forward_seqvlad(self, x):
        patches = self.spatial.tokenizer(x)
        spatial_x = self.spatial.classifier(patches)
        return spatial_x


def SeqVladModel(args):
    encoder = cct_14_7x2_384(
        pretrained=True,
        progress=True,
        use_all_tokens=False,
        rel_pos_spatial=args.rel_pos_spatial,
        abs_pos_embed=args.abs_pos_embed,
    )
    if args.trunc_te:
        logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
        encoder.classifier.blocks = torch.nn.ModuleList(
            encoder.classifier.blocks[: args.trunc_te].children()
        )
    if args.freeze_te:
        logging.debug(
            f"Freeze all the layers up to tranformer encoder {args.freeze_te}"
        )
        for p in encoder.parameters():
            p.requires_grad = False
        for (
            name,
            child,
        ) in (
            encoder.classifier.blocks.named_children()
        ):  # name from 0 to args.trunc_te-1
            if int(name) > args.freeze_te:
                for params in child.parameters():
                    params.requires_grad = True
    args.features_dim = 384
    return encoder


def dinov2_encoder(opt):
    
    if opt.backbone == "dinov2_vits14":
        dinov2_layer = DinoV2_self(
            model_name="dinov2_vits14", layer1=11, norm_descs=True, use_cls=False
        )
    elif opt.backbone == "dinov2_vitb14":
        dinov2_layer = DinoV2_self(
            model_name="dinov2_vitb14", layer1=11, norm_descs=True, use_cls=False
        )

    for name, param in dinov2_layer.named_parameters():
        if "blocks.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in dinov2_layer.named_parameters():
        if param.requires_grad:
            print(f"{name} is not frozen")
            
    return dinov2_layer


def vit_cct_encoder(opt):
    """
    Instantiate Vit_CCT, then freeze everything except:
      • the SpatialConvPatchEmbed stem (patch_embed)
      • the final transformer block (blocks.11)
    """
    # 1) build the model
    encoder = Vit_CCT(
        model_name="dinov2_vits14",
        layer1=11,        # used to define how many blocks exist
        pretrained=True,
    )

    # 2) freeze all parameters
    for _, param in encoder.named_parameters():
        param.requires_grad = False

    # 3) un-freeze the spatial conv stem (patch_embed)
    for name, param in encoder.named_parameters():
        if "patch_embed" in name:
            param.requires_grad = True

    # 4) un-freeze only the very last block (block index 11)
    for name, param in encoder.named_parameters():
        if "blocks.11" in name:
            param.requires_grad = True

    # 5) sanity-check
    print("\nTrainable parameters:")
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            print("  ", name)
    print()

    return encoder


 #======================================== crica vpr ==================================================#

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import partial


class crica_encoder(nn.Module):
    def __init__(self, opt):
        """
        opt: argument namespace with opt.backbone in {"crica_small","crica_big"}
        """
        super().__init__()

        # print("print(opt.backbone): ",print(opt.backbone))
        if opt.backbone == "crica_big":
            backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
            print("🔍 pos_embed shape before loading weights:", backbone.pos_embed.shape)

            # 1) get your model’s keys
            own_sd = backbone.state_dict()

            # 2) load the raw checkpoint
            ckpt = torch.load("/scratch/yl9727/adaptiveNet/AdaptiveNet/models/backbone/CricaVPR.pth")['model_state_dict']

            # 3) clean up the prefixes
            clean_ckpt = {}
            for k, v in ckpt.items():
                new_k = k
                # strip DataParallel “module.”
                if new_k.startswith("module."):
                    new_k = new_k[len("module."):]
                # strip the “backbone.” wrapper
                if new_k.startswith("backbone."):
                    new_k = new_k[len("backbone."):]
                clean_ckpt[new_k] = v

            # 4) merge: overwrite your model’s entries with checkpoint ones
            own_sd.update({k: v for k, v in clean_ckpt.items() if k in own_sd})

            # 5) load into your backbone
            backbone.load_state_dict(own_sd)

            print("✅ pos_embed shape after loading weights:", backbone.pos_embed.shape)



            self.embed_dim = 768

        else:
            raise ValueError(f"Unknown backbone {opt.backbone}")

        for name, param in backbone.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

        print("print(backbone.patch_embed.img_size): ",backbone.patch_embed.img_size)
        # now list the params that are still trainable, with a fire emoji
        print("🔥 Unfrozen (trainable) parameters:")
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                print(f"🔥 {name}")

        self.backbone = backbone

        # self.backbone.norm = nn.Identity()
        # self.backbone.head = nn.Identity()

        # import pdb
        # pdb.set_trace()

    def forward(self, x):
        x = self.backbone(x)

        # x_prenorm = x["x_prenorm"]            # (B, P,   D)
        # x_patches = x_prenorm[:, 1:, :]       # (B, P-1, D)
        # B, Pm1, D = x_patches.shape           # Pm1 == P-1
        # W = H = int(math.sqrt(Pm1))           # now W*H == P-1
        # x_p = x_patches.view(B, W, H, D)      # OK
        # x_p = x_p.permute(0, 3, 1, 2)         # (B, D, H, W)
        # print(x_p)
        # import pdb
        # pdb.set_trace()
        
        B,P,D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P-1))
        x0 = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 

        return x_p
