import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import join, isfile
import seqNet
import svpr.models.adapt_stformer as adapt_stformer
from svpr.models.net import Net
from svpr.models.JIST import JistModel
from svpr.models.hvpr_encoders import HVPR_CaseNet
from torch.optim.lr_scheduler import StepLR

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def print_model_params(opt,model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{opt.arch} Total parameters: {total_params:,}")
    print(f"{opt.arch} Trainable parameters: {trainable_params:,}")


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def disable_stochastic(m):
    name = m.__class__.__name__.lower()

    if "drop" in name or "stochasticdepth" in name:
        m.eval()

def get_model(opt, encoder_dim, device):
    model = nn.Module()

    if opt.arch.lower() == "seqnet":
        seqFt = seqNet.seqNet(encoder_dim, opt.outDims, opt.seqL, opt.w)
        model.add_module("pool", nn.Sequential(*[seqFt, Flatten(), L2Norm()]))

    elif opt.arch == "seqnet_modified":
        model = SeqNetModified(opt)

    elif opt.arch == "seqvlad" or opt.arch == "stformer":
        model = Net(opt)

    elif opt.arch.lower() == "adapt_stformer":
        if opt.backbone == "cct384":
            model = adapt_stformer.Bev_RGB_Fusion(
                opt=opt, num_tokens=576, embed_size=384, heads=8
            )
            
    elif opt.arch.lower() == "jist":
        model = JistModel(opt)
        
    elif opt.arch.lower() == "casenet":
        model = HVPR_CaseNet(
            seq_len=opt.seqL if hasattr(opt, 'seqL') else 5,
            encoder_type=getattr(opt, 'encoder_type', 'bs_d_c'),
            is_pure=getattr(opt, 'is_pure', False)
        )

    else:
        raise Exception("requested model doesn't exist")

    # if not opt.resume:
    model = model.to(device)

    scheduler, optimizer, criterion = None, None, None
    
    if opt.mode.lower() == "train":
        if opt.arch == "stformer":
            optimizer = optim.Adam(
                [
                    {"params": model.encoder.spatial.parameters(), "lr": 0.0001},
                    {"params": model.encoder.temporal.parameters(), "lr": 0.001},
                    {"params": model.aggregator.parameters(), "lr": 0.0001},
                ]
            )
            criterion = nn.TripletMarginLoss(margin=0.1, p=2, reduction="sum").to(
                device
            )
        elif opt.arch == "seqnet":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
            criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, p=2, reduction='sum').to(device)

        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=opt.lr, weight_decay=opt.weightDecay
            )
            
            criterion = nn.TripletMarginLoss(
                margin=opt.margin, p=2, reduction="sum"
            ).to(device)

    print("Device count: ", torch.cuda.device_count())
    model = nn.DataParallel(model)
    if opt.resume:
        if opt.ckpt.lower() == "latest":
            resume_ckpt = join(opt.resume, "checkpoints", "checkpoint.pth.tar")
        elif opt.ckpt.lower() == "best":
            resume_ckpt = join(opt.resume, "checkpoints", "model_best.pth.tar")

        elif opt.ckpt.lower() == "external_pretrain":
            from collections import OrderedDict, defaultdict
            if isfile(opt.resume):
                print("=> loading checkpoint '{}'".format(opt.resume))
                checkpoint = torch.load(opt.resume, map_location="cpu")
                if opt.arch == "casenet":
                    checkpoint = checkpoint["model_state_dict"]
                new_state_dict = OrderedDict()
                param_counts = OrderedDict()
                grouped_params = defaultdict(int)
                total_params = 0

                for k, v in checkpoint.items():
                    new_key = "module." + k if not k.startswith("module.") else k
                    new_state_dict[new_key] = v
                    count = v.numel()
                    param_counts[new_key] = count
                    total_params += count

                    # Grouping: module.encoder.spatial.classifier.blocks.0.linear2.weight → encoder.spatial
                    parts = new_key.split('.')
                    group_key = '.'.join(parts[1:3]) if parts[0] == 'module' else parts[0]
                    grouped_params[group_key] += count

                model.load_state_dict(new_state_dict)
            else:
                print("=> no checkpoint found at '{}'".format(opt.resume))


        if opt.ckpt.lower() == "latest" or opt.ckpt.lower() == "best":
            if isfile(resume_ckpt):
                print("=> loading checkpoint '{}'".format(resume_ckpt))
                checkpoint = torch.load(
                    resume_ckpt, map_location=lambda storage, loc: storage
                )
                opt.start_epoch = checkpoint["epoch"]
                best_metric = checkpoint["best_score"]
                if opt.arch == "adapt_stformer":
                    model.load_state_dict(checkpoint["state_dict"],strict=False)
                else:
                    model.load_state_dict(checkpoint["state_dict"])
                model = model.to(device)
                if opt.mode == "train":
                    optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        resume_ckpt, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(resume_ckpt))
    print_model_params(opt,model)
    return model, optimizer, scheduler, criterion

