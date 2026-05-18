import torch
from torch import nn
import torch.nn.functional as F
import einops
from svpr.models.seqGem import SeqGeM

class JistModel(nn.Module):
    def __init__(self, args, agg_type="seqgem"):
        super().__init__()
        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                                       backbone=args.backbone, fc_output_dim=args.outDims)
        for name, param in self.model.named_parameters():
            if name.startswith("backbone.7"):  # Train only last residual block
                break
            param.requires_grad = False
        assert name.startswith("backbone.7"), "are you using a resnet? this only work with resnets"
        
        self.features_dim = self.model.aggregation[3].in_features
        self.fc_output_dim = self.model.aggregation[3].out_features
        self.seq_length = args.seqL
        self.aggregation_dim = self.fc_output_dim
        self.seq_gem = SeqGeM()
        self.agg_type = agg_type
        
    def forward(self, x):
        frames_features = self.model(x)
        aggregated_features = einops.rearrange(frames_features, "(b sl) d -> b sl d", sl=self.seq_length)
        return self.seq_gem(aggregated_features)
