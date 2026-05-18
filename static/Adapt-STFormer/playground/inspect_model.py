import torch

# Load checkpoint
crica_ckpt = torch.load("/scratch/yl9727/adaptiveNet/AdaptiveNet/models/backbone/CricaVPR.pth")
crica_ckpt = crica_ckpt['model_state_dict'].keys()
print(crica_ckpt)

