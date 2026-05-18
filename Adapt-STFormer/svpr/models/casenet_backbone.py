import torch
import torch.nn as nn
import torchvision.models as models

from svpr.backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

vit_dict = {
    "vits": vit_small,
    "vitb": vit_base,
    "vitl": vit_large,
    "vitg": vit_giant2
}

AVAILABLE_MODELS = [
    'dinov2_vits14',
    'dinov2_vitb14',
    'dinov2_vitl14',
    'dinov2_vitg14'
]


def load_basic_model(model_name, model_pretrained=True):
    if model_name.startswith('dinov2'):
        backbone_type = model_name.split('_')[1] if len(
            model_name.split('_')) > 1 else 'vitb'
        encoder = vit_dict[backbone_type](
            patch_size=14, img_size=518, init_values=1, block_chunks=0)
        encoder_dim = encoder.embed_dim

    elif model_name.startswith('puredinov2'):
        encoder = get_pure_dinov2(num_unfrozen_blocks=2)
        encoder_dim = encoder.embed_dim
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")

    return encoder, encoder_dim


def get_pure_dinov2(backbone_name="dinov2_vitb14", num_unfrozen_blocks=2):
    assert backbone_name in AVAILABLE_MODELS, f"Backbone {backbone_name} is not recognized! Supported backbones are: {AVAILABLE_MODELS}"
    dino = torch.hub.load('facebookresearch/dinov2', backbone_name)

    dino.patch_embed.requires_grad_(False)
    dino.pos_embed.requires_grad_(False)

    for i in range(len(dino.blocks) - num_unfrozen_blocks):
        dino.blocks[i].requires_grad_(False)

    return dino