import torch.nn as nn

from svpr.models.casenet_backbone import get_pure_dinov2
from svpr.backbone.vision_transformer import vit_base
from svpr.models.casenet import CaseNet

class HVPR_CaseNet(nn.Module):
    def __init__(self, seq_len=5, encoder_type="bs_d_c", is_pure=False):
        super().__init__()
        if is_pure:
            self.encoder = get_pure_dinov2(num_unfrozen_blocks=2)
        else:
            self.encoder = vit_base(patch_size=14, img_size=518,
                             init_values=1, block_chunks=0)
        self.pool = CaseNet(seq_len=seq_len, encode_type=encoder_type)
        self.seq_length = seq_len

    def forward(self, x, single_img=False):
        features = self.encoder(x, is_training=True)
        # img_desc, seq_desc = self.pool(features, output_img=False, single_img = single_img)
        # return img_desc, seq_desc #NOTE: Check this?
        seq_desc = self.pool(features, output_img=False, single_img = single_img)
        return seq_desc