import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import copy

from svpr.models.adapt_stformer import Bev_RGB_Fusion


class SimpleOpt:
    def __init__(self):
        self.spatial_ablation = False
        self.num_levels = 1
        self.k_points = 8
        self.no_recurrent = False
        self.use_temporal_encoder = False
        self.seqL = 3
        self.ft_reshape = True
        self.backbone = "cct384"
        self.rel_pos_spatial = False
        self.rel_pos_temporal = False
        self.abs_pos_embed = False
        self.trunc_te = 8
        self.freeze_te = 1
        self.seq_length = 3
        self.no_seqgem = False
        self.clusters = 64
        self.no_recurrent_dte = False
        self.img_shape = [384, 384]
        self.arch = "adapt_stformer"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_stochastic(m):
    name = m.__class__.__name__.lower()
    if "drop" in name or "stochasticdepth" in name:
        m.eval()


def flatten_output(x):
    return x.view(x.shape[0], -1)


def main():
    set_seed(42)

    num_tokens = 576
    embed_size = 384
    heads = 1
    batch_size = 2
    seq_length = 3
    num_iters = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    opt = SimpleOpt()

    # -------------------------
    # Create two separate models
    # -------------------------
    model_loop = Bev_RGB_Fusion(opt, num_tokens, embed_size, heads).to(device)
    model_batch = Bev_RGB_Fusion(opt, num_tokens, embed_size, heads).to(device)

    # Force identical initialization
    model_batch.load_state_dict(copy.deepcopy(model_loop.state_dict()))

    model_loop.train()
    model_batch.train()

    model_loop.apply(disable_stochastic)
    model_batch.apply(disable_stochastic)

    optimizer_loop = optim.Adam(
        [p for p in model_loop.parameters() if p.requires_grad],
        lr=1e-5,
    )

    optimizer_batch = optim.Adam(
        [p for p in model_batch.parameters() if p.requires_grad],
        lr=1e-5,
    )

    for it in range(num_iters):
        set_seed(1000 + it)

        query = torch.randn(
            batch_size,
            seq_length,
            3,
            opt.img_shape[0],
            opt.img_shape[1],
            device=device,
        )
        batched_input = query.contiguous().view(
            -1, 3, opt.img_shape[0], opt.img_shape[1]
        )

        # ============================================================
        # Scenario 1: looped model train step
        # ============================================================
        optimizer_loop.zero_grad(set_to_none=True)

        # concat_input = query.contiguous().view(
        #     -1, 3, opt.img_shape[0], opt.img_shape[1]
        # )
        loop_outputs = []
        for inp in query:
            out = model_loop(inp, device=device)
            loop_outputs.append(out)

        seq_encoding_loop = torch.stack(loop_outputs).squeeze(-1).squeeze(1)

        # ============================================================
        # Scenario 2: batched model train step
        # ============================================================
        optimizer_batch.zero_grad(set_to_none=True)

        seq_encoding_batch = model_batch(
            batched_input,
            device=device,
        )
        breakpoint()
        # Same target for both models
        target = torch.randn_like(seq_encoding_batch)
        target = torch.randn_like(seq_encoding_loop)

        loss_loop = nn.functional.mse_loss(seq_encoding_loop, target)
        loss_batch = nn.functional.mse_loss(seq_encoding_batch, target)

        loss_loop.backward()
        loss_batch.backward()

        optimizer_loop.step()
        optimizer_batch.step()

        # ============================================================
        # Compare AFTER optimization
        # ============================================================
        with torch.no_grad():
            loop_outputs_eval = []
            for inp in query:
                out = model_loop(inp, device=device)
                loop_outputs_eval.append(out)

            seq_encoding_loop_eval = torch.stack(loop_outputs_eval).squeeze(-1).squeeze(1)

            seq_encoding_batch_eval = model_batch(
                batched_input,
                device=device,
            )
            breakpoint()
            diff = (seq_encoding_loop_eval - seq_encoding_batch_eval).abs()

            print(f"\niter {it}")
            print("loop shape:", seq_encoding_loop_eval.shape)
            print("batch shape:", seq_encoding_batch_eval.shape)
            print("max abs diff:", diff.max().item())
            print("mean abs diff:", diff.mean().item())


    print("done")


if __name__ == "__main__":
    main()