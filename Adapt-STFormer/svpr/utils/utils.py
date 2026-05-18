import shutil
from torchvision import transforms
import torch
import logging
import numpy as np
import os
import argparse
from os.path import join
from get_datasets import get_dataset, get_splits, prefix_data
import numpy as np
import os


def save_checkpoint(args, state, is_best, filename):
    model_path = f"{args.output_folder}/{filename}"
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, f"{args.output_folder}/best_model.pth")


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(
        f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
        f"current_best_R@5 = {best_r5:.1f}"
    )
    if args.resume.endswith(
        "last_model.pth"
    ):  # Copy best model to current output_folder
        shutil.copy(
            args.resume.replace("last_model.pth", "best_model.pth"), args.output_folder
        )
    return model, optimizer, best_r5, start_epoch_num, not_improved_num


# def load_pretrained_backbone(args, model):
#     """Load a pretrained backbone"""
#     logging.debug(f"Loading checkpoint: {args.pretrain_model}")
#     checkpoint = torch.load(args.pretrain_model)
#     model.load_state_dict(checkpoint["model_state_dict"], strict=False)
#     return model


def configure_transform(image_dim, meta):
    normalize = transforms.Normalize(mean=meta["mean"], std=meta["std"])
    transform = transforms.Compose(
        [
            transforms.Resize(image_dim),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform


def print_free_memory():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)  # Actively used memory
    reserved_memory = torch.cuda.memory_reserved(0)    # PyTorch's reserved memory

    free_memory = total_memory - allocated_memory  # Correct free memory calculation
    # print(f"Total CUDA memory: {total_memory / 1e9:.2f} GB")
    # print(f"Allocated CUDA memory: {allocated_memory / 1e9:.2f} GB")
    # print(f"Reserved CUDA memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Free CUDA memory: {free_memory / 1e9:.2f} GB")  # Correct free memory


def getRecallAtN(n_values, predictions, gt, opt):
    correct_at_n = np.zeros(len(n_values))
    numQWithoutGt = 0
    incorrect_at_5 = {}

    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        # print(qIx)
        if len(gt[qIx]) == 0:
            numQWithoutGt += 1
            continue
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
        if opt.save_incorrect and (not np.any(np.in1d(pred[:5], gt[qIx]))):
            incorrect_at_5[qIx] = (pred[0], gt[qIx])

    if opt.save_incorrect:
        # Define the directory path
        dir_path = "incorrect_at_5_"
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        # Define the save path and save the file
        save_path = os.path.join(dir_path, f"incorrect_at_5_{opt.arch}_{opt.dataset}_seqL{opt.seqL}.npy")
        np.save(save_path, incorrect_at_5)

    print("(len(gt) - numQWithoutGt): ",(len(gt) - numQWithoutGt))
    return correct_at_n / (len(gt) - numQWithoutGt)

def create_dummy_input(opt):
    if opt.arch == "seqnet":
        dummy_input = torch.randn(1, opt.seqL, 4096).to(opt.device)
    else:
        dummy_input = torch.randn(1, opt.seqL, 3, opt.img_shape[0], opt.img_shape[1]).to(opt.device)
        dummy_input = dummy_input.contiguous().view(-1, 3, opt.img_shape[0], opt.img_shape[1])
    return dummy_input


def build_svpr_parser():
    parser = argparse.ArgumentParser(description="svpr")

    # General settings
    parser.add_argument(
        "--mode", type=str, default="train", help="Mode", choices=["train", "test"]
    )
    parser.add_argument("--expName", default="0", help="Unique string for an experiment")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--nocuda", action="store_true", help="Dont use cuda")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Train settings
    parser.add_argument(
        "--batchSize",
        type=int,
        default=16,
        help="Number of triplets (query, pos, negs). Each triplet consists of 12 images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Number of triplets (query, pos, negs). Each triplet consists of 12 images.",
    )
    parser.add_argument(
        "--cacheBatchSize", type=int, default=24, help="Batch size for caching and testing"
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=8,
        help="Batch size for inference (caching and testing)",
    )
    parser.add_argument(
        "--cacheRefreshRate",
        type=int,
        default=0,
        help="How often to refresh cache, in number of queries. 0 for off",
    )
    parser.add_argument(
        "--nEpochs", type=int, default=200, help="number of epochs to train for"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument("--nGPU", type=int, default=1, help="number of GPU to use.")
    parser.add_argument(
        "--optim", type=str, default="SGD", help="optimizer to use", choices=["SGD", "ADAM"]
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate.")
    parser.add_argument("--lrStep", type=float, default=50, help="Decay LR ever N steps.")
    parser.add_argument(
        "--lrGamma", type=float, default=0.5, help="Multiply LR by Gamma for decaying."
    )
    parser.add_argument(
        "--weightDecay", type=float, default=0.001, help="Weight decay for SGD."
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for each data loader to use",
    )
    parser.add_argument("--skip", default="1", help="sampling skip rate")
    parser.add_argument(
        "--patience", type=int, default=0, help="Patience for early stopping. 0 is off."
    )

    # Path settings
    parser.add_argument(
        "--runsPath",
        type=str,
        default=join(prefix_data, "runs"),
        help="Path to save runs to.",
    )
    parser.add_argument(
        "--savePath",
        type=str,
        default="checkpoints",
        help="Path to save checkpoints to in logdir. Default=checkpoints/",
    )
    parser.add_argument(
        "--cachePath",
        type=str,
        default=join(prefix_data, "cache"),
        help="Path to save cache to.",
    )
    parser.add_argument(
        "--resultsPath",
        type=str,
        default=None,
        help="Path to save evaluation results to when mode=test",
    )

    # Test settings
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to load checkpoint from, for resuming training or testing.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latest",
        help="Resume from latest or best checkpoint.",
        choices=["latest", "best", "external_pretrain"],
    )
    parser.add_argument(
        "--evalEvery",
        type=int,
        default=1,
        help="Do a validation set run, and save, every N epochs.",
    )
    parser.add_argument(
        "--evalTrain",
        type=str,
        default="false",
        help="Do a validation set run, and save, every N epochs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Data split to use for testing. Default is val",
        choices=["test", "train", "val"],
    )
    parser.add_argument("--extractOnly", action="store_true", help="extract descriptors")
    parser.add_argument(
        "--predictionsFile", type=str, default=None, help="path to prior predictions data"
    )
    parser.add_argument(
        "--seqL_filterData",
        type=int,
        help="during testing, db and qry inds will be removed that violate sequence boundaries for this given sequence length",
    )
    parser.add_argument("--save_incorrect", action="store_true", help="Save incorrectly classified samples")
    parser.add_argument("--vpr_application", action="store_true", help="Save valid predictions for VPR application")

    # Dataset and model settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="nordland-sw",
        help="Dataset to use",
        choices=[
            "nordland-sw",
            "nordland-sf",
            "oxford-hard",
            "oxford-easy",
            "nuscene",
            "oxford-nuscene-pairing",
        ],
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="simple_conv",
        choices=["simple_conv", "seqvlad"],
    )
    parser.add_argument("--seqL", type=int, default=5, help="Sequence Length")
    parser.add_argument(
        "--seq_length", type=int, default=15, help="Number of images in each sequence"
    )
    parser.add_argument("--w", type=int, default=3, help="filter size for seqNet")
    parser.add_argument(
        "--outDims", type=int, default=None, help="Output descriptor dimensions"
    )
    parser.add_argument(
        "--pca_dim", type=int, default=0, help="Output descriptor dimensions"
    )
    parser.add_argument(
        "--margin", type=float, default=0.1, help="Margin for triplet loss. Default=0.1"
    )
    parser.add_argument(
        "--descType",
        type=str,
        default="netvlad-pytorch",
        help="underlying descriptor type",
        choices=["netvlad-pytorch", "raw_378", "raw_224", "raw_384", "raw_322"],
    )
    parser.add_argument(
        "--nNeg", type=int, default=10, help="number of negatives per query"
    )
    parser.add_argument(
        "--freeze_layer",
        type=str,
        default="layer3",
        choices=["layer1", "layer2", "layer3", "layer4"],
        help="_",
    )
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    parser.add_argument(
        "--arch",
        type=str,
        default="seqvlad",
        choices=["seqvlad", "stformer", "adapt_stformer", "seqnet", "jist", "casenet"],
    )
    parser.add_argument(
        "--backbone", type=str, default="cct384", choices=["ResNet18", "dinov2_vits14", "dinov2_vitb14", "cct384", "vit_cct"]
    )
    parser.add_argument("--trunc_te_tatt", type=int, default=4, choices=list(range(0, 14)))
    parser.add_argument(
        "--freeze_te_tatt", type=int, default=-1, choices=list(range(-1, 14))
    )
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--rel_pos_temporal", action="store_true", default=False)
    parser.add_argument("--rel_pos_spatial", action="store_true", default=False)
    parser.add_argument("--abs_pos_embed", action="store_true", default=False)
    parser.add_argument("--no_recurrent", action="store_true", default=False)
    parser.add_argument("--no_seqgem", action="store_true", default=False)
    parser.add_argument("--no_recurrent_dte", action="store_true", default=False)
    parser.add_argument("--use_temporal_encoder", action="store_true", default=False)
    parser.add_argument("--spatial_ablation", action="store_true", default=False)
    parser.add_argument("--ft_reshape", action="store_true", default=False)
    parser.add_argument(
        "--k_points", type=int, default=8
    )
    parser.add_argument(
        "--num_levels", type=int, default=2
    )
    parser.add_argument(
        "--part", type=str, default=None, choices=["only_spatial", "only_temporal"]
    )
    parser.add_argument(
        "--fc_out", type=int, help="output size if aggregation=fc", default=768
    )
    parser.add_argument(
        "--img_shape",
        type=int,
        default=[0, 0],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )

    return parser

def save_valid_predictions(predictions,
                           seq_match_times,
                           eval_set,
                           gt,
                           inter_query_s=0.05,
                           save_dir="traj_results",
                           opt=None):
    """Save valid top-1 predicted GPS coordinates under FIFO real-time constraints.
    
    Args:
        predictions: 2D array of predicted indices for each query [n_queries, top_k]
        seq_match_times: Array of matching times for each query
        eval_set: Dataset object containing ground truth information
        inter_query_s: Time interval between queries in seconds
        save_dir: Directory to save output files
        opt: Options object with model and dataset information
        
    Returns:
        Tuple containing:
        - valid_preds: Valid predicted coordinates
        - valid_queries: Valid query coordinates
        - valid_times: Processing times for valid queries
        - on_time_mask: Boolean mask of which queries were processed on time
        - valid_q_indices: Indices of valid queries
        - valid_db_indices: Indices of valid database matches
        - adjusted_recall: Dictionary of recall@N metrics after FIFO scheduling
    """
    import os
    import numpy as np
    from .utils import getRecallAtN

    os.makedirs(save_dir, exist_ok=True)

    
    pred_top1 = predictions[:, 0]
    utm_db = np.asarray(eval_set.dbStruct.utmDb)
    utm_q  = np.asarray(eval_set.dbStruct.utmQ)

    pred_coords = utm_db[pred_top1]
    seq_match_times_q = np.asarray(seq_match_times[:len(pred_coords)], dtype=float)

    # FIFO scheduling
    T = float(inter_query_s)
    p = seq_match_times_q
    n = len(p)
    on_time_mask = np.zeros(n, dtype=bool)

    server_free_time = 0.0
    for i in range(n):
        arrival = i * T
        if arrival >= server_free_time:
            finish = arrival + p[i]
            server_free_time = finish
            on_time_mask[i] = True

    # Apply FIFO mask to predictions and ground truth
    valid_preds = pred_coords[on_time_mask]
    valid_times = seq_match_times_q[on_time_mask]
    valid_queries = utm_q[on_time_mask]
    valid_q_indices = np.nonzero(on_time_mask)[0]
    valid_db_indices = pred_top1[on_time_mask]
    
    # Get valid ground truth matches (only for queries that were processed on time)
    valid_gt = [gt[i] for i in range(len(gt)) if on_time_mask[i]]
    
    # Get valid predictions (only top-1 for each valid query)
    valid_predictions = np.zeros((len(valid_gt), 1), dtype=np.int32)
    for i, idx in enumerate(valid_q_indices):
        if idx < len(valid_gt):
            valid_predictions[i, 0] = predictions[idx, 0]  # Top-1 prediction
    
    # Calculate and print adjusted recall metrics
    n_values = [1]
    if len(valid_gt) > 0:  # Only calculate if we have valid queries
        recall_values = getRecallAtN(n_values, valid_predictions, valid_gt, opt)
        print("\n==== Adjusted Recall Metrics (FIFO Scheduling) ====")
        for i, n in enumerate(n_values):
            print(f"Recall@{n}: {recall_values[i]:.4f} "
                  f"(based on {len(valid_gt)}/{len(gt)} valid queries)")
        print("==========================================\n")
    
    # Save valid predictions to file
    out_txt = np.column_stack([valid_q_indices, valid_preds])
    np.savetxt(os.path.join(save_dir, f"{opt.arch}_valid_{opt.dataset}_preds.txt"),
               out_txt,
               fmt="%d %.6f %.6f")

    print(f"[save_vpr_valid_preds] Accepted {on_time_mask.sum()} / {len(on_time_mask)} queries "
          f"({on_time_mask.mean()*100:.1f}%) with FIFO scheduling, T={inter_query_s:.3f}s")

    return valid_preds, valid_queries, valid_times, on_time_mask, valid_q_indices, valid_db_indices
