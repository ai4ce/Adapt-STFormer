from __future__ import print_function
import argparse
import random, shutil, json
from os.path import join, exists
from os import makedirs
import torch
from datetime import datetime
from tqdm import tqdm

from tensorboardX import SummaryWriter
import numpy as np
import sys

from joblib import dump, load

from get_datasets import get_dataset, get_splits, prefix_data
from get_models import get_model
from plot_graphs import plot_graphs
from train import train
from test import test

import matplotlib.pyplot as plt
import numpy as np

import subprocess

from svpr.utils.pca import compute_pca
from svpr.utils.utils import build_svpr_parser

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, "model_best.pth.tar"))


if __name__ == "__main__":
    print("entering running code")
    
    parser = build_svpr_parser()
    opt = parser.parse_args()

    restore_var = [
        "lr",
        "lrStep",
        "lrGamma",
        "weightDecay",
        "momentum",
        "runsPath",
        "savePath",
        "optim",
        "margin",
        "seed",
        "patience",
        "outDims",
        "w",
    ]
    if opt.arch.lower() != "s1+seqmatch":
        restore_var = restore_var + ["arch"]
    if opt.resume:
        if opt.arch.lower() in ["single", "smooth", "delta", "single+seqmatch"]:
            raise Exception("Use arch 'seqnet' with resume")
        flag_file = join(opt.resume, "checkpoints", "flags.json")
        if exists(flag_file):
            with open(flag_file, "r") as f:
                stored_flags = {
                    "--" + k: str(v)
                    for k, v in json.load(f).items()
                    if k in restore_var
                }
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ""
                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [
                    x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0
                ]
                print("Restored flags:", train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    opt.seq_length = opt.seqL
    opt.train_batch_size = opt.batchSize
    opt.infer_batch_size = opt.cacheBatchSize
    opt.device = device
    print(opt)
    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )
    branch_name = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    print("Current git commit:", commit_hash)
    print("Current git branch:", branch_name)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print("===> Loading dataset(s)")
    dataset, encoder_dim = get_dataset(opt)
    whole_train_set, whole_training_data_loader, train_set, whole_test_set = get_splits(
        opt, dataset
    )
    print(whole_train_set)
    print(whole_test_set)

    print("===> Building model")
    model, optimizer, scheduler, criterion = get_model(opt, encoder_dim, device)
    print("Model:", model)
    print("Optimizer:", optimizer)
    print("Scheduler:", scheduler)
    print("Criterion:", criterion)

    if opt.arch in {"seqvlad", "stformer","adapt_stformer"} and not opt.resume:
        init_kwargs = {
            "args": opt,
            "queries_num": whole_train_set.dbStruct.numQ,
            "database_num": whole_train_set.dbStruct.numDb,
            "cluster_ds": whole_train_set,
            "encoder": model.module.encoder,
        }
        model.module.aggregator.modified_initialize_seqvlad_layer(**init_kwargs)


    unique_string = (
        f"{opt.arch}_{opt.backbone}_{datetime.now().strftime('%b%d_%H-%M-%S')}_{opt.dataset}"
    )

    # add flag if special mode is active
    if opt.no_recurrent:
        unique_string += "_norecurrent"
    elif opt.use_temporal_encoder:
        unique_string += "_temporal_encoder"
    writer = None

    if opt.mode.lower() == "test":
        print("===> Running evaluation step")
        epoch = 1
        print("len whole test set: ",len(whole_test_set))

        if opt.pca_dim != 0:
            opt.mode = "train"
            whole_train_set, whole_training_data_loader, _ , _ = get_splits(
                opt, dataset
            )
            opt.mode = "test"
            print("whole_training_data_loader: ",whole_training_data_loader)
            pca_model = compute_pca(opt, model, device, train_set, whole_train_set, whole_training_data_loader)
        else:
            pca_model = None

        recallsOrDesc, dbEmb, qEmb, rAtL, preds = test(
            opt,
            model,
            encoder_dim,
            device,
            whole_test_set,
            writer,
            epoch,
            extract_noEval=opt.extractOnly,
            pca_model=pca_model
        )
        if opt.resultsPath is not None:
            if not exists(opt.resultsPath):
                makedirs(opt.resultsPath)
            if opt.extractOnly:
                gt = whole_test_set.get_positives()
                numDb = whole_test_set.dbStruct.numDb
                np.savez(
                    join(opt.resultsPath, unique_string),
                    dbDesc=recallsOrDesc[:numDb],
                    qDesc=recallsOrDesc[numDb:],
                    gt=gt,
                )
            else:
                np.savez(
                    join(opt.resultsPath, unique_string),
                    args=opt.__dict__,
                    recalls=recallsOrDesc,
                    dbEmb=dbEmb,
                    qEmb=qEmb,
                    rAtL=rAtL,
                    preds=preds,
                )

    elif opt.mode.lower() == "train":
        print("===> Training model")
        writer = SummaryWriter(log_dir=join(opt.runsPath, unique_string))
        train_set.cache = join(
            opt.cachePath,
            train_set.whichSet + "_feat_cache_{}.hdf5".format(unique_string),
        )
        if not exists(opt.cachePath):
            makedirs(opt.cachePath)

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()


        if opt.ckpt == "external_pretrain":  
            opt.savePath = join(logdir, "checkpoints")
            makedirs(opt.savePath)
        elif opt.resume:  # If resume is defined, always use it for savePath
            opt.savePath = join(opt.resume, "checkpoints")
            if not exists(opt.savePath):
                makedirs(opt.savePath)
        else:
            opt.savePath = join(logdir, "checkpoints")
            makedirs(opt.savePath)

        opt.device = opt.nocuda
        with open(join(opt.savePath, "flags.json"), "w") as f:
            f.write(json.dumps({k: v for k, v in vars(opt).items()}))

        print("===> Saving state to:", opt.savePath)

        not_improved = 0
        best_score = 0
        all_loss = []
        test_recalls = [[], [], []]  # For recall@1, recall@5, recall@10
        train_recall_at_1 = []
        for epoch in range(opt.start_epoch + 1, opt.nEpochs + 1):
            if ((epoch - 1) % opt.evalEvery) == 0:
                test_set_recalls = test(
                    opt, model, encoder_dim, device, whole_test_set, writer, epoch
                )[0]
                if opt.evalTrain.lower() == "true":
                    train_set_recalls = test(
                        opt, model, encoder_dim, device, whole_train_set, writer, epoch
                    )[0]
                else:
                    train_set_recalls = {1: 0}
                    
                test_recalls[0].append(test_set_recalls[1])
                test_recalls[1].append(test_set_recalls[5])
                test_recalls[2].append(test_set_recalls[10])
                train_recall_at_1.append(train_set_recalls[1])

                is_best = test_set_recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = test_set_recalls[5]
                else:
                    not_improved += 1

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "recalls": test_set_recalls,
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        "parallel": False,
                    },
                    is_best,
                )

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print(
                        "Performance did not improve for",
                        opt.patience,
                        "epochs. Stopping.",
                    )
                    break

            epoch_train_loss = train(
                opt,
                model,
                encoder_dim,
                device,
                dataset,
                criterion,
                optimizer,
                train_set,
                whole_train_set,
                whole_training_data_loader,
                epoch,
                writer,
            )
            all_loss.append(epoch_train_loss)

            if opt.optim.upper() == "SGD":
                scheduler.step(epoch)

            start_epoch = opt.start_epoch
            curr_epoch = epoch

            plot_graphs(
                opt=opt,
                test_recalls=test_recalls,
                train_recall_at_1=train_recall_at_1,
                all_loss=all_loss,
                start_epoch=start_epoch,
                curr_epoch=curr_epoch,
                unique_string=unique_string,
            )

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
