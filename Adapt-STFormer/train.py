import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import numpy as np
import os
from os.path import join
from os import remove
import h5py
from math import ceil
from svpr.utils.utils import print_free_memory
from get_models import disable_stochastic
import gc

def train(
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
):
    
    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging
    print("In the training function")

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        start_indices = np.arange(0, len(train_set), opt.cacheRefreshRate)
        # TODO randomise the arange before splitting?
        subsetIdx = [start_indices[i : i + 1] for i in range(len(start_indices))]
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    image_encodings_list = []

    for subIter in range(subsetN):
        print("====> Building Cache")
        # how many images in batch
        model.eval()
        with h5py.File(train_set.cache, mode="w") as h5:
            pool_size = opt.outDims
            h5feat = h5.create_dataset(
                "features", [len(whole_train_set), pool_size], dtype=np.float32
            )  # empty creation of a dataset
            print(
                "len(whole_training_data_loader)-1, :",
                len(whole_training_data_loader) - 1,
            )
            with torch.no_grad():
                for iteration, (input, indices) in tqdm(
                    enumerate(whole_training_data_loader, 1),
                    total=len(whole_training_data_loader) - 1,
                    leave=False,
                ):  # Where does whole train loader come from?
                        
                    if opt.arch in {"stformer","seqvlad","seqnet_modified","adapt_stformer"}:
                        input = (input).float().to(device)
                        input = input.contiguous().view(-1, 3, opt.img_shape[0], opt.img_shape[1])
                        seq_encoding = model(input)
                        h5feat[indices.detach().numpy(), :] = (
                            seq_encoding.detach().cpu().numpy()
                        )
                        del input, seq_encoding

                    elif opt.arch == "seqnet":
                        image_encoding = (input).float().to(device)
                        seq_encoding = model.module.pool(image_encoding)
                        h5feat[indices.detach().numpy(), :] = seq_encoding.detach().cpu().numpy()
                        del input, image_encoding, seq_encoding

                    else:
                        raise Exception("invalid model")
                    torch.cuda.empty_cache()

        print("In the sub Iter")

        print("image_encodings_list length: ", len(image_encodings_list))
        subset_indices = np.array(subsetIdx[subIter])
        print("subset_indices: ", subset_indices)
        sub_train_set = Subset(dataset=train_set, indices=subset_indices)

        training_data_loader = DataLoader(
            dataset=sub_train_set,
            num_workers=opt.threads,
            batch_size=opt.batchSize,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            pin_memory=not opt.nocuda,
        )

        model.train()
        for iteration, (query, positives, negatives, negCounts, indices) in tqdm(
            enumerate(training_data_loader, startIter),
            total=len(training_data_loader),
            desc="Training Loop",
            leave=False,
        ):
            loss = 0

            if query is None:
                continue  # In case we get an empty batch

            B = query.shape[0]  # Batch size
            nNeg = torch.sum(negCounts)  # Total negatives

            if opt.arch in {"stformer","seqvlad","seqnet_modified","adapt_stformer"}:
                input = torch.cat([query, positives, negatives]).float().to(device)
                input = input.contiguous().view(-1, 3, opt.img_shape[0], opt.img_shape[1])
                seq_encoding = model(input)
                seq_encoding_tensor = seq_encoding

            elif opt.arch == "seqnet":
                input = torch.cat([query,positives,negatives]).float()
                input = input.to(device)
                seq_encoding = model.module.pool(input)
                seq_encoding_tensor = seq_encoding

            seqQ, seqP, seqN = torch.split(seq_encoding_tensor, [B, B, nNeg])

            optimizer.zero_grad()

            # Loss calculation
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(
                        seqQ[i : i + 1], seqP[i : i + 1], seqN[negIx : negIx + 1]
                    )

            loss /= nNeg.float().to(device)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Store batch loss
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Print out timing information
            if iteration % 50 == 0 or nBatches <= 10:
                writer.add_scalar(
                    "Train/Loss", batch_loss, ((epoch - 1) * nBatches) + iteration
                )
                writer.add_scalar(
                    "Train/nNeg", nNeg, ((epoch - 1) * nBatches) + iteration
                )

            # Cleanup
            del (
                seq_encoding_tensor,
                seq_encoding,
                seqQ,
                seqP,
                seqN,
                query,
                positives,
                negatives,
                input
            )

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache)  # delete HDF5 cache

    avg_loss = epoch_loss / nBatches
    print("model.training: ",model.training)
    print(
        "===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True
    )
    writer.add_scalar("Train/AvgLoss", avg_loss, epoch)

    return avg_loss
