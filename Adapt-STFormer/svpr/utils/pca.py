import torch
from tqdm import tqdm
import numpy as np
from os.path import join
from os import remove
import h5py
from sklearn.decomposition import PCA
from joblib import dump, load

def compute_pca(
    opt,
    model,
    device,
    train_set,
    whole_train_set,
    whole_training_data_loader,
):
    print("====> Building Cache (in-memory)")

    model.eval()
    all_features = []  # list to hold feature batches

    pool_size = opt.outDims

    with torch.no_grad():
        for iteration, (input, indices) in tqdm(
            enumerate(whole_training_data_loader, 1),
            total=len(whole_training_data_loader) - 1,
            leave=False,
        ):
            if opt.arch == "adapt_stformer":
                concat_input = input
                seq_encoding = []
                for input_part in concat_input:
                    seq_encoding.append(model(x=input_part, device=device))
                seq_encoding_tensor = torch.stack(seq_encoding).squeeze(-1).squeeze(1)
                all_features.append(seq_encoding_tensor.detach().cpu())

                del seq_encoding, seq_encoding_tensor, concat_input, input

            elif opt.arch in {"stformer", "seqvlad", "seqnet_modified","casenet"}:
                input = input.float().to(device)
                input = input.contiguous().view(-1, 3, opt.img_shape[0], opt.img_shape[1])
                seq_encoding = model(input)
                all_features.append(seq_encoding.detach().cpu())

                del input, seq_encoding

            else:
                raise Exception("invalid model")

            torch.cuda.empty_cache()

    # After all batches collected
    all_features = torch.cat(all_features, dim=0).numpy()  # (N, pool_size)

    print(f"Collected {all_features.shape[0]} feature vectors. Fitting PCA model...")

    pca_model = PCA(n_components=opt.pca_dim, whiten=False)
    pca_model.fit(all_features)

    return pca_model
