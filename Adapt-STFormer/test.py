import torch
from torch.utils.data import DataLoader
import faiss
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import numpy as np
import time
from PIL import Image 
from svpr.utils.utils import getRecallAtN,create_dummy_input, save_valid_predictions
import matplotlib.pyplot as plt
import os
# from thop import profile
from ptflops import get_model_complexity_info


def test(
    opt, model, encoder_dim, device, eval_set, writer, epoch=0, extract_noEval=False, pca_model=None
):

    test_data_loader = DataLoader(
        dataset=eval_set,
        num_workers=opt.threads,
        batch_size=opt.cacheBatchSize,
        shuffle=False,
        pin_memory=not opt.nocuda,
    )

    model.to(device)
    model.eval()


    #############################################################
    # Model Statistics Calculation (FLOPs, Memory, Parameters)
    ############################################################
    if opt.mode == "test": #NOTE: what happens if we remove the conditional???
        dummy_input = create_dummy_input(opt)
        flops_value = None

        if dummy_input is not None:
            dummy_input = dummy_input.to(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            with torch.no_grad():
                if opt.arch == "seqnet":
                    macs, params = get_model_complexity_info(
                        model.module.pool,
                        input_res=tuple(dummy_input.shape[1:]),
                        input_constructor=lambda _: (dummy_input,),
                        as_strings=False,
                        print_per_layer_stat=False,
                    )
                else:
                    macs, params = get_model_complexity_info(
                        model,
                        (1,),
                        input_constructor=lambda _: {'x': dummy_input},
                        as_strings=False,
                        print_per_layer_stat=False,
                    )
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                if opt.arch == "seqnet":
                    _ = model.module.pool(dummy_input)
                else:
                    _ = model(dummy_input)  # Single forward pass

                gflops = (macs * 2) / 1e9
                peak_mem = torch.cuda.max_memory_allocated(device)

                print(f"{opt.arch} GFLOPs: {gflops:.2f}")
                print(f"{opt.arch} Peak Memory: {peak_mem / 1e6:.2f} MB")


    with torch.no_grad():

        print("====> Extracting Features")
        pool_size = opt.outDims

        print("pool_size: ", pool_size)

        dbFeat = torch.empty((len(eval_set), pool_size), device=device)
        print("dbFeat.shape: ", dbFeat.shape)
        durs_batch = []

        for iteration, (input, indices) in tqdm(
            enumerate(test_data_loader, 1), total=len(test_data_loader) - 1, leave=False
        ):
            input = input.float().to(device)

            seq_encoding = []
            
            torch.cuda.synchronize()
            t1 = time.time()

            if opt.arch in {"stformer","seqvlad","jist","casenet", "adapt_stformer"}:
                input = input.contiguous().view(-1, 3, opt.img_shape[0], opt.img_shape[1])

                torch.cuda.synchronize()
                t2 = time.time()

                seq_encoding = model(input)
                seq_encoding_tensor = seq_encoding

            elif opt.arch == "seqnet":
                seq_encoding = model.module.pool(input)
                seq_encoding_tensor = seq_encoding

            else:
                raise Exception("model not defined!")
            # import pdb; pdb.set_trace()
            dbFeat[indices, :] = seq_encoding_tensor    

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print(
                    "==> Batch ({}/{})".format(iteration, len(test_data_loader)),
                    flush=True,
                )

            torch.cuda.synchronize()
            # print(f"{opt.arch} durs resulting time: ",time.time() - t1)

            durs_batch.append(time.time() - t1)
            del input

    del test_data_loader
    print(f"{opt.arch} Average batch sequence extraction time:", np.mean(durs_batch), np.std(durs_batch))    
    # Divided by sampling rate to reflect skip_rate
    print("eval_set.dbStruct.numDb: ",eval_set.dbStruct.numDb)
    print("len dbFeat: ",len(dbFeat))
    qFeat = dbFeat[eval_set.dbStruct.numDb:]
    dbFeat = dbFeat[:eval_set.dbStruct.numDb]

    print(dbFeat.shape, qFeat.shape)

    if pca_model is not None:
        print(f"applying PCA transform to {opt.pca_dim}: ")
        torch.cuda.synchronize()
        t_q0 = time.time()
        qFeat_np = pca_model.transform(qFeat.detach().cpu().numpy())
        t_q1 = time.time()
        pca_per_query_time = (t_q1 - t_q0) / qFeat.shape[0]
        dbFeat_np = pca_model.transform(dbFeat.detach().cpu().numpy())
        print("qFeat_np shape: ",qFeat_np.shape)
        print("dbFeat_np shape: ",dbFeat_np.shape)
    else:
        qFeat_np = qFeat.detach().cpu().numpy().astype("float32")
        dbFeat_np = dbFeat.detach().cpu().numpy().astype("float32")

    db_emb, q_emb = None, None
    if extract_noEval:
        return np.vstack([dbFeat_np, qFeat_np]), db_emb, q_emb, None, None

    n_values = [1, 5, 10, 20, 100]

    print("====> Building faiss index")
    # predictions = np.random.randint(0, 100, size=(n_queries, n_values))
    if opt.pca_dim != 0:
        faiss_index = faiss.IndexFlatL2(opt.pca_dim)
    else:
        faiss_index = faiss.IndexFlatL2(pool_size)

    faiss_index.add(dbFeat_np)
    t_search0 = time.time()
    distances, predictions = faiss_index.search(qFeat_np, max(n_values))
    faiss_search_time = time.time() - t_search0        
    print("faiss_search_time: ",faiss_search_time)
    per_query_search_time = faiss_search_time / max(1, qFeat_np.shape[0])
    bestDists = distances[:, 0]

    print("====> Calculating recall @ N")

    print("eval_set length: ", len(eval_set))  # This is correct.
    # for each query get those within threshold distance
    # gt, gtDists = eval_set.get_positives(retDists=True)
    gt = eval_set.get_positives()
    print("gt_length: ", len(gt))
    print("gt type: ", type(gt))
    print("gt index 0: ",gt[0:10])
    print("pred: ", predictions)
    gtDistsMat = cdist(eval_set.dbStruct.utmDb, eval_set.dbStruct.utmQ)

    # compute recall for different loc radii
    rAtL = []

    gt = gt[: gtDistsMat.shape[1]]
    recall_at_n = getRecallAtN(n_values, predictions, gt, opt)

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if writer is not None:
            writer.add_scalar("Val/Recall@" + str(n), recall_at_n[i], epoch)

    if opt.save_incorrect:
        pass

    if opt.vpr_application:
        seq_extract_times_q = np.asarray(durs_batch, dtype=float)[eval_set.dbStruct.numDb:]
        seq_match_times = seq_extract_times_q + per_query_search_time + (pca_per_query_time if pca_model is not None else 0.0)
        save_valid_predictions(
            predictions=predictions,
            seq_match_times=seq_match_times,
            eval_set=eval_set,
            gt=gt,  # Pass the ground truth to the function
            inter_query_s=0.033,
            opt=opt
        )

    return recalls, db_emb, q_emb, rAtL, predictions