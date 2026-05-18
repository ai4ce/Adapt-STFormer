from itertools import product
from os.path import join

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from datasets import Dataset
from tqdm import tqdm


prefix_data = "./data/"


def path_to_pil_img(paths):
    imgs = []
    for path in paths:
        imgs.append(Image.open(path).convert("RGB"))
        print(path)
    return imgs


base_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_dataset(opt):
    if "nordland" in opt.dataset.lower():
        dataset = Dataset(
            "nordland",
            "nordland_train_d-40_d2-10.db",
            "nordland_test_d-1_d2-1.db",
            "nordland_test_d-1_d2-1.db",
            opt,
        )  # train, test, val structs

        if opt.dataset.lower() == "nordland-sw":
            train_ref, train_qry = "summer", "winter"
            val_ref, val_qry = "spring", "fall"
        elif opt.dataset.lower() == "nordland-sf":
            train_ref, train_qry = "spring", "fall"
            val_ref, val_qry = "summer", "winter"

        skip_rate = int(opt.skip.lower())

        ft_all = []

        for trav in [train_ref, train_qry, val_ref, val_qry]:
            if opt.descType == "netvlad-pytorch":
                ft = np.load(
                    join("/scratch/yl9727/nordland/", f"nordland-clean-{trav}.npy"),
                    mmap_mode="r",
                )
            elif opt.descType == "raw_224":
                ft = np.load(
                    join("/vast/yl9727/nordland/", f"nordland_raw_{trav}.npy"),
                    mmap_mode="r",
                )
            elif opt.descType == "raw_384":
                ft = np.load(
                    join("/vast/yl9727/nordland/", f"nordland_raw_{trav}_384.npy"),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_378":
                ft = np.load(
                    join("/vast/yl9727/nordland/", f"nordland_raw_{trav}_378.npy"),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_322":
                ft = np.load(
                    join("/vast/yl9727/nordland/", f"nordland_raw_{trav}_322.npy"),
                    mmap_mode="r",
                )

            ft_all.append(ft)

        ft_train_ref, ft_train_qry, ft_val_ref, ft_val_qry = ft_all

        train_len = ft_train_ref.shape[0]
        trainInds = np.arange(0, 15000, skip_rate)
        testInds = train_len + np.arange(15100, 18100, skip_rate)

        dataset.trainInds = [trainInds, trainInds]
        dataset.testInds = [testInds, testInds]
        dataset.valInds = dataset.testInds

        encoder_dim = dataset.loadPreComputedDescriptors(
            ft1=np.vstack([ft_train_ref, ft_val_ref]),
            ft2=np.vstack([ft_train_qry, ft_val_qry]),
        )

    elif "oxford-easy" == opt.dataset.lower():
        dataset = Dataset(
            "oxford2",
            "oxford2_2014-12-16-09-14-09-2014-12-17-18-18-43_train.db",
            "oxford2_2014-11-18-13-20-12-2014-12-16-18-44-24_test.db",
            "oxford2_2014-11-18-13-20-12-2014-12-16-18-44-24_test.db",
            opt,
        )  # train, test, val structs

        ft_all = []
        train_ref, train_qry = "2014-12-16-09-14-09", "2014-12-17-18-18-43"
        val_ref, val_qry = "2014-11-18-13-20-12", "2014-12-16-18-44-24"

        for split_type, trav_type in [
            ("train", "database"),
            ("train", "query"),
            ("test", "database"),
            ("test", "query"),
        ]:

            if opt.descType == "netvlad-pytorch":
                ft = np.load(
                    join("/scratch/yl9727/patch-netvlad/patch-netvlad/patchnetvlad/output_features/", f"oxford_easy_{split_type}_{trav_type}/globalfeats.npy"),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_384":
                ft = np.load(
                    join(
                        "/scratch/yl9727/oxford2/",
                        f"oxford2_raw_{split_type}_{trav_type}_384.npy",
                    ),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_378":
                ft = np.load(
                    join(
                        "/vast/yl9727/oxford2/",
                        f"oxford2_raw_{split_type}_{trav_type}_378.npy",
                    ),
                    mmap_mode="r",
                )
                
            elif opt.descType == "raw_224":
                ft = np.load(
                    join(
                        "/vast/yl9727/oxford2/",
                        f"oxford2_raw_{split_type}_{trav_type}_224.npy",
                    ),
                    mmap_mode="r",
                )
                
            elif opt.descType == "raw_322":
                ft = np.load(
                    join(
                        "/vast/yl9727/oxford2/",
                        f"oxford2_raw_{split_type}_{trav_type}_322.npy",
                    ),
                    mmap_mode="r",
                )

            ft_all.append(ft)

        ft_train_ref, ft_train_qry, ft_val_ref, ft_val_qry = ft_all

        # Calculate lengths for easier reference
        train_ref_len = ft_train_ref.shape[0]
        train_qry_len = ft_train_qry.shape[0]
        val_ref_len = ft_val_ref.shape[0]
        val_qry_len = ft_val_qry.shape[0]

        # Set the indices for training, validation, and test sets
        dataset.trainInds = [np.arange(train_ref_len), np.arange(train_qry_len)]
        dataset.testInds = [
            np.arange(train_ref_len, train_ref_len + val_ref_len),
            np.arange(train_qry_len, train_qry_len + val_qry_len),
        ]
        dataset.valInds = dataset.testInds  # Validation and test sets are the same

        encoder_dim = dataset.loadPreComputedDescriptors(
            ft1=np.vstack([ft_train_ref, ft_val_ref]),
            ft2=np.vstack([ft_train_qry, ft_val_qry]),
        )   

    elif "oxford-hard" == opt.dataset.lower():
        dataset = Dataset(
            opt.dataset,
            "oxford_hard_train.db",
            "oxford_hard_test.db",
            "oxford_hard_test.db",
            opt,
        )  # train, test, val structs
        ft_all = []

        for trav, trav_type in tqdm([
            ("train", "database"),
            ("train", "query"),
            ("test", "database"),
            ("test", "query"),
        ], desc="Loading descriptors"):

            if opt.descType == "netvlad-pytorch":
                ft = np.load(
                    join("/scratch/yl9727/patch-netvlad/patch-netvlad/patchnetvlad/output_features/", f"oxford_hard_{trav}_{trav_type}/globalfeats.npy"),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_224":
                ft = np.load(
                    join(
                        "/vast/yl9727/oxford-hard/",
                        f"oxford_hard_{trav}_raw_{trav_type}_224.npy",
                    ),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_378":
                ft = np.load(
                    join(
                        "/vast/yl9727/oxford-hard/", #Scratch has the new set up
                        f"oxford_hard_{trav}_raw_{trav_type}_378.npy",
                    ),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_384":
                ft = np.load(
                    join(
                        "/scratch/yl9727/oxford-hard/",
                        f"oxford_hard_{trav}_raw_{trav_type}_384.npy",
                    ),
                    mmap_mode="r",
                )
                
            elif opt.descType == "raw_322":
                ft = np.load(
                    join(
                        "/scratch/yl9727/oxford-hard/",
                        f"oxford_hard_{trav}_raw_{trav_type}_322.npy",
                    ),
                    mmap_mode="r",
                )
            ft_all.append(ft)

        # Unpack the feature arrays
        ft_train_ref, ft_train_qry, ft_val_ref, ft_val_qry = ft_all

        # Calculate lengths for easier reference
        train_ref_len = ft_train_ref.shape[0]
        train_qry_len = ft_train_qry.shape[0]
        print("train_ref_len: ",train_ref_len)
        print("train_qry_len: ",train_qry_len)
        val_ref_len = ft_val_ref.shape[0]
        val_qry_len = ft_val_qry.shape[0]

        # Set the indices for training, validation, and test sets
        dataset.trainInds = [np.arange(train_ref_len), np.arange(train_qry_len)]
        dataset.testInds = [
            np.arange(train_ref_len, train_ref_len + val_ref_len),
            np.arange(train_qry_len, train_qry_len + val_qry_len),
        ]
        dataset.valInds = dataset.testInds  # Validation and test sets are the same

        encoder_dim = dataset.loadPreComputedDescriptors(
            ft1=np.vstack([ft_train_ref, ft_val_ref]),
            ft2=np.vstack([ft_train_qry, ft_val_qry]),
        )

    elif opt.dataset == "nuscene":
        dataset = Dataset(
            opt.dataset,
            "nuscene_large.db",
            "nuscene_large.db",
            "nuscene_large.db",
            opt,
        ) 

        seqSize_travType_list = [("large", "database"),("large", "query")]
        
        seqBounds_all, ft_all = [], []

        for size, trav_type in seqSize_travType_list:
            if opt.descType == "netvlad-pytorch":
                ft = np.load(
                    join("/scratch/yl9727/patch-netvlad/patch-netvlad/patchnetvlad/output_features/", f"nuscene_{size}_{trav_type}/globalfeats.npy"),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_224":
                ft = np.load(
                    join("/scratch/yl9727/nuscenes/", f"nuscene_{trav_type}_raw_{size}_224.npy"),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_378":
                ft = np.load(
                    join("/scratch/yl9727/nuscenes/", f"nuscene_{trav_type}_raw_{size}_378.npy"),
                    mmap_mode="r",
                )

            elif opt.descType == "raw_384":
                ft = np.load(
                    join("/scratch/yl9727/nuscenes/", f"nuscene_{trav_type}_raw_{size}_384.npy"),
                    mmap_mode="r",
                )
                
            elif opt.descType == "raw_322":
                ft = np.load(
                    join("/scratch/yl9727/nuscenes/", f"nuscene_{trav_type}_raw_{size}_322.npy"),
                    mmap_mode="r",
                )
                
            seqBounds_all.append(np.loadtxt(f"./structFiles/seqBoundsFiles/nuscene_{size}_{trav_type}_seqbounds.txt",int))
            ft_all.append(ft)

        ft_test_ref, ft_test_qry = ft_all
        sb_test_ref, sb_test_qry = seqBounds_all

        test_ref_len = ft_test_ref.shape[0]
        test_qry_len = ft_test_qry.shape[0]

        # Set the indices for training, validation, and test sets
        dataset.testInds = [
            np.arange(test_ref_len),
            np.arange(test_qry_len),
        ]
        dataset.valInds = dataset.testInds  # Validation and test sets are the same
        dataset.trainInds = dataset.testInds

        encoder_dim = dataset.loadPreComputedDescriptors(
            ft1=np.vstack([ft_test_ref]),
            ft2=np.vstack([ft_test_qry]),
            seqBounds = [np.vstack([sb_test_ref]),np.vstack([sb_test_qry])],
        )

    elif "oxford-nuscene-pairing" == opt.dataset.lower():
        dataset = Dataset(
            "oxford-nuscene-pairing",
            "oxford2_2014-12-16-09-14-09-2014-12-17-18-18-43_train.db",
            "nuscene_small.db",
            "nuscene_small.db",
            opt,
        )
        
        # Load Oxford 2 features for training
        ft_train_ref = np.load(
            "/scratch/yl9727/oxford2/oxford2_raw_train_database_384.npy",
            mmap_mode="r"
        )
        ft_train_qry = np.load(
            "/scratch/yl9727/oxford2/oxford2_raw_train_query_384.npy",
            mmap_mode="r"
        )
        
        # Load NUSCENES features for testing
        ft_val_ref = np.load(
            "/scratch/yl9727/nuscenes/nuscene_database_raw_small_384.npy",
            mmap_mode="r"
        )
        ft_val_qry = np.load(
            "/scratch/yl9727/nuscenes/nuscene_query_raw_small_384.npy",
            mmap_mode="r"
        )

        train_ref_len = ft_train_ref.shape[0]
        train_qry_len = ft_train_qry.shape[0]
        print("train_ref_len: ",train_ref_len)
        print("train_qry_len: ",train_qry_len)
        val_ref_len = ft_val_ref.shape[0]
        val_qry_len = ft_val_qry.shape[0]

        # Set the indices for training, validation, and test sets
        dataset.trainInds = [np.arange(train_ref_len), np.arange(train_qry_len)]
        dataset.testInds = [
            np.arange(train_ref_len, train_ref_len + val_ref_len),
            np.arange(train_qry_len, train_qry_len + val_qry_len),
        ]
        dataset.valInds = dataset.testInds  # Validation and test sets are the same

        encoder_dim = dataset.loadPreComputedDescriptors(
            ft1=np.vstack([ft_train_ref, ft_val_ref]),
            ft2=np.vstack([ft_train_qry, ft_val_qry]),
        )
        
    else:
        raise Exception("Unknown dataset")
    
    print("dataset.trainInds: ", dataset.trainInds)
    print("dataset.valInds: ", dataset.valInds)
    print("dataset.testInds: ", dataset.testInds)

    return dataset, encoder_dim


def get_splits(opt, dataset):
    whole_train_set, whole_training_data_loader, train_set, whole_test_set = (
        None,
        None,
        None,
        None,
    )
    if opt.mode.lower() == "train":
        whole_train_set = dataset.get_whole_training_set(opt=opt)
        whole_training_data_loader = DataLoader(
            dataset=whole_train_set,
            num_workers=opt.threads,
            batch_size=opt.cacheBatchSize,
            shuffle=False,
            pin_memory=not opt.nocuda,
        )

        train_set = dataset.get_training_query_set(margin=opt.margin, opt=opt)

        print("====> Training whole set:", len(whole_train_set))
        print("====> Training query set:", len(train_set))
        whole_test_set = dataset.get_whole_val_set(opt=opt)
        print("===> Evaluating on val set, query count:", whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == "test":
        if opt.split.lower() == "test":
            whole_test_set = dataset.get_whole_test_set(opt=opt)
            print("===> Evaluating on test set")
        elif opt.split.lower() == "train":
            whole_test_set = dataset.get_whole_training_set(opt=opt)
            print("===> Evaluating on train set")
        elif opt.split.lower() in ["val"]:
            whole_test_set = dataset.get_whole_val_set(opt=opt)
            print("===> Evaluating on val set")
        else:
            raise ValueError("Unknown dataset split: " + opt.split)
        print("====> Query count:", whole_test_set.dbStruct.numQ)

    return whole_train_set, whole_training_data_loader, train_set, whole_test_set
