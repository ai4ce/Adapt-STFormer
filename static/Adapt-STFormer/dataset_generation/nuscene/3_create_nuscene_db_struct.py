from scipy.io import savemat
import numpy as np
import glob
import os
from collections import namedtuple
import pandas as pd

import numpy as np
from scipy.io import savemat
import json

def save_db_struct(path, db_struct):
    assert db_struct.numDb == len(db_struct.dbImage)
    assert db_struct.numQ == len(db_struct.qImage)

    inner_dict = {
        "whichSet": db_struct.whichSet,
        "dbImageFns": np.array(db_struct.dbImage, dtype=object).reshape(-1, 1), 
        "qImageFns": np.array(db_struct.qImage, dtype=object).reshape(-1, 1),   
        "numImages": db_struct.numDb,
        "numQueries": db_struct.numQ,
        "posDistThr": db_struct.posDistThr,
        "posDistSqThr": db_struct.posDistSqThr,
    }

    if db_struct.dataset is not None:
        inner_dict["dataset"] = db_struct.dataset

    if db_struct.nonTrivPosDistSqThr is not None:
        inner_dict["nonTrivPosDistSqThr"] = db_struct.nonTrivPosDistSqThr

    if db_struct.utmDb is not None and db_struct.utmQ is not None:
        assert db_struct.numDb == len(db_struct.utmDb)
        assert db_struct.numQ == len(db_struct.utmQ)
        inner_dict["utmDb"] = db_struct.utmDb.T
        inner_dict["utmQ"] = db_struct.utmQ.T

    if db_struct.rotationDb is not None and db_struct.rotationQ is not None:
        assert db_struct.numDb == len(db_struct.rotationDb)
        assert db_struct.numQ == len(db_struct.rotationQ)
        inner_dict["rotationDb"] = db_struct.rotationDb.T
        inner_dict["rotationQ"] = db_struct.rotationQ.T

    if db_struct.dbTimeStamp is not None and db_struct.qTimeStamp is not None:
        inner_dict["dbTimeStamp"] = db_struct.dbTimeStamp.astype(np.float64)
        inner_dict["qTimeStamp"] = db_struct.qTimeStamp.astype(np.float64)

    savemat(path, {"dbStruct": inner_dict})


dbStruct = namedtuple(
    "dbStruct",
    [
        "whichSet",
        "dataset",
        "dbImage",
        "utmDb",
        "qImage",
        "utmQ",
        "numDb",
        "numQ",
        "posDistThr",
        "posDistSqThr",
        "nonTrivPosDistSqThr",
        "dbTimeStamp",
        "qTimeStamp",
        "gpsDb",
        "gpsQ",
        "rotationQ",
        "rotationDb",
    ],
)

def match_utm(image_paths, utm_file_path):
    # Load ego_pose data
    with open(utm_file_path) as f:
        ego_pose = json.load(f)
    
    # Create dictionaries to map timestamps to UTM coordinates and rotations
    timestamp_to_utm = {item["timestamp"]: (item["translation"][0], item["translation"][1]) for item in ego_pose}
    timestamp_to_rotation = {item["timestamp"]: item["rotation"] for item in ego_pose}
    
    matched_utm = []
    matched_rotation = []
    for image_path in image_paths:
        timestamp = int(image_path.split('/')[-1].split('.')[0])
        if timestamp in timestamp_to_utm and timestamp in timestamp_to_rotation:
            x_pos, y_pos = timestamp_to_utm[timestamp]
            rotation = timestamp_to_rotation[timestamp]
            matched_utm.append([x_pos, y_pos])
            matched_rotation.append(rotation)
        else:
            raise Exception(f"Could not find utm or rotation for image: {image_path}")
    
    return np.array(matched_utm), np.array(matched_rotation)
if __name__ == "__main__":
    # split_type = "small"
    # dataset_string = f"nuscene_small"
    # database_path, query_path = "nuscene_small_database", "nuscene_small_query"
    # utm_file = "v1.0-test_meta/v1.0-test/ego_pose.json"
    # dbImage, qImage = sorted(glob.glob(os.path.join(database_path, "*.jpg"))), sorted(glob.glob(os.path.join(query_path, "*.jpg")))

    split_type = "large"
    dataset_string = f"nuscene_large"
    database_path, query_path = "nuscene_large_database", "nuscene_large_query"
    utm_file = "v1.0-trainval_meta/v1.0-trainval/ego_pose.json"
    dbImage, qImage = sorted(glob.glob(os.path.join(database_path, "*.jpg"))), sorted(glob.glob(os.path.join(query_path, "*.jpg")))

    utmDb, rotationDb = match_utm(dbImage, utm_file)
    utmQ, rotationQ = match_utm(qImage, utm_file)

    db_struct_instance = dbStruct(
        whichSet=split_type,
        dataset=dataset_string,
        dbImage=dbImage,
        utmDb=utmDb,
        qImage=qImage,
        utmQ=utmQ,
        numDb=len(dbImage),
        numQ=len(qImage),
        posDistThr=10.0,
        posDistSqThr=100.0,
        nonTrivPosDistSqThr=25.0,
        dbTimeStamp=None,
        qTimeStamp=None,
        gpsDb=None,
        gpsQ=None,
        rotationQ=rotationQ,
        rotationDb=rotationDb
    )

    print(rotationQ,len(rotationQ))
    print(rotationDb,len(rotationDb))

    save_db_struct(f"{dataset_string}.db", db_struct_instance)


