from scipy.io import savemat
import numpy as np
import glob
import os
from collections import namedtuple
import pandas as pd

import numpy as np
from scipy.io import savemat

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
        # inner_dict["gpsQ"] = db_struct.gpsQ.T

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
    ],
)

import os
import re

def match_utm(directory):
    # Get all jpg files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # Function to extract information from filename
    def extract_info(filename):
        # Use regex to extract lat, lon, and timestamp
        match = re.search(r'@([\d.]+)@([\d.]+)@[\w-]+@\d+@(\d+)@\.jpg$', filename)
        if match:
            return float(match.group(1)), float(match.group(2)), int(match.group(3))
        return 0, 0, 0  # Return zeros if no match found (shouldn't happen with consistent naming)

    # Extract info and sort
    image_info = [(*extract_info(f), f) for f in image_files]
    sorted_info = sorted(image_info, key=lambda x: x[2])  # Sort by timestamp

    # Separate into numpy arrays
    latitudes = np.array([info[0] for info in sorted_info])
    longitudes = np.array([info[1] for info in sorted_info])
    sorted_files = [info[3] for info in sorted_info]

    return sorted_files, np.column_stack((latitudes, longitudes))

split_type = "train"
database_time,query_time = "2015-03-17-11-08-44","2014-12-16-18-44-24"

dataset_string = f"oxford-v1.0_{database_time}-{query_time}"

dbImage,utmDb = match_utm(f"oxford1_database_rgb_cut_2m/{database_time}")
qImage,utmQ = match_utm(f"oxford1_query_rgb_cut_2m/{query_time}")

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
)

print(utmDb,len(utmDb))
print(utmQ,len(utmQ))



save_db_struct(f"{dataset_string}_{split_type}.db", db_struct_instance)

