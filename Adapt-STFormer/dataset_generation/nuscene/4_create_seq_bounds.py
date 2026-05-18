import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_timestamp_differences(folder_path):
    # get a list of all timestamps
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
    timestamps = [int(os.path.basename(path).split('.')[0]) for path in image_paths]

    # sort the timestamps and calculate the differences
    sorted_timestamps = sorted(timestamps)
    differences = np.diff(sorted_timestamps)

    # plot the histogram distribution of the differences
    # ignore the largest value when plotting
    print("differences: ",differences)
    print("mean: ", np.mean(differences))
    print("median: ", np.median(differences))
    print("min: ", np.min(differences))
    print("max: ", np.max(differences))
    values, counts = np.unique(differences, return_counts=True)
    print("most common diff: ", values[np.argmax(counts)])
    # find the indexes with significant gaps
    significant_gaps = np.where(differences > 1000000)[0] + 1

    # output the sequence bounds to a file
    with open(os.path.join(f'{folder_path}_unformated_bounds.txt'), 'w') as f:
        for idx in significant_gaps:
            f.write(str(idx) + '\n')

    with open(os.path.join(f'{folder_path}_seqbounds.txt'), 'w') as f:
        start = 0
        for idx in significant_gaps:
            for _ in range(start, idx):
                f.write(f"{start} {idx}\n")
            start = idx  # Update start for the next sequence
        for _ in range(start, len(image_paths)):
            f.write(f"{start} {len(image_paths)}\n")
    

analyze_timestamp_differences("nuscene_large_database")
analyze_timestamp_differences("nuscene_large_query")
analyze_timestamp_differences("nuscene_small_database")
analyze_timestamp_differences("nuscene_small_query")