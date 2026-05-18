from sklearn.neighbors import NearestNeighbors
from print_db_struct import parse_db_struct, dbStruct
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import os
import pickle

def quaternion_to_rotation_matrix(q):
    # The Rotation.from_quat() expects quaternions in [x, y, z, w] order by default.
    # If your quaternion is in [w, x, y, z] order, you need to convert it.
    # Here we assume the input is [w, x, y, z]:
    w, x, y, z = q
    # Convert to [x, y, z, w] order for scipy
    q_converted = [x, y, z, w]
    return Rotation.from_quat(q_converted).as_matrix()

def calculate_image_position_and_direction(camera_position, camera_rotation, projection_distance=25.0):
    """
    Calculate the projected image position and the corresponding 2D forward vector.
    
    Parameters:
        camera_position: numpy array of shape (2,) for x,y position.
        camera_rotation: quaternion as list [w, x, y, z].
        projection_distance: distance to project ahead along the camera's forward direction.
                             Use 25.0 for the paper's setting or 0.0 to not project.
    
    Returns:
        pimg: The projected image position (2D), i.e. camera_position plus the forward displacement.
        v: The forward vector (from camera_position to pimg).
    """
    # Convert quaternion to rotation matrix.
    R = quaternion_to_rotation_matrix(camera_rotation)
    # Extract the 2D forward direction (ignore vertical component).
    # Here we assume that the camera's forward direction in its coordinate frame is along the positive y-axis.
    forward_vector = R @ np.array([0, 1, 0])
    forward_vector_2d = forward_vector[:2]  # Use only x and y.
    forward_vector_2d = forward_vector_2d / np.linalg.norm(forward_vector_2d)  # Normalize.
    
    # Calculate the projected image position:
    pimg = camera_position + projection_distance * forward_vector_2d

    # The forward (direction) vector is simply the displacement.
    if projection_distance == 0.0:
        v = forward_vector_2d
    else:
        v = pimg - camera_position

    return pimg, v

def find_positives(utmQ, utmDb, rotationQ, rotationDb, distance_threshold=10, angle_threshold=30, projection_distance=25.0,dataset_name=None):
    """
    For each query, find database images that are considered positive matches.
    A candidate is positive if:
      1. The Euclidean distance between the projected image positions is less than distance_threshold.
      2. The angle between their 2D forward vectors is less than angle_threshold (in degrees).
    
    Parameters:
        utmQ: List (or array) of query positions (2D).
        utmDb: List (or array) of database positions (2D).
        rotationQ: List of query rotation quaternions ([w, x, y, z]).
        rotationDb: List of database rotation quaternions ([w, x, y, z]).
        distance_threshold: Maximum allowed Euclidean distance between projected positions.
        angle_threshold: Maximum allowed angle (in degrees) between the direction vectors.
        projection_distance: Distance to project ahead (e.g., 25 for the paper's method).
        cache_file: Filename for caching results.
    
    Returns:
        positives: A numpy array where each element is a list of indices (in utmDb) that are positives for that query.
    """
    cache_file = f"nuscene_positives_{dataset_name}_distance_threshold.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Computing positives...")
    positives = []
    for i, (q_pos, q_rot) in enumerate(tqdm(zip(utmQ, rotationQ), total=len(utmQ))):
        # Calculate projected position and forward vector for the query.
        q_pimg, q_v = calculate_image_position_and_direction(q_pos, q_rot, projection_distance)
        
        candidates = []
        for j, (db_pos, db_rot) in enumerate(zip(utmDb, rotationDb)):
            # Calculate for each database candidate.
            db_pimg, db_v = calculate_image_position_and_direction(db_pos, db_rot, projection_distance)
            
            # Compute the Euclidean distance between the projected image positions.
            distance = np.linalg.norm(q_pimg - db_pimg)
            if distance < distance_threshold:
                # Compute the angle between the two forward (direction) vectors.
                cos_theta = np.dot(q_v, db_v) / (np.linalg.norm(q_v) * np.linalg.norm(db_v))
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
                
                if angle < angle_threshold:
                    candidates.append(j)
        
        positives.append(candidates)
        
    # Cache the results for faster future computation.
    print(f"Caching results to {cache_file}")
    with open(cache_file, 'wb') as f:
        # Convert each sublist to a numpy array of int64
        positives = [np.array(sublist, dtype=np.int64) for sublist in positives]
        # Create a numpy object array
        positives = np.array(positives, dtype=object)
        print("positives type: ",type(positives))
        print("positives values: ",positives[0:15])
        pickle.dump(positives, f)
    return positives

if __name__ == "__main__":
    # ---------------------------
    # Usage of the functions:
    # ---------------------------
    my_db = parse_db_struct('nuscene_large.db')
    utmQ = my_db.utmQ       # Query positions (2D, e.g., [x, y])
    utmDb = my_db.utmDb     # Database positions (2D)
    rotationQ = my_db.rotationQ   # Query rotations (quaternions: [w, x, y, z])
    rotationDb = my_db.rotationDb # Database rotations (quaternions: [w, x, y, z])
    dataset_name = my_db.dataset

    print("Lengths:")
    print("utmQ:", len(utmQ))
    print("utmDb:", len(utmDb))
    print("rotationQ:", len(rotationQ))
    print("rotationDb:", len(rotationDb))

    # Find positives using the 25m projection. To try without projecting, set projection_distance=0.0.
    projection_distance=0.0
    positives = find_positives(utmQ, utmDb, rotationQ, rotationDb, distance_threshold=2.0, angle_threshold=30, projection_distance=projection_distance,dataset_name=dataset_name)

    # Print summary results.
    num_positives = sum(1 for pos in positives if pos.size > 0)
    print(f"{num_positives} / {len(positives)} queries have positives")

    with open(f'positives_proj_{projection_distance}.txt', 'w') as f:
        for idx, pos in enumerate(positives):
            if pos.size == 0:
                f.write(f"{idx}: No positives\n")
            else:
                f.write(f"{idx}: {', '.join(map(str, pos))}\n")

