import numpy as np
from os.path import join
import torchvision.transforms as transforms
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import datetime
import re

prefix_data = "./data/"

base_transform = transforms.Compose(
    [
        transforms.Resize([384, 384]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def path_to_pil_img(paths):
    imgs = []
    for path in paths:
        imgs.append(Image.open(path).convert("RGB"))
    return imgs

def transform_and_save(paths, name, split_type):
    images = path_to_pil_img(paths)
    print(f"Transforming {split_type} images")
    ft_list = []
    for img in tqdm(images, desc=f"Transforming {split_type} images"):
        transformed_img = base_transform(img).numpy()
        ft_list.append(transformed_img)
    ft = np.array(ft_list)
    print(f"Saving images for {split_type}")
    np.save(
        os.path.join(
            f"{name}_raw_{split_type}_{base_transform.transforms[0].size[0]}.npy"
        ),
        ft,
    )
    print("Current time: ", datetime.datetime.now())
    del ft, images


def sort_files(directory):
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

    sorted_files = [os.path.join(directory, info[3]) for info in sorted_info]

    return sorted_files


def convert_to_image(train_test, db_folder, query_folder):
    db_paths = sort_files(db_folder)
    query_paths = sort_files(query_folder)

    print("db_paths: ",db_paths)
    print("query_paths: ",query_paths)

    transform_and_save(query_paths, f"oxford_hard_{train_test}", "query")
    transform_and_save(db_paths, f"oxford_hard_{train_test}", "database")

convert_to_image(train_test = "train", 
                db_folder = "oxford_hard_train_rgb_cut_1m/2014-11-18-13-20-12", 
                query_folder = "oxford_hard_train_rgb_cut_1m/2014-12-16-18-44-24")

convert_to_image(train_test = "test", 
                db_folder = "oxford_hard_test_rgb_cut_1m/2014-12-16-09-14-09", 
                query_folder = "oxford_hard_test_rgb_cut_1m/2014-12-17-18-18-43")
