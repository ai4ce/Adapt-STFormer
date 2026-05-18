import numpy as np
from os.path import join
import torchvision.transforms as transforms
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import datetime
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

def convert_to_image(split_type, db_folder, query_folder):
    db_paths = sorted(glob(join(db_folder, "*.jpg")))
    query_paths = sorted(glob(join(query_folder, "*.jpg")))

    for path in db_paths:
        print(path)
    for path in query_paths:
        print(path)

    # transform_and_save(query_paths, "", split_type)
    # transform_and_save(db_paths, "", split_type)


# convert_to_image("", "", "")

