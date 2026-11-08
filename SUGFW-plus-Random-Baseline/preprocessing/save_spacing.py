import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom


# 定义原始数据和目标路径
def get_promise12_spacing():

    RAW_DATA_DIR = "../data/Promise12/raw_data"

    Foreslices_IMAGE_DIR = "../data/Promise12/Foreslices/image"
    Foreslices_LABEL_DIR = "../data/Promise12/Foreslices/label"
    Foreslices_spacing_DIR = "../data/Promise12/Foreslices/spacing"


    for dir_path in [
        Foreslices_IMAGE_DIR,
        Foreslices_LABEL_DIR,
        Foreslices_spacing_DIR,
    ]:
        os.makedirs(dir_path, exist_ok=True)

    global_idx = 0
    alls = ("training_data", "livechallenge_test_data", "test_data")
    a = []
    for s in alls:
        base_dir = os.path.join(RAW_DATA_DIR, s)
        files = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.endswith("mhd") and "seg" not in f
        ]
        files.sort()
        for file in tqdm(files):
            image = sitk.ReadImage(file)
            spacing_path = os.path.join(
                Foreslices_spacing_DIR,
                f"Case{global_idx:03d}_spacing.pkl",
            )
            with open(spacing_path, "wb") as f:
                pickle.dump(image.GetSpacing(), f)


            

    
            global_idx += 1

def get_liver_spacing():
    Foreslices_IMAGE_DIR = "../data/Liver/Foreslices/image"
    Foreslices_LABEL_DIR = "../data/Liver/Foreslices/label"
    Foreslices_Spacing_DIR = "../data/Liver/Foreslices/spacing"
    os.makedirs(Foreslices_IMAGE_DIR, exist_ok=True)
    os.makedirs(Foreslices_LABEL_DIR, exist_ok=True)
    os.makedirs(Foreslices_Spacing_DIR, exist_ok=True)

    base_dir = "../data/Liver/raw/Task03_Liver/imagesTr"

    for filename in tqdm(
        [os.path.join(base_dir, file) for file in os.listdir(base_dir)], ncols=100
    ):
        file = os.path.basename(filename)
        if file.startswith(".") or not file.endswith(".nii.gz"):
            continue

        idx = int(file.split("_")[1].split(".")[0])

        image = sitk.ReadImage(filename)
        spacing_path = os.path.join(
            Foreslices_Spacing_DIR,
            f"Case{idx:03d}_spacing.pkl",
        )
        with open(spacing_path, "wb") as f:
            pickle.dump(image.GetSpacing(), f)
    
if __name__ == "__main__":
    # get_promise12_spacing()
    get_liver_spacing()


        



