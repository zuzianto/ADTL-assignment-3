import os
import csv
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from scipy.ndimage import zoom
from tqdm import tqdm

# === 路径设置 ===
root_dir = "/data/sxl/MA-SAM/data/TN3K/raw/tn3k"
image_dirs = {
    "trainval-image": os.path.join(root_dir, "trainval-image"),
    "trainval-mask": os.path.join(root_dir, "trainval-mask"),
    "test-image": os.path.join(root_dir, "test-image"),
    "test-mask": os.path.join(root_dir, "test-mask"),
}
json_path = os.path.join(root_dir, "tn3k-trainval-fold0.json")

# 输出路径
csv_dir = "/data/sxl/MA-SAM/data/TN3K/Foreslices/splits"
image_out_dir = "/data/sxl/MA-SAM/data/TN3K/Foreslices/image"
label_out_dir = "/data/sxl/MA-SAM/data/TN3K/Foreslices/label"
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(image_out_dir, exist_ok=True)
os.makedirs(label_out_dir, exist_ok=True)

# === 第一步：生成CSV文件 ===
with open(json_path, "r") as f:
    fold_data = json.load(f)


def write_csv_from_ids(image_dir, mask_dir, ids, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_pth", "mask_pth"])
        for id_ in ids:
            image_path = os.path.join(image_dir, f"{id_:04d}.jpg")
            mask_path = os.path.join(mask_dir, f"{id_:04d}.jpg")
            writer.writerow([image_path, mask_path])


write_csv_from_ids(
    image_dirs["trainval-image"],
    image_dirs["trainval-mask"],
    fold_data["train"],
    os.path.join(csv_dir, "train.csv"),
)
write_csv_from_ids(
    image_dirs["trainval-image"],
    image_dirs["trainval-mask"],
    fold_data["val"],
    os.path.join(csv_dir, "valid.csv"),
)

# 生成 test.csv
test_image_dir = image_dirs["test-image"]
test_mask_dir = image_dirs["test-mask"]
test_images = sorted([f for f in os.listdir(test_image_dir) if f.endswith(".jpg")])
with open(os.path.join(csv_dir, "test.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_pth", "mask_pth"])
    for img_name in test_images:
        writer.writerow(
            [
                os.path.join(test_image_dir, img_name),
                os.path.join(test_mask_dir, img_name),
            ]
        )


# === 第二步：重命名为pkl并保存 ===
def normalize_image(img):
    """归一化图像到[0, 1]"""
    img = np.array(img).astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return img


def save_as_pkl(image_path, mask_path, index, image_out_dir, label_out_dir):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_np = np.array(image).astype(np.uint8)
    mask_np = normalize_image(mask)

    image_np = zoom(
        image_np, (512 / image_np.shape[0], 512 / image_np.shape[1], 1), order=3
    )
    mask_np = zoom(mask_np, (512 / mask_np.shape[0], 512 / mask_np.shape[1]), order=0)
    case_name = f"Case{index:03d}_Slice000.pkl"

    with open(os.path.join(image_out_dir, case_name), "wb") as f:
        pickle.dump(image_np, f)

    with open(os.path.join(label_out_dir, case_name), "wb") as f:
        pickle.dump(mask_np, f)


index = 0
for split in ["train", "valid", "test"]:
    df = pd.read_csv(os.path.join(csv_dir, f"{split}.csv"))
    for _, row in tqdm(df.iterrows()):
        save_as_pkl(
            row["image_pth"], row["mask_pth"], index, image_out_dir, label_out_dir
        )
        index += 1

print("✅ 所有文件已处理并保存为pkl格式。")

# === 第三步：生成新的 train.csv, valid.csv, test.csv ===
new_csv_dir = "/data/sxl/MA-SAM/data/TN3K/Foreslices/splits"
new_image_dir = os.path.abspath(image_out_dir)
new_label_dir = os.path.abspath(label_out_dir)

index = 0
for split in ["train", "valid", "test"]:
    df = pd.read_csv(os.path.join(csv_dir, f"{split}.csv"))
    new_rows = []
    for _ in range(len(df)):
        case_name = f"Case{index:03d}_Slice000.pkl"
        image_path = os.path.join(new_image_dir, case_name)
        mask_path = os.path.join(new_label_dir, case_name)
        new_rows.append([image_path, mask_path])
        index += 1
    new_df = pd.DataFrame(new_rows, columns=["image_pth", "mask_pth"])
    new_df.to_csv(os.path.join(new_csv_dir, f"{split}.csv"), index=False)
