import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom

def process_isic_dataset():
    # 定义路径
    base_path = "../data/ISIC/raw/ISIC 2018 Segmentation"
    output_path = "../data/ISIC/Foreslices"
    
    # 创建输出目录结构
    os.makedirs(os.path.join(output_path, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "label"), exist_ok=True)
    
    # 定义数据集类型和对应的文件夹
    datasets = {
        "train": {
            "input": "ISIC2018_Task1-2_Training_Input",
            "gt": "ISIC2018_Task1_Training_GroundTruth"
        },
        "val": {
            "input": "ISIC2018_Task1-2_Validation_Input",
            "gt": "ISIC2018_Task1_Validation_GroundTruth"
        },
        "test": {
            "input": "ISIC2018_Task1-2_Test_Input",
            "gt": "ISIC2018_Task1_Test_GroundTruth"
        }
    }
    
    global_counter = 0
    
    for dataset_type in ["train", "val", "test"]:
        input_dir = os.path.join(base_path, datasets[dataset_type]["input"])
        gt_dir = os.path.join(base_path, datasets[dataset_type]["gt"])
        
        # 获取排序后的文件列表
        input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')])
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png') or f.endswith('.jpg')])
        
        # 确保文件一一对应
        assert len(input_files) == len(gt_files), f"文件数量不匹配: {dataset_type}"
        
        for i, (input_file, gt_file) in tqdm(enumerate(zip(input_files, gt_files)), 
                                           total=len(input_files),
                                           desc=f"Processing {dataset_type} set"):
            # 确保文件名一致
            assert input_file.replace(".jpg", "_segmentation.png") == gt_file, f"文件名不匹配: {input_file} vs {gt_file}"
            
            # 读取图像
            input_img = np.array(Image.open(os.path.join(input_dir, input_file)))
            gt_img = np.array(Image.open(os.path.join(gt_dir, gt_file)))
            
            # 归一化标签图像 (0-1范围)
            gt_img = gt_img.astype(np.float32) / 255.0
            gt_img = (gt_img > 0.5).astype(np.float32)  # 二值化
            
            # 生成新文件名
            new_name = f"Case{global_counter:03d}_Slice000.pkl"


            gt_img = np.expand_dims(gt_img, axis=-1)  
            input_img = zoom(input_img, (512 / input_img.shape[0], 512 / input_img.shape[1], 1), order=3)
            gt_img = zoom(gt_img, (512 / gt_img.shape[0], 512 / gt_img.shape[1], 1), order=0)
            
            # 保存图像和标签
            with open(os.path.join(output_path, "image", new_name), 'wb') as f:
                pickle.dump(input_img, f)
            
            with open(os.path.join(output_path, "label", new_name), 'wb') as f:
                pickle.dump(gt_img, f)
            
            global_counter += 1

    print(f"处理完成! 共处理了 {global_counter} 个样本。")
    print(f"数据保存在: {output_path}")

if __name__ == "__main__":
    process_isic_dataset()