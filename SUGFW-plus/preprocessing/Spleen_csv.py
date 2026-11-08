import os
import random
import pandas as pd

# 假设图片和掩码文件夹路径如下
def get_case_num(filename):
    # 例如: Case000_Slice010.pkl
    return int(filename.split('_')[0][4:])

def make_records(file_list):
    records = []
    for img_file in file_list:
        img_path = os.path.join(IMAGE_DIR, img_file)
        mask_path = os.path.join(LABEL_DIR, img_file)
        records.append({'image_pth': img_path, 'mask_pth': mask_path})
    return records

IMAGE_DIR = f'../data/Spleen/Foreslices/image'
LABEL_DIR = f'../data/Spleen/Foreslices/label'
SPLITS_DIR = f'../data/Spleen/Foreslices/splits'

os.makedirs(SPLITS_DIR, exist_ok=True)

# 获取所有图片文件名
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.pkl')])


# 按casenum分组
train_files = []
valid_files = []
test_files = []

case_nums = list(set([get_case_num(f) for f in image_files]))

random.seed(1234)
random.shuffle(case_nums)
train_id = [x for i, x in enumerate(case_nums) if i < 29]
val_id = [x for i, x in enumerate(case_nums) if 29 <= i < 33]
test_id = [x for i, x in enumerate(case_nums) if 33 <= i]

for i, f in enumerate(image_files):
    casenum = get_case_num(f)
    if casenum in train_id:
        train_files.append(f)
    elif casenum in val_id:
        valid_files.append(f)
    elif casenum in test_id:
        test_files.append(f)


train_files.sort()
valid_files.sort()
test_files.sort()

# 生成csv
pd.DataFrame(make_records(train_files)).to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
pd.DataFrame(make_records(valid_files)).to_csv(os.path.join(SPLITS_DIR, 'valid.csv'), index=False)
pd.DataFrame(make_records(test_files)).to_csv(os.path.join(SPLITS_DIR, 'test.csv'), index=False)