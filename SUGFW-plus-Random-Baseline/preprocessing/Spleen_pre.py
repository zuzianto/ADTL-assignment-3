import os
import SimpleITK as sitk
import pickle
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

Foreslices_IMAGE_DIR = "../data/Spleen/Foreslices/image"
Foreslices_LABEL_DIR = "../data/Spleen/Foreslices/label"
os.makedirs(Foreslices_IMAGE_DIR, exist_ok=True)
os.makedirs(Foreslices_LABEL_DIR, exist_ok=True)

base_dir = "../data/Spleen/raw/Task09_Spleen/Task09_Spleen/imagesTr"

for filename in tqdm([os.path.join(base_dir, file) for file in os.listdir(base_dir)], ncols=100):
    file = os.path.basename(filename)
    if file.startswith('.') or not file.endswith('.nii.gz'):
        continue
    
    idx = int(file.split('_')[1].split('.')[0])

    image = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    label = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace('imagesTr', 'labelsTr')))

    """ Intensity Truncation """
    percentile_99 = np.percentile(image, 99)
    image = np.clip(image, a_min=None, a_max=percentile_99)

    """ Normalization """
    image = (image - image.mean() + 1e-6) / (image.std() + 1e-6)

    """ Scale to Dx512x512 """
    if image.shape[1] != 512 or image.shape[2] != 512:
        image = zoom(image, (1.0, 512 / image.shape[1], 512 / image.shape[2]), order=3)
        label = zoom(label, (1.0, 512 / label.shape[1], 512 / label.shape[2]), order=0)

    label[label > 0] = 1

    fore = np.any(label, axis=(1, 2))
    nonzero_indices = np.where(fore)[0]
    z_min, z_max = nonzero_indices[0], nonzero_indices[-1]
    image = np.transpose(image, (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))

    for i in range(image.shape[2]):
        slice_id = i

        image_slice = image[:, :, i : i + 1]
        label_slice = label[:, :, i : i + 1]

        """ generate fore slices """
        if i >= z_min - 2 and i <= z_max + 2:
            image_path = os.path.join(
                Foreslices_IMAGE_DIR,
                f"Case{idx:03d}_Slice{slice_id:03d}.pkl",
            )
            label_path = os.path.join(
                Foreslices_LABEL_DIR,
                f"Case{idx:03d}_Slice{slice_id:03d}.pkl",
            )
            with open(image_path, "wb") as f:
                pickle.dump(image_slice, f)
            with open(label_path, "wb") as f:
                pickle.dump(label_slice, f)
