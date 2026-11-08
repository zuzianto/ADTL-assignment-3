import pickle
from PIL import Image
import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sam_fact_tt_image_encoder import Fact_tt_Sam, Fact_tt_un_Sam, Fact_tt_new_un_Sam
from importlib import import_module
from segment_anything import sam_model_registry
from collections import defaultdict
from medpy.metric.binary import dc, hd95
from icecream import ic
import pandas as pd
from scipy.ndimage import zoom
import SimpleITK as sitk
from skimage.measure import block_reduce
import time

sys.path.append('/media/dist7t_/xiaochuan/SUGFW/visualization/scripts')
def mean_dice(pred, label, num_classes):
    """
    Calculate the mean Dice coefficient for multi-class segmentation.

    Args:
        pred: Either a np.ndarray (predicted segmentation mask) or str (path to NIfTI file).
        label: Either a np.ndarray (ground truth) or str (path to NIfTI file).
        num_classes (int): Number of classes (excluding background, assuming classes start from 1).

    Returns:
        float: Mean Dice coefficient across all classes.
    """
    if isinstance(pred, str) and isinstance(label, str):
        # Load images using SimpleITK
        pred_img = sitk.ReadImage(pred)
        pred = sitk.GetArrayFromImage(pred_img)
        
        label_img = sitk.ReadImage(label)
        label = sitk.GetArrayFromImage(label_img)
        label = label[66:, :, :]  # Assuming this is needed for alignment
    
    # Ensure shapes match
    if pred.shape != label.shape:
        raise ValueError("Shape mismatch: pred {}, label {}".format(pred.shape, label.shape))
    
    dice_scores = []
    for c in range(1, num_classes + 1):
        pred_c = (pred == c).astype(np.uint8)
        label_c = (label == c).astype(np.uint8)
        intersection = np.sum(pred_c & label_c)
        union = np.sum(pred_c) + np.sum(label_c)
        if union == 0:
            dice = 1.0  # Perfect score if no pixels of this class in both
        else:
            dice = 2.0 * intersection / union
        dice_scores.append(dice)
    return np.mean(dice_scores)



def average_pooling(
    uncertainty: np.ndarray, target_size: tuple = (64, 64)
) -> np.ndarray:
    """
    Perform average pooling to resize the uncertainty map to the target size.

    Args:
        uncertainty (np.ndarray): Input uncertainty map of shape (H, W).
        target_size (tuple): Target size (height, width), default is (64, 64).

    Returns:
        np.ndarray: Resized uncertainty map of shape (64, 64).
    """
    input_height, input_width = uncertainty.shape
    target_height, target_width = target_size

    # Calculate pooling block size
    block_size = (input_height // target_height, input_width // target_width)

    # Perform average pooling using block_reduce
    pooled_uncertainty = block_reduce(uncertainty, block_size=block_size, func=np.mean)

    return pooled_uncertainty


def multi_hd95(pred, label, spacing=None):
    unique_labels = np.unique(label)
    hd_scores = []
    for c in unique_labels:
        if c == 0:
            continue
        pred_c = (pred == c).astype(np.uint8)
        label_c = (label == c).astype(np.uint8)
        if np.sum(pred_c) == 0 or np.sum(label_c) == 0:
            hd_scores.append(0)  # Assign 0 if no pixels
        else:
            hd_scores.append(hd95(pred_c, label_c, voxelspacing=spacing))
    return np.mean(hd_scores) if hd_scores else 0



def test_single_volume_new(
    image,
    label,
    uncertainty_avg,
    net,
    multimask_output,
    patch_size=512,
    test_save_path=None,
    case=None,
    cal_hd95=False,
    spacing=None,
):
    preds = []

    uncertainty_avg = torch.from_numpy(uncertainty_avg)
    uncertainty_avg = uncertainty_avg.unsqueeze(0).unsqueeze(0).float().cuda()
    
    for i in range(image.shape[0]):
        slice_input = (
            torch.from_numpy(image[i, ...])
            .unsqueeze(0)
            .unsqueeze(0)
            .permute(0, 1, 4, 2, 3)
            .float()
            .cuda()
        )


        
        with torch.no_grad():
            outputs = net(
                batched_input=slice_input,
                multimask_output=multimask_output,
                image_size=patch_size,
                uncertainty_avg=uncertainty_avg,
                uncertainty=None,
                stage="test",
            )
            pred = outputs["masks"]
            pred = torch.argmax(pred, dim=1)
            pred = pred.squeeze().cpu().detach().numpy().astype(np.uint8)
            # if pred.shape[0] != patch_size:
            #     pred = zoom(pred, (label.shape[1] / pred.shape[0], label.shape[2] / pred.shape[1]), order=0)

            preds.append(pred)

    preds = np.array(preds)
    if preds.shape != label.shape:
        zoom_factor = (
            label.shape[0] / preds.shape[0],
            label.shape[1] / preds.shape[1],
            label.shape[2] / preds.shape[2],
        )
        preds = zoom(preds, zoom_factor, order=0).astype(np.uint8)

    unique_labels = np.unique(label)
    if len(unique_labels) > 2:
        # Multi-class segmentation
        num_classes = len(unique_labels) - 1  # Assuming 0 is background
        dice = mean_dice(preds, label, num_classes)
        HD95 = multi_hd95(preds, label, spacing) if cal_hd95 else 10000
    else:
        # Binary segmentation
        dice = dc(preds, label)
        HD95 = hd95(preds, label, voxelspacing=spacing) if cal_hd95 else 10000

    if test_save_path is not None:
        sitk.WriteImage(
            sitk.GetImageFromArray(preds.astype(np.float32)),
            f"{test_save_path}/{case}_pred.nii.gz",
        )

    return dice, HD95


def inference(
    args,
    multimask_output,
    model,
    stage,
    cal_hd95=False,
    test_save_path=None,
):
    model.eval()

    df = pd.read_csv(args.root_path + "/splits/" + stage + ".csv")
    cases = defaultdict(list)
    for _, row in df.iterrows():
        image_path = row["image_pth"]
        mask_path = row["mask_pth"]
        uncertainty_path = image_path.replace(".pkl", "_uncertainty_10.pkl").replace(
            "image", "uncertainty"
        )
        avg_uncertainty_path = os.path.join(
            os.path.dirname(image_path).replace("image", "uncertainty"),
            "uncertainty_10_avg.pkl",
        )
        case_name = os.path.basename(image_path).split("_Slice")[0]
        slice_id = int(os.path.basename(image_path).split("_Slice")[1].split(".")[0])
        cases[case_name].append(
            (slice_id, image_path, mask_path, uncertainty_path, avg_uncertainty_path)
        )

    # Sort slices within each Case by slice ID
    for case_name in cases:
        cases[case_name] = sorted(cases[case_name], key=lambda x: x[0])

    # Process each Case
    all_dice = []
    all_hd95 = []

    for case_name, slices in tqdm(cases.items()):
        spacing = None
        if cal_hd95:
            spacing_path = os.path.join(
                args.root_path, "spacing", f"{case_name}_spacing.pkl"
            )
            with open(spacing_path, "rb") as f:
                spacing = pickle.load(f)
        # Load and stack slices to form 3D volume
        image_arr_list = []
        mask_arr_list = []
        uncertainty_arr_list = []
        for _, image_path, mask_path, uncertainty_path, avg_uncertainty_path in slices:

            a, b = os.path.dirname(image_path), os.path.basename(image_path)
            num = int(b[4:7])

            # spacing_path = a.replace("image", "spacing")
            # spacing_path = os.path.join(spacing_path, f"Case{num:03d}_spacing.pkl")

            with open(image_path, "rb+") as file:
                image_arr = pickle.load(file)
            with open(mask_path, "rb+") as file:
                mask_arr = pickle.load(file)
            # with open(spacing_path, "rb+") as file:
            #     spacing = pickle.load(file)
            with open(avg_uncertainty_path, "rb+") as file:
                avg_uncertainty = pickle.load(file)

            # spacing = spacing[::-1]

            # image_arr = np.clip(image_arr, HU_min, HU_max)
            # image_arr = (image_arr - HU_min) / (HU_max - HU_min) * 255.0
            # image_arr = np.float32(image_arr)
            # image_arr = (image_arr - data_mean) / data_std
            image_arr = (image_arr - image_arr.min()) / (
                image_arr.max() - image_arr.min() + 0.00000001
            )

            avg_uncertainty = average_pooling(
                avg_uncertainty, target_size=(args.img_size // 4, args.img_size // 4)
            )
            mask_arr = np.float32(mask_arr)

            image_arr_list.append(image_arr)
            mask_arr_list.append(mask_arr)

        image = np.array(image_arr_list)
        label = np.array(mask_arr_list)

        if image.shape[1] != args.img_size:
            zoom_factor = (
                1.0,
                args.img_size / image.shape[1],
                args.img_size / image.shape[2],
                1.0,
            )
            image = zoom(image, zoom_factor, order=3)

        if image.shape[-1] == 1:
            image = image.repeat(3, axis=-1)  # z x y 3

        label = label.squeeze(-1)
        dice, HD95 = test_single_volume_new(
            image,
            label,
            avg_uncertainty,
            model,
            multimask_output=multimask_output,
            patch_size=args.img_size,
            test_save_path=test_save_path,
            case=case_name,
            cal_hd95=cal_hd95,
            spacing=spacing,
            # spacing=(1.0, 1.0, 1.0),
        )
        all_dice.append(dice)
        all_hd95.append(HD95)
        # all_times.append(times)
        logging.info(f"{case_name} - Dice: {dice:.4f}, HD95: {HD95:.4f}")

    all_dice = np.array(all_dice)
    all_hd95 = np.array(all_hd95)
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_hd95 = np.mean(all_hd95)
    std_hd95 = np.std(all_hd95)
    logging.info(
        f"{case_name} - Dice: {mean_dice:.4f}±{std_dice:.4f}, HD95: {mean_hd95:.4f}±{std_hd95:.4f}"
    )

    # all_times = np.array(all_times)
    # logging.info(f"Average inference time per case: {np.mean(all_times):.4f} seconds, std: {np.std(all_times):.4f} seconds")

    model.train()

    return mean_dice, std_dice, mean_hd95, std_hd95


def config_to_dict(config):
    items_dict = {}
    with open(config, "r") as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(": ")
        items_dict[key] = value
    return items_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapt_ckpt",
        type=str,
        default="../output/Promise12/train_origin/best_model.pth",
        help="The checkpoint after adaptation",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="../data/Promise12",
    )

    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--img_size", type=int, default=512, help="Input image size of the network"
    )

    parser.add_argument("--stage", type=str, default="test", help="valid or test")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--is_savenii",
        action="store_true",
        help="Whether to save results during inference",
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../checkpoint/sam_vit_b_01ec64.pth",
        help="Pretrained checkpoint",
    )
    parser.add_argument(
        "--vit_name", type=str, default="vit_b_new", help="Select one vit model"
    )
    parser.add_argument("--cal_hd95", action="store_true", help="Calculate HD95")
    parser.add_argument("--rank", type=int, default=32, help="Rank for FacT adaptation")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--module", type=str, default="sam_fact_tt_image_encoder")
    parser.add_argument(
        "--empirical", action="store_true", help="Whether to use empirical model"
    )

    args = parser.parse_args()
    args.output_dir = os.path.join(os.path.dirname(args.adapt_ckpt), args.stage)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )

    pkg = import_module(args.module)
    net = pkg.Fact_tt_new_un_Sam(sam, args.rank, s=args.scale).cuda()


   
    assert args.adapt_ckpt is not None
    # net.load_parameters(args.adapt_ckpt)
    
    state_dict = torch.load(args.adapt_ckpt)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)

    
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    os.makedirs("./testing_log/" + args.adapt_ckpt.split("/")[-3], exist_ok=True)
    logging.basicConfig(
        filename="./testing_log/"
        + args.adapt_ckpt.split("/")[-3]
        + "/"
        + args.adapt_ckpt.split("/")[-2]
        + "_"
        + "_log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = args.output_dir
    else:
        test_save_path = None

    inference(
        args,
        multimask_output,
        net,
        args.stage,
        args.cal_hd95,
        test_save_path,
    )
    logging.info("Testing Finished!")
