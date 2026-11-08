import argparse
from collections import defaultdict
import logging
import os
import pickle
import random
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datetime import datetime
from inference import inference
import warnings


warnings.filterwarnings("ignore")


def calc_loss(
    outputs,
    low_res_label_batch,
    ce_loss,
    dice_loss,
    mse_loss,
    dice_weight: float = 0.8,
    mse_param: float = 0.02,
):
    low_res_logits = outputs["low_res_logits"]
    uncertainty_embedding = outputs["uncertainty_embeddings"]
    predicted_un = outputs["predicted_un"]

    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss_consistency = mse_loss(predicted_un, uncertainty_embedding.detach())

    loss = (
        (1 - dice_weight) * loss_ce
        + dice_weight * loss_dice
        + mse_param * loss_consistency
    )

    return loss, loss_ce, loss_dice, loss_consistency



def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset import dataset_reader, RandomGenerator

    if not os.path.exists("./training_log"):
        os.mkdir("./training_log")
    logging.basicConfig(
        filename="./training_log/" + args.output.split("/")[-1] + "_log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes

    db_train = dataset_reader(
        csv_path=args.csv_path,
        num_classes=args.num_classes,
        transform=transforms.Compose(
            [
                RandomGenerator(
                    output_size=[args.img_size, args.img_size],
                    low_res=[low_res, low_res],
                )
            ]
        ),
    )
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(num_classes)
    mse_loss = nn.MSELoss(reduction="mean")

    batch_size = args.batch_size * args.n_gpu

    print("Starting to create DataLoader...")
    print(f"Dataset size: {len(db_train)}")
    print(f"Batch size: {batch_size}")
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    best_dice = 0.0
    b_lr = base_lr / args.warmup_period

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=b_lr,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )

    iter_num = 0
    max_epoch = args.train_max_epochs
    max_iterations = max_epoch * len(trainloader)
    logging.info(
        f"max_iterations: {max_iterations} iterations per epoch: {len(trainloader)}"
    )

    dataset_path = os.path.dirname(args.csv_path)

    for epoch_num in tqdm(range(max_epoch)):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = (
                sampled_batch["image"],
                sampled_batch["label"],
            )
            uncertainty_batch = sampled_batch["uncertainty"].cuda().contiguous()
            avg_uncertainty_batch = sampled_batch["avg_uncertainty"].cuda().contiguous()

            image_batch = image_batch.unsqueeze(1)  # [B, 1, C, H, W]
            if image_batch.shape[2] == 1:
                image_batch = torch.cat(
                    (image_batch, image_batch, image_batch), dim=2
                )  # [B, 1, 3, H, W]

            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)

            low_res_label_batch = sampled_batch["low_res_label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()

            if iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_
                
                mse_param = args.mse_param * ((iter_num + 1) / args.warmup_period)
            else:
                shift_iter = iter_num - args.warmup_period
                
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_
                
                mse_param = args.mse_param

            if iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_
                
            

            outputs = model(
                batched_input=image_batch,
                multimask_output=multimask_output,
                image_size=args.img_size,
                uncertainty_avg=avg_uncertainty_batch,
                uncertainty=uncertainty_batch,
                stage="train",
            )
            loss, loss_ce, loss_dice, loss_cons = calc_loss(
                outputs, label_batch, ce_loss, dice_loss, mse_loss, args.dice_param, mse_param
            )
            logging.info(
                f"iteration: {iter_num} loss: {loss.item():.4f} loss_ce: {loss_ce.item():.4f} loss_dice: {loss_dice.item():.4f} loss_mse: {loss_cons.item():.4f} lr: {lr_:.7f} mse_param: {mse_param:.7f}"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

        if (epoch_num + 1) % 100 == 0:
            dice, _, _, _ = inference(
                args,
                multimask_output=multimask_output,
                model=model,
                stage="valid",
                cal_hd95=False,
                # test_save_path="/data/mxc/MA-SAM/MA-SAM/outputs/NEU_Seg",
            )

            if dice > best_dice:
                best_dice = dice
                save_mode_path = os.path.join(
                    snapshot_path,
                    "epoch_" + str(epoch_num) + f"_{dice:.4f}" + ".pth",
                )
                save_best_path = os.path.join(snapshot_path, "train_best_model.pth")
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best_path)
                # try:
                #     model.save_parameters(save_mode_path)
                #     model.save_parameters(save_best_path)
                # except:
                #     model.module.save_parameters(save_mode_path)
                #     model.module.save_parameters(save_best_path)

    return "Training Finished!"
