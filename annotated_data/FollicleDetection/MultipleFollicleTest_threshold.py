import os
import copy
import gc
import pandas as pd
import time
import csv
import cv2
from skimage import io, measure
import json
import numpy as np
import wandb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore

from DataLoader_lightning_MultipleFollicle import TrachomaDataModule, ToTensor
from Training_lightning_MultipleFollicle import TrachomaGradableArea


def count_islands_and_masses(grid):
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    masses = []

    def dfs(i, j):
        if i < 0 or i >= rows or j < 0 or j >= cols:
            return 0
        if visited[i][j] or grid[i][j] == 0:
            return 0
        visited[i][j] = True
        mass = 1
        # Check up, down, left, right
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            mass += dfs(i + dx, j + dy)
        return mass

    island_count = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                masses.append(dfs(i, j))
                island_count += 1

    return island_count, masses


predicted_mask_dir = "./MultipleFollicleImages/OutMasks"

target_mask_dir = "./MultipleFollicleImages/Masks/Test_Targets/"

predicted_mask_list = np.sort(os.listdir(predicted_mask_dir))

target_mask_list = np.sort(os.listdir(target_mask_dir))

figures = []

target_islands_list = []
predicted_islands_list = []
target_masses_list = []
predicted_masses_list = []


# target_list = []
for predicted_mask_name, target_mask_name in zip(predicted_mask_list, target_mask_list):
    # target_list.append(target_name)
    predicted_mask = np.load(
        os.path.join(predicted_mask_dir, predicted_mask_name)
    ).astype(int)

    target_mask = np.load(os.path.join(target_mask_dir, target_mask_name))
    target_mask[target_mask != 0] = 1

    # print(predicted_mask_name)
    # print(np.shape(predicted_mask))
    # print(target_mask_name)
    # print(np.shape(target_mask))

    num_target_islands, target_masses = count_islands_and_masses(target_mask)
    target_islands_list.append(num_target_islands)
    target_masses_list.append(target_masses)
    print(f"Target # Islands: {num_target_islands}")
    print(f"Target Masses: {target_masses}")

    num_pred_islands, pred_masses = count_islands_and_masses(predicted_mask)
    predicted_islands_list.append(num_pred_islands)
    predicted_masses_list.append(pred_masses)
    print(f"Predicted # Islands: {num_pred_islands}")
    print(f"Predicted Masses: {pred_masses}")

    fig, ax = plt.subplots(1, 2)  # 1 row, 2 columns
    ax[0].imshow(target_mask)
    ax[0].axis("off")
    ax[0].set_title("target")
    ax[1].imshow(predicted_mask)
    ax[1].axis("off")
    ax[1].set_title("predicted")
    fig.suptitle(target_mask_name)  # optional title
    figures.append(fig)
    plt.show()

# Find threshold form these test images (doesn't work all that well):

all_masses = sorted({m for masses in predicted_masses_list for m in masses})
best_thresh, best_cost = None, float("inf")
for thresh in all_masses:
    cost = 0
    for pred_masses, target_count in zip(predicted_masses_list, target_islands_list):
        keep_count = sum(1 for m in pred_masses if m >= thresh)
        cost += (keep_count - target_count) ** 2
    if cost < best_cost:
        best_cost, best_thresh = cost, thresh

print(f"→ best threshold = {best_thresh},  total absolute‐error = {best_cost}")


# Plan for getting the size threshold from larger dataset:
# get training set from black box model --> process images in identical manner to those used for follicle detection model --> parameter sweep for follicle size (and amount?), using model performance as objective function --> set follicle detection model to predict Positive when size and amount of follicles lands above the new found thresholds--> evaluate the follicle detection model on the black box test set and compare performance


# get black box training set:

img_dir_o = "TrachomaData/tarsal plate zip/allTZphotos/allTZphotos"
img_keys_o = "2300consensus8-2021.csv"
img_dir_m = "m"
img_keys_m = "m/tfti.csv"

# follicle detection best model
path = "/home/Trachoma/annotated_data/FollicleDetection/Checkpoints/Segmentation_Test_MultipleFollicle_520/last.ckpt"
