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

plt.ioff()
import sklearn.metrics as metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
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
from PIL import Image
from scipy import ndimage

from DataLoader_lightning_MultipleFollicle_no_masks import TrachomaDataModule, ToTensor
from Training_lightning_MultipleFollicle import (
    TrachomaGradableArea as SegmentationModel,
)  # trachoma follicle detection

import cProfile
import pstats


# profiler = cProfile.Profile()
# profiler.enable()
def param_sweep(results):
    true_labels = np.array([r[0] for r in results])
    follicle_counts = np.array([r[1] for r in results])

    # Sweep thresholds
    thresholds = range(0, max(follicle_counts) + 1)
    metrics_dict = {
        "threshold": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    for thresh in thresholds:
        preds = (follicle_counts >= thresh).astype(int)
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds)

        metrics_dict["threshold"].append(thresh)
        metrics_dict["accuracy"].append(acc)
        metrics_dict["f1"].append(f1)
        metrics_dict["precision"].append(prec)
        metrics_dict["recall"].append(rec)

    # Convert to DataFrame for easy analysis
    df_metrics = pd.DataFrame(metrics_dict)

    # Print best threshold by F1
    best_idx = np.argmax(df_metrics["f1"])
    best_threshold = df_metrics["threshold"][best_idx]
    print(f"✅ Best threshold: {best_threshold} (F1: {df_metrics['f1'][best_idx]:.2f})")

    # Optional: plot F1 vs threshold
    plt.plot(df_metrics["threshold"], df_metrics["f1"], label="F1 Score")
    plt.plot(
        df_metrics["threshold"],
        df_metrics["accuracy"],
        label="Accuracy",
        linestyle="--",
    )
    plt.xlabel("Threshold (# follicles)")
    plt.ylabel("Score")
    plt.title("TF Detection Threshold Sweep")
    plt.legend()
    plt.grid(True)
    plt.savefig("follicle_detection_figs/new_model_parameter_sweep")
    plt.close()
    # plt.show()
    return best_threshold


# Gradable area model (ga) Setup
ga_path = "../Checkpoints/GradableAreaTest_new_data/last.ckpt"

ga_fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)

ga_model = SegmentationModel.load_from_checkpoint(ga_path, model=ga_fcn, strict=True)


# Follicle Detection Setup
img_dir_o = "/home/Trachoma/TrachomaData/tarsal plate zip/allTZphotos/allTZphotos"
img_keys_o = "/home/Trachoma/2300consensus8-2021.csv"
img_dir_m = "/home/Trachoma/m"
img_keys_m = "/home/Trachoma/m/tfti.csv"

# follicle detection best model
# path = "/home/Trachoma/annotated_data/FollicleDetection/Checkpoints/Segmentation_Test_MultipleFollicle_520/last.ckpt"

path = "/home/Trachoma/annotated_data/FollicleDetection/Checkpoints/Old_script_new_data/last.ckpt"

trans_0 = transforms.Compose(
    [ToTensor(), transforms.Resize(540), transforms.CenterCrop(520)]
)
trans_1 = transforms.Compose(
    [
        ToTensor(),
        transforms.Resize(540),
        transforms.RandomApply(
            nn.ModuleList(
                [
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(90),
                    transforms.RandomPerspective(0.3),
                ]
            )
        ),
        transforms.CenterCrop(520),
    ]
)

# setup dataloader to get the black box data, but processed with the follicle detection transforms
dm = TrachomaDataModule(
    img_dir_m,
    img_dir_o,
    img_keys_m,
    img_keys_o,
    "imagename",
    "consensus",
    transforms_0=trans_0,
    transforms_1=trans_1,
    batch_size=1,
    num_workers=4,
    oversample=False,
    oversample_amt=1,
    normalize=True,
)


#
fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)


print("starting setup")
dm.setup()
print("saving image paths")
# save the image names from training set
# train_image_paths = []
# for entry in dm.train_keys:
#     if entry[2] == "m":
#         img_path = os.path.join(dm.img_dir_m, "image" + str(entry[0])) + ".jpg"
#     else:
#         img_path = os.path.join(dm.img_dir_o, "0" + entry[0]) + ".jpg"
#     train_image_paths.append(img_path)

# # Save to file for offline normalization
# with open("clean_images/train_image_paths.txt", "w") as f:
#     for p in train_image_paths:
#         f.write(p + "\n")

# # then do the same for test images
# test_image_paths = []
# for entry in dm.test_keys:
#     if entry[2] == "m":
#         img_path = os.path.join(dm.img_dir_m, "image" + str(entry[0])) + ".jpg"
#     else:
#         img_path = os.path.join(dm.img_dir_o, "0" + entry[0]) + ".jpg"
#     test_image_paths.append(img_path)

# # Save to file for offline normalization
# with open("clean_images/test_image_paths.txt", "w") as f:
#     for p in test_image_paths:
#         f.write(p + "\n")

# print(test_image_paths[:5])
# print(train_image_paths[:5])

test_data = dm.test_dataloader()
train_data = dm.train_dataloader()
print("test", len(test_data))
print("training", len(train_data))

# load the follicle detection model
model = SegmentationModel.load_from_checkpoint(path, model=fcn, strict=True)

if torch.cuda.is_available():
    accelerator = "gpu"
    num_devices = torch.cuda.device_count()  # use all available GPUs
else:
    accelerator = "cpu"
    num_devices = 1  # single CPU

trainer = pl.Trainer(
    accelerator=accelerator,  # or "gpu" if CUDA available
    devices=num_devices,
    log_every_n_steps=2,
    max_epochs=1,
    default_root_dir="Training_Checkpoints",
)

model.eval()


# fig1, ax1 = plt.subplots(1, 3)
os.makedirs("follicle_detection_figs/", exist_ok=True)


output_list = []

# we need to dynamically set the threshold to maximize the number of correct TF predictions

results = []

for i, batch in enumerate(train_data):

    images, labels = batch["image"], batch["label"]
    true_label = labels.item()

    # print(true_label)
    # print(np.shape(images), np.shape(labels))
    # continue
    device = next(model.parameters()).device
    images = images.to(device)

    crop_mask = ga_model(images)["out"]  # .squeeze().detach().cpu().numpy()
    crop_mask = crop_mask >= 0

    cropped_image = images * crop_mask
    cropped_im = cropped_image.squeeze().permute(1, 2, 0).cpu().numpy()

    outputs = model(cropped_image)["out"].squeeze().detach().cpu().numpy()
    im = images.squeeze().permute(1, 2, 0).cpu().numpy()

    outputs_recropped = outputs * crop_mask.squeeze().cpu().numpy()

    output_list.append(outputs)

    mean = np.mean(outputs)
    std = np.std(outputs)
    min_follicle_value = mean + 1 * std
    # outMask = outputs >= min_follicle_value
    outMask = outputs > 0
    # collect (num_follicle, label) pairs for threshold sweep
    labeled_array, num_follicles = ndimage.label(outMask)
    results.append((true_label, num_follicles))

    fig, ax = plt.subplots(1, 4)

    ax[0].imshow(im)
    ax[0].axis("off")
    ax[0].set_title("image")
    ax[1].imshow(cropped_im)
    ax[1].axis("off")
    ax[1].set_title("cropped_image")
    ax[2].imshow(outputs)
    ax[2].axis("off")
    ax[2].set_title("output")
    ax[3].imshow(outMask)
    ax[3].axis("off")
    ax[3].set_title("outmask")
    fig.suptitle(f"Num Pred Follicles: {num_follicles}")
    # os.makedirs(
    #     "follicle_detection_figs/new_model/positive_case_pred_results/", exist_ok=True
    # )
    # os.makedirs(
    #     "follicle_detection_figs/new_model/negative_case_pred_results/", exist_ok=True
    # )
    # if true_label == 0:
    #     plt.savefig(
    #         f"follicle_detection_figs/new_model/negative_case_pred_results/sample_{i}"
    #     )
    # else:
    #     plt.savefig(
    #         f"follicle_detection_figs/new_model/positive_case_pred_results/sample_{i}"
    #     )

    plt.show()
    plt.imshow(outputs_recropped)
    plt.show()
    plt.close()
    # print(num_follicles)
    if i >= 50:
        break
    # name = batch['name'][0].split('.')[0]

    # outpath = f'{out_dir}/{name}_outMask.npy'
    # np.save(outpath, outMask)

    # targetpath = f'{target_mask_test_dir}/{name}.npy'
    # np.save(targetpath, targets)

param_sweep(results)


