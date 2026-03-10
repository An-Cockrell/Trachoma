import numpy as np
import cv2
import matplotlib.pyplot as plt
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
from dataloader_multiple_follicle_sanity_check import TrachomaDataModule, ToTensor
from training_multiple_follicle_sanity_check import TrachomaGradableArea

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay


fig_save_dir = "./follicle_detection_figs/detection_performance/"
img_dir = "./MultipleFollicleImages/Images/"
mask_dir = "./MultipleFollicleImages/Masks/"

os.makedirs(fig_save_dir, exist_ok=True)


def compare_follicles_by_centroid(target_mask, predicted_mask, img_name):
    """
    Compare the target and predicted binary masks based on the centroids of the connected components.
    Display the comparison plot with:
    - True positives in green
    - False positives in blue
    - Misses in red

    Parameters:
        target_mask (np.array): Binary mask (1s for follicles, 0s for background).
        predicted_mask (np.array): Binary mask (1s for follicles, 0s for background).

    Returns:
        true_positives (list): List of tuples (cX, cY) for correct matches.
        false_negatives (list): List of tuples (cX, cY) for missed follicles (target not predicted).
        false_positives (list): List of tuples (cX, cY) for erroneous follicles (predicted but not target).
    """

    # Ensure masks are of type np.uint8
    target_mask = np.array(target_mask, dtype=np.uint8)
    predicted_mask = np.array(predicted_mask, dtype=np.uint8)
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)
    img_array = np.array(img)
    # Get connected components and stats for target and predicted masks
    num_labels_target, _, stats_target, centroids_target = (
        cv2.connectedComponentsWithStats(target_mask)
    )
    num_labels_pred, _, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(
        predicted_mask
    )

    # comparison_image = np.zeros_like(target_mask, dtype=np.uint8)

    true_positives = []
    false_negatives = []
    false_positives = []

    # Process the target components (target_mask)
    target_bboxes = {
        i: stats_target[i][:4] for i in range(1, len(stats_target))
    }  # Excluding background (label 0)

    # Process the predicted components (predicted_mask)
    predicted_bboxes = {
        i: stats_pred[i][:4] for i in range(1, len(stats_pred))
    }  # Excluding background (label 0)

    # To track the matches for centroids
    matched_pred_labels = set()
    radius = 0
    # Loop over the target centroids and match them with predicted centroids
    for target_idx, target_centroid in enumerate(
        centroids_target[1:], 1
    ):  # Skip the background label (0)
        cX_target, cY_target = target_centroid
        target_bbox = target_bboxes[target_idx]
        x, y, w, h = target_bbox
        if radius == 0:
            radius = int(min(w, h) / 2)
        # Check predicted centroids and see if they fall within the bounding box of the target
        match_found = False
        for pred_idx, pred_centroid in enumerate(
            centroids_pred[1:], 1
        ):  # Skip the background label (0)
            cX_pred, cY_pred = pred_centroid
            pred_bbox = predicted_bboxes[pred_idx]
            px, py, pw, ph = pred_bbox

            # Check if predicted centroid falls within the target's bounding box
            if x <= cX_pred <= x + w and y <= cY_pred <= y + h:
                # cv2.circle(comparison_image, (int(cX_pred), int(cY_pred)), int(min(w, h)), (0, 255, 0), -1)
                true_positives.append((cX_pred, cY_pred))  # True positive match
                matched_pred_labels.add(
                    pred_idx
                )  # Mark this predicted label as matched
                match_found = True
                break

        if not match_found:
            false_negatives.append(
                (cX_target, cY_target)
            )  # Missed follicle (target but no match in prediction)
            # cv2.circle(comparison_image, (int(cX_target), int(cY_target)), int(min(w, h)), (255, 0, 0), -1)  # Red

    # Now loop over predicted centroids to identify false positives (erroneous)
    for pred_idx, pred_centroid in enumerate(
        centroids_pred[1:], 1
    ):  # Skip the background label (0)
        if pred_idx not in matched_pred_labels:
            cX_pred, cY_pred = pred_centroid
            false_positives.append(
                (cX_pred, cY_pred)
            )  # False positive (predicted but not in target)
            # cv2.circle(comparison_image, (int(cX_pred), int(cY_pred)), int(min(pw, ph)), (0, 0, 255), -1)  # Blue

    # result_image = np.zeros((target_mask.shape[0], target_mask.shape[1], 3), dtype=np.uint8)

    # for (x, y) in true_positives:
    #     cv2.circle(result_image, (int(x), int(y)), radius, (0, 255, 0), -1)  # Green

    # # Draw red circles for false negatives
    # for (x, y) in false_negatives:
    #     cv2.circle(result_image, (int(x), int(y)), radius, (255, 0, 0), -1)  # Red

    # # Draw blue circles for false positives
    # for (x, y) in false_positives:
    #     cv2.circle(result_image, (int(x), int(y)), radius, (0, 0, 255), -1)  # Blue

    # fig, ax = plt.subplots(1, 3)
    # fig.suptitle('Follicle Detection Evaluation')
    # # ax[0].imshow(img_array)
    # # ax[0].axis('off')
    # # ax[0].set_title('Original Image')
    # ax[0].imshow(target_mask)
    # ax[0].axis("off")
    # ax[0].set_title("Target")
    # ax[1].imshow(predicted_mask)
    # ax[1].axis("off")
    # ax[1].set_title("Predicted")
    # ax[2].imshow(result_image)
    # ax[2].axis("off")
    # ax[2].set_title('Comparison')
    # fig.savefig(f'{fig_save_dir}{img_name}')
    # plt.show()
    return (
        true_positives,
        false_negatives,
        false_positives,
        num_labels_target - 1,
        num_labels_pred - 1,
    )


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

dm = TrachomaDataModule(
    img_dir,
    mask_dir,
    transforms_0=trans_0,
    transforms_1=trans_1,
    batch_size=1,
    num_workers=1,
    oversample=False,
    oversample_amt=20,
    normalize=True,
)

fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)


dm.setup()
test_data = dm.test_dataloader()
print("test", len(test_data))
print("training", len(dm.train_dataloader()))


model = TrachomaGradableArea.load_from_checkpoint(path, model=fcn, strict=True)

stats = []
total_folls = []

for i, batch in enumerate(test_data):

    images, targets = batch["image"], batch["label"]
    targets = targets.squeeze().numpy()

    device = next(model.parameters()).device
    images = images.to(device)

    outputs = model(images)["out"].squeeze().detach().cpu().numpy()
    im = images.squeeze().permute(1, 2, 0).cpu().numpy()
    outMask = (outputs >= 0).astype(np.uint8)
    targets = np.array(targets, dtype=np.uint8)
    img_name = batch["name"][0]

    true_positives, false_negatives, false_positives, num_foll_target, num_foll_pred = (
        compare_follicles_by_centroid(targets, outMask, img_name)
    )
    # print("True Positives (Correct Matches):", true_positives)
    # print("False Negatives (Misses):", false_negatives)
    # print("False Positives (Erroneous):", false_positives)

    num_tp = len(true_positives)
    num_fn = len(false_negatives)
    num_fp = len(false_positives)

    stats.append((num_tp, num_fn, num_fp))
    total_folls.append((num_foll_target, num_foll_pred))

# stat_sums is tuple of (total_tp, total_fn, total_fp)
stat_sums = tuple(map(sum, zip(*stats)))
total_foll_sums = tuple(map(sum, zip(*total_folls)))

unweighted_counts_per_image = tuple(x / len(stats) for x in stat_sums)

total_tp = stat_sums[0]
total_fn = stat_sums[1]
total_fp = stat_sums[2]
total_target = total_foll_sums[0]
total_pred = total_foll_sums[1]


weighted_tp = total_tp / total_target
weighted_fn = total_fn / total_target
weighted_fp = total_fp / total_pred

conf_matrix_counts = np.array([[total_tp, total_fn], [total_fp, 0]])
conf_matrix_rates = np.array([[weighted_tp, weighted_fn], [weighted_fp, 0]])

print(conf_matrix_counts)
# Display
ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_counts, display_labels=["Follicle", " Non-Follicle"]
).plot()

ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_rates, display_labels=["Follicle", " Non-Follicle"]
).plot()


# recall = total_TP / (total_TP + total_FN)
# precision = total_TP / (total_TP + total_FP)
# f1 = 2 * precision * recall / (precision + recall)


# Weighted avgs should be number of each type of prediction divided by the number of total prediction types for a given image, and then sum those?
