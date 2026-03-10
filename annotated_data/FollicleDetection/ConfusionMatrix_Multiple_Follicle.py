import os
import copy
import gc
import pandas as pd
import time
import csv
import cv2
from scipy import ndimage
from skimage import io
import json
import numpy as np
import torchmetrics.classification
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
import torchmetrics

from DataLoader_lightning_MultipleFollicle_no_masks import TrachomaDataModule, ToTensor
from Training_lightning_MultipleFollicle import (
    TrachomaGradableArea as SegmentationModel,
)  # trachoma follicle detection

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

path = "/home/Trachoma/annotated_data/FollicleDetection/Checkpoints/Sanity_Check/last.ckpt"

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

fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)

dm.setup()
test_data = dm.test_dataloader()
print("test", len(test_data))
test_data.dataset.name = True

if torch.cuda.is_available():
    accelerator = "gpu"
    num_devices = torch.cuda.device_count()  # use all available GPUs
else:
    accelerator = "cpu"
    num_devices = 1  # single CPU
#
trainer = pl.Trainer(
    accelerator=accelerator,  # or "gpu" if CUDA available
    devices=num_devices,
    log_every_n_steps=2,
    max_epochs=1,
    default_root_dir="Training_Checkpoints",
)

preds = []
target = []




results = pd.DataFrame(columns=["image_path", "model_output_path", 'mask_path', "true_label", "predicted_label"])

model_output_dir = 'follicle_detection_figs/model_eval/test_data_model_outputs/'

pred_mask_dir = 'follicle_detection_figs/model_eval/test_data_predicted_masks/'

os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(pred_mask_dir, exist_ok=True)

model = SegmentationModel.load_from_checkpoint(path, model=fcn, strict=True)
model.eval()

# device = torch.device("cpu")
# model = model.to(device)

confmat = torchmetrics.ConfusionMatrix(task="binary")
rec = torchmetrics.Recall(task="binary")
pre = torchmetrics.Precision(task="binary")
f1 = torchmetrics.F1Score(task="binary")
kappa = torchmetrics.CohenKappa(task='binary')


with torch.no_grad():
    for batch in test_data:

        images, labels = batch["image"], batch["label"]
        label = labels.item()

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

        mean = np.mean(outputs)
        std = np.std(outputs)
        min_follicle_value = 0 # mean + 2 * std
        outMask = outputs >= min_follicle_value
        labeled_array, num_follicles = ndimage.label(outMask)

        pred = num_follicles >= 3
        pred = int(pred)
        pred = torch.tensor([pred])
        label = torch.tensor([label])

        rec.update(pred, label)
        pre.update(pred, label)
        confmat.update(pred, label)
        f1.update(pred, label)
        kappa.update(pred, label)
        preds.extend(pred)
        target.extend(label)

        image_path = batch['path'][0]
        model_output_path = os.path.join(model_output_dir, batch['name'][0])
        pred_mask_path = os.path.join(pred_mask_dir, batch['name'][0])

        # plt.imsave(model_output_path, outputs)
        # plt.imsave(pred_mask_path, outMask)
        # np.save(model_output_path, outputs)
        # np.save(pred_mask_path, outMask)

        df_row = pd.DataFrame([{'image_path': image_path, 'model_output_path': model_output_path + '.npy', 'mask_path': pred_mask_path + '.npy', 'true_label': int(label), 'predicted_label': int(pred)}])
        results = pd.concat([results, df_row], ignore_index=True)


# r.to_csv("BestModelResults_CombinedData.csv", index="False")
#
cm = confmat.compute().numpy()
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
# plt.savefig('follicle_detection_figs/model_eval/confusion_matrix.png')
plt.show()

# results.to_csv('follicle_detection_figs/model_eval/model_predictions.csv', index=False)
# fig = plt.figure()
# precision, recall, thresholds = prc.compute()

# plt.plot(recall, precision)
# plt.ylabel("Precision")
# plt.xlabel("Recall")
# print(thresholds)
# print(recall)
# print(precision)

model_summary = pd.DataFrame([{'Precision': float(pre.compute()), 'Recall': float(rec.compute()), 'F1': float(f1.compute()), 'Kappa': float(kappa.compute())}])

model_summary.to_csv('follicle_detection_figs/model_eval/model_summary.csv')


print(pre.compute(), rec.compute(), f1.compute(), kappa.compute())


# fig = plt.figure() 
# plt.scatter(target, preds)
# plt.plot([0, 1], [thresh, thresh])
#
# # fig, ax = plt.subplots(1, len(bad))
# # for i, im in enumerate(bad):
# #     # ax[0, i].imshow(im)
# #     ax[i].imshow(bad_true[i])
# #     ax[i].axis('off')
# #     ax[i].axis('off')
# # fig.suptitle('False Negatives')
# # #
# # fig, ax = plt.subplots(1, len(good))
# # for i, im in enumerate(good):
# #     # ax[0, i].imshow(im)
# #     ax[i].imshow(good_true[i])
# #     ax[i].axis('off')
# #     ax[i].axis('off')
# # fig.suptitle('True Positives')
#
# # fig, ax = plt.subplots(1, len(fp))
# # for i, im in enumerate(fp):
# #     ax[i].imshow(im)
# #     ax[i].axis('off')
#
# # print(bad_name)
plt.show()
#
#
#
