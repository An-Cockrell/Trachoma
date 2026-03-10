import os
import copy
import gc
import pandas as pd
import time
import csv
import cv2
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

from DataLoader_lightning_AllData import (
    TrachomaDataModule,
    ToTensor,
    FollicleEnhance,
    CustomCrop,
)
from Training_lightning_AllData import TrachomaClassifier

# np.random.seed(100)

# img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
img_dir_o = "/home/Trachoma/TrachomaData/tarsal plate zip/allTZphotos/allTZphotos"
img_keys_o = "/home/Trachoma/2300consensus8-2021.csv"
img_dir_m = "/home/Trachoma/m"
img_keys_m = "/home/Trachoma/m/tfti.csv"
# img_keys_o = 'TrachomaData/trachomagroundtruthkey.csv'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flip_rotate_perspective_pretrained_resnet101_mData/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_follicleenhance_flip_rotate_norm_pretrained_resnet101_allData/last.ckpt'

# path = "TrainedModels/Pytorch_lightning_consensus_oversample10_flip_rotate_perspective_pretrained_resnet101_allData/last.ckpt"

path = "Checkpoints/all_clean_data/run_20251010_175725/last.ckpt"

# path = 'TrainedModels/Pytorch_lightning_flip_rotate_perspective_pretrained_resnet101_mData_synthPretrain6/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning__follicleEnhance_flip_rotate_perspective_pretrained_resnet101_mData_synthPretrain2/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_follicleEnhance_flip_rotate_perspective_resnet101_m_synth_dp0.3/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flip_rotate_perspective_pretrained_resnet101_mData_5percent_3/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flop_rotate_perspective_pretrained_norm_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flop_rotate_perspective_pretrained_norm_accumuate3batches_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_follicleenhance_flip_rotate_pretrained_norm_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_posweight40_pretrained_norm_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_posweight4_follicleenhance_flip_rotate_norm_pretrained_accum5batch_resnet18/last.ckpt'

thresh = 0.1
print(path, thresh)
# trans_0 = transforms.Compose(
#     [FollicleEnhance(), ToTensor(),
#      #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.CenterCrop(224)])
# trans_1 = transforms.Compose(
#     [FollicleEnhance(), ToTensor(), #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.RandomHorizontalFlip(),
#      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])

trans_0 = transforms.Compose(
    [
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(226),
        transforms.CenterCrop(224),
    ]
)
trans_1 = transforms.Compose(
    [
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(226),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
        transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])),
        transforms.CenterCrop(224),
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
    batch_size=7,
    num_workers=4,
    oversample=False,
    oversample_amt=1,
)
# dm = TrachomaDataModule(img_dir_m, img_keys_m, transforms_0=trans_0, transforms_1=trans_1, batch_size=10, num_workers=4, oversample=False, oversample_amt=0.5, split=True)
# trans_0 = transforms.Compose(
#     [ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.CenterCrop(224)])
# trans_1 = transforms.Compose(
#     [ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.RandomHorizontalFlip(),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
# dm = TrachomaDataModule(img_dir_o, img_keys_o, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
#                         batch_size=6, num_workers=4, oversample=0.5, normalize=True)

# trans_3 = transforms.Compose(
#     [ToTensor(), transforms.Resize(224)])
# dm3 = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_3,
#                         batch_size=2, num_workers=4, oversample=False, split=False)
# dm3.setup()
# test_true = dm3.test_dataloader()
# img_dir = 'm'
# img_keys = 'm/tfti.csv'
#
# img_dir_tf = 'synthetic_TF_images'
# img_dir_non_tf = 'synthetic_NonTF_images'
#
# trans_0 = transforms.Compose(
#     [FollicleEnhance(), ToTensor(),
#      transforms.Normalize(mean=[0.4324, 0.2903, 0.2679], std=[0.1566, 0.1189, 0.1178]),
#      transforms.Resize(226),
#      transforms.CenterCrop(224)])
#
# trans_1 = transforms.Compose(
#     [FollicleEnhance(), ToTensor(),
#      transforms.Normalize(mean=[0.4324, 0.2903, 0.2679], std=[0.1566, 0.1189, 0.1178]), transforms.Resize(226), transforms.RandomHorizontalFlip(),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224)])
#
# dm = TrachomaDataModule(img_dir, img_keys, img_dir_tf, img_dir_non_tf, transforms_0=trans_0, transforms_1=trans_0,
#                             batch_size=10, num_workers=4, oversample=False, split=True, synthDataPercent=0.3)

dm.setup()
test_data = dm.test_dataloader()
print("test", len(test_data))
test_data.dataset.name = True

# vgg16 = models.vgg11_bn()
# # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
# print(vgg16.classifier[6].out_features)  # 1000
#
# # Newly created modules have require_grad=True by default
# num_features = vgg16.classifier[6].in_features
# features = list(vgg16.classifier.children())[:-1]  # Remove last layer
# features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
# vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
# print(vgg16)

res101 = models.resnet101(pretrained=True)
# vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
# print(res101)  # 1000

# Newly created modules have require_grad=True by default
num_features = res101.fc.in_features
# features = list(res101.classifier.children())[:-1]  # Remove last layer
# features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
# print(res101)

# classifier = TrachomaClassifier(model=vgg16)
# state_dict = torch.load(path)

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

# if we want to continue training from checkpoint model:
# trainer.fit(model, datamodule=dm, ckpt_path=path)
# #
# # # test
# trainer.test(datamodule=dm, model=model, ckpt_path='best')


preds = []
target = []
names = []

bad = []
bad_name = []
bad_true = []
good = []
good_true = []

r = pd.DataFrame(columns=["image", "True CLass", "Predicted Class"])


fp = []
dir = "/media/dsocia22/T7/Trachoma/BestModelResults_CombinedData_Images"

model = TrachomaClassifier.load_from_checkpoint(path, model=res101, strict=True)
model.eval()
device = torch.device("cpu")
model = model.to(device)

confmat = torchmetrics.ConfusionMatrix(task="binary", threshold=thresh)
prc = torchmetrics.PrecisionRecallCurve(task="binary")
rec = torchmetrics.Recall(task="binary", threshold=thresh)
pre = torchmetrics.Precision(task="binary", threshold=thresh)
f1 = torchmetrics.F1Score(task="binary", threshold=thresh)
kappa = torchmetrics.CohenKappa(task='binary', threshold=thresh)

with torch.no_grad():
    for batch in test_data:
        # print('B', batch['label'], 'T', t['label'])

        images = batch["image"]
        labels = batch["label"]
        out = model(images).squeeze()
        # pred = out.detach()
        # pred = torch.softmax(out, -1).detach()
        pred = torch.sigmoid(out)
        rec.update(pred, labels)
        pre.update(pred, labels)
        confmat.update(pred, labels)
        prc.update(pred, labels)
        f1.update(pred, labels)
        kappa.update(pred, labels)

        for i, l in enumerate(labels):
            # if pred[i] > thresh:
            r.loc[len(r.index)] = [
                str(batch["name"][i]),
                int(l.cpu()),
                1 if pred[i] > thresh else 0,
            ]
            # if pred[i] > thresh:
            # path = os.path.join(dir, str(batch['name'][i]))
            # img_path = batch['path'][i]
            # image = io.imread(img_path)
            # io.imsave(path, image)

        #     if l == 1:
        #         if pred[i] < thresh:
        #             bad.append(batch['image'][i].permute(1, 2, 0))
        #             # bad_true.append(t['image'][i].permute(1, 2, 0))
        #             bad_name.append(batch['name'][i])
        #         else:
        #             good.append(batch['image'][i].permute(1, 2, 0))
        #             # good_true.append(t['image'][i].permute(1, 2, 0))
        #     else:
        #         if pred[i] > thresh:
        #             fp.append(batch['image'][i].permute(1, 2, 0))
        # plt.imshow(batch['image'][i].permute(1, 2, 0))
        # plt.show()
        # pred = out.detach()
        # l = [int(x) for x in batch['label']]
        preds.extend(pred.numpy())
        target.extend(labels.numpy())


r.to_csv("BestModelResults_CombinedData.csv", index="False")
#
cm = confmat.compute().numpy()
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

fig = plt.figure()
precision, recall, thresholds = prc.compute()

plt.plot(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
print(thresholds)
print(recall)
print(precision)

print(pre.compute(), rec.compute(), f1.compute(), kappa.compute())


fig = plt.figure()
plt.scatter(target, preds)
plt.plot([0, 1], [thresh, thresh])
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
