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
import wandb
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

from DataLoader_lightning_Gradablearea import TrachomaDataModule, ToTensor
from Training_lightning_gradeableArea import TrachomaGradableArea

# np.random.seed(100)

img_dir = './GradableAreaData/Images'
mask_dir = './GradableAreaData/Mask'

path = '/media/dsocia22/T7/Trachoma/annotated_data/Checkpoints/Segmentation_Test/last.ckpt'

trans_0 = transforms.Compose(
    [ToTensor(),
     transforms.Resize(540),
     transforms.CenterCrop(520)])
trans_1 = [transforms.Compose(
    [ToTensor(), transforms.Resize(540),
     transforms.RandomApply(nn.ModuleList(
         [transforms.RandomVerticalFlip(.5), transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(90),
          transforms.RandomPerspective(0.3)])), transforms.CenterCrop(520)
     ]), transforms.ColorJitter(1, 1, 1, 0.5)]

dm = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0, transforms_1=trans_1,
                        batch_size=1, num_workers=1, oversample=True, oversample_amt=50, normalize=True)

fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
# print(fcn)
# segModel = TrachomaGradableArea(fcn)

dm.setup()
test_data = dm.test_dataloader()

print('Here')
model = TrachomaGradableArea.load_from_checkpoint(path, model=fcn, strict=True)

#
trainer = pl.Trainer(num_processes=1, log_every_n_steps=2, max_epochs=1, default_root_dir='Training_Checkpoints', resume_from_checkpoint=path)
# #
# # # test
# trainer.test(datamodule=dm, model=model, ckpt_path='best')

model.eval()

# for batch, t in zip(test_data):
# fig1, ax1 = plt.subplots(4, 4)

fig = plt.figure()

gs = gridspec.GridSpec(4, 4,
         wspace=0, hspace=0.05, top=0.95, bottom=0.05, left=0.15, right=0.85)
ims = []
ts = []
ous = []
oms = []
# fig2, ax2 = plt.subplots(4, 4)
for i, batch in enumerate(test_data):
    images, targets = batch['image'], batch['label']

    outputs = model(images)['out'].squeeze().detach().numpy()
    im = images.squeeze().permute(1, 2, 0)
    outMask = outputs >= 0
    # fig, ax = plt.subplots(1, 3)
    if i < 4:

        # im = np.random.rand(28, 28)
        ax = plt.subplot(gs[i, 0])
        ax.imshow(im)
        ims.append(im)
        # ax1[i, 0].imshow(im)
        ax.axis('off')
        ax = plt.subplot(gs[i, 1])
        ax.imshow(targets.squeeze())
        ts.append(targets.squeeze())
        ax.axis('off')
        ax = plt.subplot(gs[i, 2])
        ax.imshow(outputs)
        ous.append(outputs)
        ax.axis('off')
        ax = plt.subplot(gs[i, 3])
        ax.imshow(outMask)
        oms.append(oms)
        ax.axis('off')
    else:
        break
    #     ax2[i-4, 0].imshow(im)
    #     ax2[i-4, 0].axis('off')
    #     ax2[i-4, 1].imshow(targets.squeeze())
    #     ax2[i-4, 1].axis('off')
    #     ax2[i-4, 2].imshow(outputs)
    #     ax2[i-4, 2].axis('off')
    #     ax2[i-4, 3].imshow(outMask)
    #     ax2[i-4, 3].axis('off')

# ax1[0, 0].set_title('Normalized Image')
# ax1[0, 1].set_title('Target')
# ax1[0, 2].set_title('Model Output')
# ax1[0, 3].set_title('Masked Output')
# ax2[0, 0].set_title('Normalized Image')
# ax2[0, 1].set_title('Target')
# ax2[0, 2].set_title('Model Output')
# ax2[0, 3].set_title('Masked Output')
# plt.axis('off')
# plt.tight_layout()
plt.show()

