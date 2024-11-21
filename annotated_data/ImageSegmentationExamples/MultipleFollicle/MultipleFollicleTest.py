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
from torchmetrics import Dice

from DataLoader_lightning_MultipleFollicle import TrachomaDataModule, ToTensor
from Training_lightning_MultipleFollicle import TrachomaGradableArea

# np.random.seed(100)

img_dir = './MultipleFollicleImages/Images'
mask_dir = './MultipleFollicleImages/Masks'

path = '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/Checkpoints/Segmentation_Test_MultipleFollicle_520/last.ckpt'

trans_0 = transforms.Compose(
        [ToTensor(),
         transforms.Resize(540),
         transforms.CenterCrop(520)])
trans_1 = transforms.Compose(
    [ToTensor(), transforms.Resize(540),
     transforms.RandomApply(nn.ModuleList(
         [transforms.RandomVerticalFlip(.5), transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(90),
          transforms.RandomPerspective(0.3)])), transforms.CenterCrop(520)
     ])

dm = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0, transforms_1=trans_1,
                            batch_size=1, num_workers=1, oversample=True, oversample_amt=20, normalize=True)
fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
# print(fcn)
# segModel = TrachomaGradableArea(fcn)

dm.setup()
test_data = dm.test_dataloader()
print('test', len(test_data))
print('training', len(dm.train_dataloader()))


model = TrachomaGradableArea.load_from_checkpoint(path, model=fcn, strict=True)

#
trainer = pl.Trainer(num_processes=1, log_every_n_steps=2, max_epochs=1, default_root_dir='Training_Checkpoints', resume_from_checkpoint=path)
# #
# # # test
# trainer.test(datamodule=dm, model=model, ckpt_path='best')

model.eval()


jaccard = JaccardIndex(task='multiclass', num_classes=2)
j = JaccardIndex(task='multiclass', num_classes=2)
d = Dice(num_classes=2)
dice = Dice(num_classes=2)

# # for batch, t in zip(test_data):
fig1, ax1 = plt.subplots(1, 4)
# fig2, ax2 = plt.subplots(1, 4)
# fig3, ax3 = plt.subplots(4, 4)
# fig4, ax4 = plt.subplots(4, 4)
r = []
for i, batch in enumerate(test_data):
    # if i == 16:
    #     break
    images, targets = batch['image'], batch['label']
    targets = targets.squeeze()

    outputs = model(images)['out'].squeeze().detach().numpy()
    im = images.squeeze().permute(1, 2, 0)
    outMask = outputs >= 0

    # jaccard.update(outMask.int(), targets.int())
    # dice.update(outMask.int(), targets.int())
    # print(j(outMask.int(), targets.int()), d(outMask.int(), targets.int()))
    #
    # outputs = outputs.detach().numpy()
    #
    # _, island_count_in = measure.label(targets, background=0, return_num=True, connectivity=1)
    # _, island_count_out = measure.label(outMask, background=0, return_num=True, connectivity=1)
    # print(island_count_in, island_count_out)
    # r.append(abs(island_count_in - island_count_out) / island_count_in)

    # fig, ax = plt.subplots(1, 3)
    if i == 0:
        ax1[0].imshow(im)
        ax1[0].axis('off')
        ax1[1].imshow(targets.squeeze())
        ax1[1].axis('off')
        ax1[2].imshow(outputs)
        ax1[2].axis('off')
        ax1[3].imshow(outMask)
        ax1[3].axis('off')
    # elif i==1:
    #     ax2[0].imshow(im)
    #     ax2[0].axis('off')
    #     ax2[1].imshow(targets.squeeze())
    #     ax2[1].axis('off')
    #     ax2[2].imshow(outputs)
    #     ax2[2].axis('off')
    #     ax2[3].imshow(outMask)
    #     ax2[3].axis('off')
    # if i < 4:
    #     ax1[i, 0].imshow(im)
    #     ax1[i, 0].axis('off')
    #     ax1[i, 1].imshow(targets.squeeze())
    #     ax1[i, 1].axis('off')
    #     ax1[i, 2].imshow(outputs)
    #     ax1[i, 2].axis('off')
    #     ax1[i, 3].imshow(outMask)
    #     ax1[i, 3].axis('off')
    # elif (i >= 4) & (i < 8):
    #     ax2[i-4, 0].imshow(im)
    #     ax2[i-4, 0].axis('off')
    #     ax2[i-4, 1].imshow(targets.squeeze())
    #     ax2[i-4, 1].axis('off')
    #     ax2[i-4, 2].imshow(outputs)
    #     ax2[i-4, 2].axis('off')
    #     ax2[i-4, 3].imshow(outMask)
    #     ax2[i-4, 3].axis('off')
    # elif (i >= 8) & (i < 12):
    #     ax3[i-8, 0].imshow(im)
    #     ax3[i-8, 0].axis('off')
    #     ax3[i-8, 1].imshow(targets.squeeze())
    #     ax3[i-8, 1].axis('off')
    #     ax3[i-8, 2].imshow(outputs)
    #     ax3[i-8, 2].axis('off')
    #     ax3[i-8, 3].imshow(outMask)
    #     ax3[i-8, 3].axis('off')
    # else:
    #     ax4[i - 12, 0].imshow(im)
    #     ax4[i - 12, 0].axis('off')
    #     ax4[i - 12, 1].imshow(targets.squeeze())
    #     ax4[i - 12, 1].axis('off')
    #     ax4[i - 12, 2].imshow(outputs)
    #     ax4[i - 12, 2].axis('off')
    #     ax4[i - 12, 3].imshow(outMask)
    #     ax4[i - 12, 3].axis('off')


# ax1[0, 0].set_title('Normalized Image')
# ax1[0, 1].set_title('Target')
# ax1[0, 2].set_title('Model Output')
# ax1[0, 3].set_title('Masked Output')
# ax2[0, 0].set_title('Normalized Image')
# ax2[0, 1].set_title('Target')
# ax2[0, 2].set_title('Model Output')
# ax2[0, 3].set_title('Masked Output')
# ax3[0, 0].set_title('Normalized Image')
# ax3[0, 1].set_title('Target')
# ax3[0, 2].set_title('Model Output')
# ax3[0, 3].set_title('Masked Output')
# ax4[0, 0].set_title('Normalized Image')
# ax4[0, 1].set_title('Target')
# ax4[0, 2].set_title('Model Output')
# ax4[0, 3].set_title('Masked Output')
# plt.axis('off')
plt.tight_layout()
plt.show()
#

# jaccard.compute()
print('total', jaccard.compute())
print('total Dice', dice.compute())

print(np.mean(r))