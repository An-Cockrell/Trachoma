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

path = '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/Checkpoints/Segmentation_Test_SingleFollicle_100/last.ckpt'

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

# dm = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0, transforms_1=trans_1,
#                         batch_size=1, num_workers=1, oversample=True, oversample_amt=10, normalize=True)

dm_chris = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0, transforms_1=trans_1,
                        batch_size=1, num_workers=1, oversample=True, oversample_amt=20, normalize=True,
                        alternate_test_data_image_dir='./MultipleFollicleImages/Test/Chris/Images',
                        alternate_test_data_mask_dir='./MultipleFollicleImages/Test/Chris/Masks')

dm_lindsay = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0, transforms_1=trans_1,
                        batch_size=1, num_workers=1, oversample=True, oversample_amt=20, normalize=True,
                                alternate_test_data_image_dir='./MultipleFollicleImages/Test/Lindsay/Images',
                                alternate_test_data_mask_dir='./MultipleFollicleImages/Test/Lindsay/Masks')

fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
print(fcn)
segModel = TrachomaGradableArea(fcn)
print('Here')
dm_chris.setup()
print('Here')
dm_lindsay.setup()
test_data_c = dm_chris.test_dataloader()
test_data_l = dm_lindsay.test_dataloader()


model = TrachomaGradableArea.load_from_checkpoint(path, model=fcn, strict=True)

#
# trainer = pl.Trainer(num_processes=1, log_every_n_steps=2, max_epochs=1, default_root_dir='Training_Checkpoints', resume_from_checkpoint=path)
# #
# # # test
# trainer.test(datamodule=dm, model=model, ckpt_path='best')

model.eval()

jaccard = JaccardIndex(task='multiclass', num_classes=2)
j = JaccardIndex(task='multiclass', num_classes=2)
d = Dice(num_classes=2)
dice = Dice(num_classes=2)

# for batch, t in zip(test_data):
# fig1, ax1 = plt.subplots(4, 4)
# fig2, ax2 = plt.subplots(4, 4)
print('Here')
r = pd.DataFrame(np.random.rand(len(test_data_c), 6), columns=['Chris_Lindsay_IoU', 'Chris_ML_IoU', 'Lindsay_ML_IoU','Chris_Lindsay_Dice', 'Chris_ML_Dice', 'Lindsay_ML_Dice'])

for i, (batch_c, batch_l) in enumerate(zip(test_data_c, test_data_l)):
    print(batch_c['name'], batch_l['name'])
    images, targets_c = batch_c['image'], batch_c['label']

    outputs = model(images)['out'].squeeze() #.detach().numpy()
    im = images.squeeze().permute(1, 2, 0)
    outMask_c = outputs >= 0
    # fig, ax = plt.subplots(1, 3)
    targets_c = targets_c.squeeze() #.detach().numpy()
    # inter = np.equal(outMask_c, targets_c)
    # diff_c_ml = inter.sum() / inter.size

    images, targets_l = batch_l['image'], batch_l['label']

    outputs = model(images)['out'].squeeze() #.detach().numpy()
    im = images.squeeze().permute(1, 2, 0)
    outMask_l = outputs >= 0
    # fig, ax = plt.subplots(1, 3)
    targets_l = targets_l.squeeze() #.detach().numpy()
    # inter = np.equal(outMask_l, targets_l)
    # diff_l_ml = inter.sum() / inter.size
    #
    # inter = np.equal(targets_c, targets_l)
    # diff_l_c = inter.sum() / inter.size

    jaccard.update(outMask_c.int(), targets_c.int())
    dice.update(outMask_c.int(), targets_c.int())
    jaccard.update(outMask_l.int(), targets_l.int())
    dice.update(outMask_l.int(), targets_l.int())
    r.iloc[i] = [j(targets_c.int(), targets_l.int()), j(outMask_c.int(), targets_c.int()), j(outMask_l.int(), targets_l.int()), d(targets_c.int(), targets_l.int()), d(outMask_c.int(), targets_c.int()), d(outMask_l.int(), targets_l.int())]

print('total', jaccard.compute())
print('total Dice', dice.compute())

