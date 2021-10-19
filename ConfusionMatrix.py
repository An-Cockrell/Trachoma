import os
import copy
import gc
import time
import csv
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
import torchmetrics

from DataLoader_lightning import TrachomaDataModule, ToTensor, FollicleEnhance, CustomCrop
from Training_lightning import TrachomaClassifier

# np.random.seed(100)

# img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
img_keys = '2300consensus8-2021.csv'
# img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
path = '/home/dsocia22/Documents/Trachoma/TrainedModels/Pytorch_lightning_consensus_oversample5_posweight4_follicleenhance_flip_rotate_norm_pretrained_accum5batch_resnet101/epoch=8-step=323.ckpt'
thresh = 0.5
print(path, thresh)
trans_0 = transforms.Compose(
    [FollicleEnhance(), ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize(226),
     transforms.CenterCrop(224)])
trans_1 = transforms.Compose(
    [FollicleEnhance(), ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize(226),
     transforms.RandomHorizontalFlip(),
     # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
     transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_0,
                        batch_size=6, num_workers=4, oversample=False)

trans_3 = transforms.Compose(
    [ToTensor()])
dm3 = TrachomaDataModule(img_dir, img_keys, 'imagename', 'consensus', transforms_0=trans_3,
                        batch_size=6, num_workers=4, oversample=False)
dm3.setup()
test_true = dm3.test_dataloader()

dm.setup()
test_data = dm.test_dataloader()
test_data.dataset.name = True

vgg16 = models.vgg11_bn()
# vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
print(vgg16.classifier[6].out_features)  # 1000

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
# print(vgg16)

res101 = models.resnet101(pretrained=True)
# vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
# print(res101)  # 1000

# Newly created modules have require_grad=True by default
num_features = res101.fc.in_features
#features = list(res101.classifier.children())[:-1]  # Remove last layer
#features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
# print(res101)

# classifier = TrachomaClassifier(model=vgg16)
# state_dict = torch.load(path)
model = TrachomaClassifier.load_from_checkpoint(path, model=res101, strict=False)

#
trainer = pl.Trainer(num_processes=1, log_every_n_steps=2, max_epochs=1, default_root_dir='Training_Checkpoints', resume_from_checkpoint=path)
# #
# # # test
# trainer.test(datamodule=dm, model=model, ckpt_path='best')

confmat = torchmetrics.ConfusionMatrix(2, threshold=thresh)
prc = torchmetrics.BinnedPrecisionRecallCurve(num_classes=1, thresholds=[0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

rec = torchmetrics.Recall(num_classes=1, threshold=thresh)
pre = torchmetrics.Precision(num_classes=1, threshold=thresh)

model.eval()

preds = []
target = []
names = []

bad = []
bad_name = []
bad_true = []
good = []
good_true = []

fp = []
for batch, t in zip(test_data, test_true):
    # print('B', batch['label'], 'T', t['label'])

    out = model(batch['image']).squeeze()

    pred = torch.sigmoid(out).detach()
    L = batch['label']
    for i, l in enumerate(L):
        if l == 1:
            if pred[i] < thresh:
                bad.append(batch['image'][i].permute(1, 2, 0))
                bad_true.append(t['image'][i].permute(1, 2, 0))
                bad_name.append(batch['name'][i])
            else:
                good.append(batch['image'][i].permute(1, 2, 0))
                good_true.append(t['image'][i].permute(1, 2, 0))
        else:
            if pred[i] > 0.3:
                fp.append(batch['image'][i].permute(1, 2, 0))
            # plt.imshow(batch['image'][i].permute(1, 2, 0))
            # plt.show()
    # pred = out.detach()
    rec.update(pred, batch['label'])
    pre.update(pred, batch['label'])
    # print(batch['label'].detach(), pred)
    preds.extend(pred)
    target.extend(batch['label'].detach())
    confmat.update(pred, batch['label'])
    prc.update(pred, batch['label'])


#
print(confmat.compute())
disp = ConfusionMatrixDisplay(confmat.confmat.detach().numpy())
disp.plot()

fig = plt.figure()
precision, recall, thresholds = prc.compute()

plt.plot(recall, precision)
plt.ylabel('Precision')
plt.xlabel('Recall')
print(thresholds)
print(recall)
print(precision)

print(pre.compute(), rec.compute())


fig = plt.figure()
plt.scatter(target, preds)
plt.plot([0, 1], [thresh, thresh])

fig, ax = plt.subplots(1, len(bad))
for i, im in enumerate(bad):
    # ax[0, i].imshow(im)
    ax[i].imshow(bad_true[i])
    ax[i].axis('off')
    ax[i].axis('off')
fig.suptitle('False Negatives')

fig, ax = plt.subplots(1, len(good))
for i, im in enumerate(good):
    # ax[0, i].imshow(im)
    ax[i].imshow(good_true[i])
    ax[i].axis('off')
    ax[i].axis('off')
fig.suptitle('True Positives')

fig, ax = plt.subplots(1, len(fp))
for i, im in enumerate(fp):
    ax[i].imshow(im)
    ax[i].axis('off')

print(bad_name)
plt.show()



