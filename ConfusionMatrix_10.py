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
import torchmetrics

from DataLoader_lightning import TrachomaDataModule, ToTensor, FollicleEnhance, CustomCrop
from Training_lightning import TrachomaClassifier

from sklearn.metrics import recall_score, precision_score


# np.random.seed(100)

# img_dir = 'TrachomaData/allTZphotos/'  # unzipped file package contains more photos than entries in csv
img_dir_o = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
img_keys_o = '2300consensus8-2021.csv'
img_dir_m = 'm'
img_keys_m = 'm/tfti.csv'
# img_keys_o = 'TrachomaData/trachomagroundtruthkey.csv'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flip_rotate_perspective_pretrained_resnet101_mData/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_follicleenhance_flip_rotate_norm_pretrained_resnet101_allData/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample10_flip_rotate_perspective_pretrained_resnet101_allData/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_flip_rotate_perspective_pretrained_resnet101_mData_synthPretrain6/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning__follicleEnhance_flip_rotate_perspective_pretrained_resnet101_mData_synthPretrain2/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_follicleEnhance_flip_rotate_perspective_resnet101_m_synth_dp0.3/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flip_rotate_perspective_pretrained_resnet101_mData_5percent_3/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flop_rotate_perspective_pretrained_norm_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flop_rotate_perspective_pretrained_norm_accumuate3batches_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_follicleenhance_flip_rotate_pretrained_norm_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_posweight40_pretrained_norm_resnet101/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_posweight4_follicleenhance_flip_rotate_norm_pretrained_accum5batch_resnet18/last.ckpt'
thresh = 0.03

# trans_0 = transforms.Compose(
#         [FollicleEnhance(), ToTensor(),
#          #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#          transforms.Resize(226),
#          transforms.CenterCrop(224)])
# trans_1 = transforms.Compose(
#     [FollicleEnhance(), ToTensor(),  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.RandomHorizontalFlip(),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])

trans_0 = transforms.Compose(
    [ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize(226),
     transforms.CenterCrop(224)])
trans_1 = transforms.Compose(
    [ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize(226),
     transforms.RandomHorizontalFlip(),
     # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
     transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])

models_10 = []

# for i in range(10):
# path = 'TrainedModels/Pytorch_lightning_consensus_flip_rotate_perspective_pretrained_norm_resnet101_noPosWeight_{}/last.ckpt'.format(i)
path = 'Checkpoints/Pytorch_lightning_consensus_posweight40_pretrained_norm_resnet101_{}/last.ckpt'.format(8)

print(path, thresh)
#
# dm = TrachomaDataModule(img_dir_o, img_keys_o, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
#                                 batch_size=6, num_workers=4, oversample=True, oversample_amt=0.5, normalize=True)

dm = TrachomaDataModule(img_dir_o, img_keys_o, 'imagename', 'consensus', transforms_0=trans_0, transforms_1=trans_1,
                                batch_size=6, num_workers=4, oversample=False, oversample_amt=0.5, normalize=False)

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
test_data.dataset.name = True


res101 = models.resnet101(pretrained=True)


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
prc = torchmetrics.BinnedPrecisionRecallCurve(num_classes=1, thresholds=[0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

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

r = pd.DataFrame(columns=['image', 'True CLass', 'Predicted Class'])

fp = []
dir = '/media/dsocia22/T7/Trachoma/BestModelResults_ICAPS_Images'
# for batch, t in zip(test_data, test_true):
for batch in test_data:
    # print('B', batch['label'], 'T', t['label'])

    out = model(batch['image']).squeeze()
    # pred = out.detach()
    # pred = torch.softmax(out, -1).detach()
    pred = torch.sigmoid(out).detach()
    L = batch['label']
    for i, l in enumerate(L):
        # if pred[i] > thresh:
        r.loc[len(r.index)] = [str(batch['name'][i]), int(batch['label'][i]), 1 if pred[i] > thresh else 0]
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
    rec.update(pred, batch['label'])
    pre.update(pred, batch['label'])
    # print(batch['label'].detach(), pred)
    preds.extend(pred)
    target.extend(batch['label'].detach())
    confmat.update(pred, batch['label'])
    prc.update(pred, batch['label'])


r.to_csv('BestModelResults_ICAPS.csv', index='False')
#
models_10.append([target, preds])
print(confmat.compute())
disp = ConfusionMatrixDisplay(confmat.confmat.detach().numpy())
disp.plot()
# plt.title()

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
# rs = []
# ps = []
# for m in models_10:
#     pre = [1 if x >= 0.03 else 0 for x in m[1]]
#     r = recall_score(m[0], pre)
#     p = precision_score(m[0], pre)
#     rs.append(r)
#     ps.append(p)
#
# print(np.mean(ps), np.std(ps), np.mean(rs), np.std(rs))