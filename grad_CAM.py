import os
import copy
import gc
import pandas as pd
import time
import cv2
import json
import numpy as np
import wandb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

from grad_CAM_resnet import resnet101

from DataLoader_lightning_mData import TrachomaDataModule, ToTensor, FollicleEnhance, CustomCrop
from Training_lightning_mData_synthPretrain import TrachomaClassifier

img_dir_o = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
img_keys_o = '2300consensus8-2021.csv'
img_dir_m = 'm'
img_keys_m = 'm/tfti.csv'
# img_keys = 'TrachomaData/trachomagroundtruthkey.csv'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_flip_rotate_perspective_pretrained_resnet101_mData/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample5_follicleenhance_flip_rotate_norm_pretrained_resnet101_allData/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_consensus_oversample10_flip_rotate_perspective_pretrained_resnet101_allData/last.ckpt'
path = 'TrainedModels/Pytorch_lightning_flip_rotate_perspective_pretrained_resnet101_mData_synthPretrain6/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning__follicleEnhance_flip_rotate_perspective_pretrained_resnet101_mData_synthPretrain2/last.ckpt'
# path = 'TrainedModels/Pytorch_lightning_follicleEnhance_flip_rotate_perspective_resnet101_m_synth_dp0.3/last.ckpt'
thresh = 0.1
print(path, thresh)
# trans_0 = transforms.Compose(
#     [FollicleEnhance(), ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.CenterCrop(224)])
# trans_1 = transforms.Compose(
#     [FollicleEnhance(), ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.RandomHorizontalFlip(),
#      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])

trans_0 = transforms.Compose(
    [ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize(226),
     transforms.CenterCrop(224)])
trans_1 = transforms.Compose(
    [ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize(226),
     transforms.RandomHorizontalFlip(),
     transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
     transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
#
# dm = TrachomaDataModule(img_dir_m, img_dir_o, img_keys_m, img_keys_o, 'imagename', 'consensus',
#                         transforms_0=trans_0, transforms_1=trans_1,
#                         batch_size=10, num_workers=4, oversample=True, oversample_amt=1)
dm = TrachomaDataModule(img_dir_m, img_keys_m,
                                                    transforms_0=trans_0, transforms_1=trans_1,
                                                    batch_size=1, num_workers=4, oversample=False, oversample_amt=0.5)
# trans_0 = transforms.Compose(
#     [ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.CenterCrop(224)])
# trans_1 = transforms.Compose(
#     [ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#      transforms.Resize(226),
#      transforms.RandomHorizontalFlip(),
#      # transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])),
#      transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(10)])), transforms.CenterCrop(224), ])
# dm = TrachomaDataModule(img_dir_o, img_keys_o, 'imagename', 'consensus', transforms_0=trans_0,
#                         batch_size=6, num_workers=4, oversample=0.5)
#
# trans_3 = transforms.Compose(
#     [ToTensor(), transforms.Resize(224)])
# dm3 = TrachomaDataModule(img_dir_m, img_keys_m,
#                                                     transforms_0=trans_3,
#                                                     batch_size=1, num_workers=4, oversample=False, oversample_amt=0.5)
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
#                             batch_size=1, num_workers=4, oversample=False, split=True, synthDataPercent=0.3)

dm.setup()
test_data = dm.test_dataloader()


# res101 = models.resnet101(pretrained=False)
#
#
# # Newly created modules have require_grad=True by default
# num_features = res101.fc.in_features
# #features = list(res101.classifier.children())[:-1]  # Remove last layer
# #features.extend([nn.Linear(num_features, 1)])  # Add our layer with 2 outputs
# res101.fc = nn.Linear(num_features, 1)  # Replace the model classifier
# print(res101)

res101 = resnet101()
print(res101)

# classifier = TrachomaClassifier(model=vgg16)
new_path = 'newPath.ckpt'
state_dict = torch.load(path, map_location=torch.device('cpu'))
temp = {}
for k, v in state_dict['state_dict'].items():
    # if 'fc' not in k:
    k = k[:6] + 'res101.' + k[6:]
    temp[k] = v

# state_dict['state_dict'] = {k[:6] + 'res101.' + k[6:]: v for k, v in state_dict['state_dict'].items() if 'fc' not in k}
state_dict['state_dict'] = temp
torch.save(state_dict, new_path)
model = TrachomaClassifier.load_from_checkpoint(new_path, model=res101, strict=True)
model.model.setup()
#
# #
trainer = pl.Trainer(num_processes=1, log_every_n_steps=2, max_epochs=1, default_root_dir='Training_Checkpoints', resume_from_checkpoint=path)

model.eval()
thresh = 0.1

# fig, ax = plt.subplots(5, 2)
# ax = ax.ravel()
k = 0
z = 0
# a = 0
preds = []
target = []
for j, batch in enumerate(test_data):
    # print('B', batch['label'], 'T', t['label'])
    img = batch['orig'].squeeze()
    out = model(batch['image'])
    # pred = out.detach()
    # pred = torch.softmax(out, -1).detach()
    pred = torch.sigmoid(out)


    title = None
    if pred <= thresh:
        title = 'Non-TF'
        # k += 1
    else:
        title = 'TF'
        # z += 1

    # pred = out.argmax(dim=1)
    pred.backward()

    gradients = model.model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.model.get_activations(batch['image']).detach()

    # weight the channels by corresponding gradients
    for i in range(2048):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    # plt.matshow(heatmap.squeeze())
    h = heatmap.squeeze()
    heatmap = heatmap.detach().numpy()

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    # heatmap = np.dstack([heatmap, np.zeros([224, 224]), np.zeros([224, 224])])
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    # superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.hstack([heatmap, img])

    plt.figure()
    plt.imshow(superimposed_img)
    plt.title(title)

    if j == 20:
        break

    # if (title == 'Non-TF') & (k<5):
    #     # plt.figure(k)
    #     if k == 0:
    #         ax[k, 0].set_title(title)
    #     ax[k, 0].imshow(superimposed_img)
    #     ax[k, 0].axis('off')
    #     ax[k, 0].set_aspect('equal')
    #     # ax[k, 1].matshow(h)
    #     k += 1
    #     # plt.title(title)
    #
    # if (title == 'TF') & (z<5):
    #     # plt.figure(k)
    #     if z == 0:
    #         ax[z, 1].set_title(title)
    #     ax[z, 1].imshow(superimposed_img)
    #     ax[z, 1].axis('off')
    #     ax[z, 0].set_aspect('equal')
    #     # ax[z, 3].matshow(h)
    #     z += 1

    preds.append(pred.detach())
    target.extend(batch['label'].detach())

    # if k > 10:
    #     break
# fig.tight_layout()
# fig.subplots_adjust(wspace=0, hspace=0)
plt.show()
# fig = plt.figure()
# plt.scatter(target, preds)
# plt.plot([0, 1], [thresh, thresh])