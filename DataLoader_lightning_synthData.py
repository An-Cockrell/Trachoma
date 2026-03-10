import os
import copy
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pytorch_lightning as pl
import cv2 as cv
from sklearn.model_selection import train_test_split

np.random.seed(1234)


class SynthTrachomaDataModule(pl.LightningDataModule):
    def __init__(self, img_dir_tf, img_dir_non_tf,  transforms=None, batch_size=32, num_workers=0, normalize=False):
        super().__init__()
        self.img_dir_tf = img_dir_tf
        self.img_dir_non_tf = img_dir_non_tf
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm = normalize

        self.get_keys()

    def get_keys(self):
        labels_tf = pd.DataFrame(glob.glob(self.img_dir_tf + '/*.jpg'), columns=['file'])
        labels_tf['TF'] = True

        labels_non_tf = pd.DataFrame(glob.glob(self.img_dir_non_tf + '/*.jpg'), columns=['file'])
        labels_non_tf['TF'] = False

        labels = pd.concat([labels_tf, labels_non_tf]).to_numpy()


        # shuffle dataset
        np.random.shuffle(labels)

        train, val = train_test_split(labels, random_state=0, test_size=.2)

        self.train_keys = train
        self.val_keys = val

    def setup(self, stage=None):
        # transforms
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # split dataset
        # if stage in (None, "fit"):
        self.trachoma_train = TrachomaDataset(self.train_keys, transform=self.transforms)
        self.trachoma_val = TrachomaDataset(self.val_keys, transform=self.transforms)

        if self.norm:
            mean, std = self.normalize()
            print(mean, std)
            trans = transforms.Normalize(mean=mean, std=std)
            self.transforms_0 = transforms.Compose([self.transforms, trans])

            self.trachoma_train.transform = self.transforms
            self.trachoma_val.transform = self.transforms

    def normalize(self):
        train = self.train_dataloader()
        nimages = 0
        mean = 0.0
        var = 0.0
        for i_batch, batch_target in enumerate(train):
            batch = batch_target['image']
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            nimages += batch.size(0)
            # Compute mean and std here
            mean += batch.mean(2).sum(0)
            var += batch.var(2).sum(0)

        mean /= nimages
        var /= nimages
        std = torch.sqrt(var)

        return mean, std

    # return the dataloader for each split
    def train_dataloader(self):
        trachoma_train = DataLoader(self.trachoma_train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True)
        return trachoma_train

    def val_dataloader(self):
        trachoma_val = DataLoader(self.trachoma_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=False)
        return trachoma_val


class TrachomaDataset(Dataset):
    def __init__(self, img_keys, transform=None, name=False):
        super().__init__()

        self.transform = transform

        self.img_keys = img_keys
        self.name = name

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = self.img_keys[item, 0]
        # image = cv.imread(img_path)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = io.imread(img_path)
        # assert image == image2

        if self.transform is not None:
            # print('transformed', self.transform)
            image = self.transform(image)

        sample = {'image': image, 'label': self.img_keys[item, 1]}

        if self.name:
            sample['name'] = self.img_keys[item, 0]
            return sample
        else:
            return sample


# Transformation classes
class ToTensor(object):
    """Converts numpy array to torch tensor"""

    def __call__(self, img):
        # numpy image: H x W x C
        # torch image: C x H x W
        # img = sample['image']
        img = img.transpose((2, 0, 1))/255
        return torch.from_numpy(img).float()


class CustomCrop(object):
    """Crops the Eyelid"""

    def __init__(self, rgb=True):
        self.rgb = rgb

    def __call__(self, img):
        # src = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        src = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        ret, thresh = cv.threshold(src[:, :, 2], 200, 255, cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(thresh, 1, 2)

        areas = []
        for cnt in contours:
            areas.append(cv.contourArea(cnt))
        if len(areas) > 0:
            max_area = max(areas)
            max_ind = areas.index(max_area)
            # print(max_area, max_ind)
            cnt = contours[max_ind]
            x, y, w, h = cv.boundingRect(cnt)
            if self.rgb:
                src = cv.cvtColor(src, cv.COLOR_YCrCb2BGR)

            crop_img = src[y:y + h, x:x + w]
        else:
            if self.rgb:
                src = cv.cvtColor(src, cv.COLOR_YCrCb2BGR)
            crop_img = src

        return crop_img


class FollicleEnhance(object):
    """Increases the contrast between the rest of the follicle and the eye"""

    def __init__(self, clipLimit=5.0, returnRGB=True, replace=None, sonly=False, addon=False):
        self.clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        self.rgb = returnRGB
        self.r = replace
        self.add = addon
        self.s=sonly

    def __call__(self, img):
        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        cl = self.clahe.apply(img_hsv[:, :, 1])

        if self.s:
            img_s = np.repeat(cl[:, :, np.newaxis], 3, axis=2)
            return img_s
        elif self.rgb:
            if self.r is None:
                if self.add:
                    return np.dstack((img, cl))
                else:
                    img_hsv[:, :, 1] = cl
                    return cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)
            else:
                img[:, :, self.r] = cl
                return img
        else:
            img_hsv[:, :, 1] = cl
            return img_hsv


if __name__ == '__main__':
    img_dir_tf = 'synthetic_TF_images'
    img_dir_non_tf = 'synthetic_NonTF_images'

    trans_0 = transforms.Compose(
        [ToTensor(), #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

         transforms.Resize(256),
         transforms.CenterCrop(224)])
    trans_1 = transforms.Compose(
        [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
         transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
         transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dm = SynthTrachomaDataModule(img_dir_tf, img_dir_non_tf, transforms=trans_0, batch_size=1, num_workers=1, normalize=True)

    dm.setup()
    data = dm.train_dataloader()
    print(len(data))
    for batch in data:
        # if batch['label'] == 1:
        img = batch['image'].squeeze()
        # print(img.size())
        img = img.permute(1, 2, 0)
        plt.imshow(img)
        plt.title(batch['label'])
        plt.show()
