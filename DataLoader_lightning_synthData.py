import os
import copy
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

np.random.seed(1234)


class TrachomaDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, img_keys_csv,  transforms_0=None, transforms_1=None, oversample=False, normalize=False, batch_size=32, num_workers=0, oversample_amt=0.2, dataPercent=1.0, split=True):
        super().__init__()
        self.img_dir = img_dir
        self.img_keys_csv = img_keys_csv
        self.dataPer = dataPercent

        self.transforms_0 = transforms_0
        self.transforms_1 = transforms_1 if transforms_1 is not None else transforms_0
        self.oversample = oversample
        self.oversample_amt = oversample_amt
        self.norm = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split= split

        self.get_keys()

    def get_keys(self):
        labels = pd.read_csv(self.img_keys_csv, sep=',')

        # remove ti images
        labels = labels[~((labels['TI'] == 1) & (labels['TF'] == 0))][['key', 'TF']]


        # changes labels from no TF and TF to 0 and 1 respectively
        # labels = labels.replace({self.key_col: self.map})
        labels = np.asarray(labels.values)

        labels = labels[:int(len(labels) * self.dataPer), :]

        # shuffle dataset
        # np.random.shuffle(labels)
        if self.split:
            # splits No TF and TF for equal distribution in training and testing and validation
            no_tf = labels[labels[:, 1] == 0]
            tf = labels[labels[:, 1] == 1]

            # test train
            train_no_tf_num = int(len(no_tf) * .8)
            train_tf_num = int(len(tf) * .8)

            train_no_tf = copy.deepcopy(no_tf[:train_no_tf_num, :])
            train_tf = copy.deepcopy(tf[:train_tf_num, :])

            test_no_tf = copy.deepcopy(no_tf[train_no_tf_num:, :])
            test_tf = copy.deepcopy(tf[train_tf_num:, :])

            # train validate
            train_no_tf_num = int(len(train_no_tf) * .8)
            train_tf_num = int(len(train_tf) * .8)

            val_no_tf = copy.deepcopy(train_no_tf[train_no_tf_num:, :])
            val_tf = copy.deepcopy(train_tf[train_tf_num:, :])

            train_no_tf = copy.deepcopy(train_no_tf[:train_no_tf_num, :])
            train_tf = copy.deepcopy(train_tf[:train_tf_num, :])

            # over sample TF images to improve the class distribution
            if self.oversample:
                # number of new sample to generate  - want 20% tf class
                num_gen = int(train_no_tf_num * self.oversample_amt) - train_tf_num
                random_ind = np.random.choice(train_tf_num, num_gen)
                train_tf = np.vstack((train_tf, train_tf[random_ind]))

            # combind tf and no tf
            train = np.vstack((train_tf, train_no_tf))
            val = np.vstack((val_tf, val_no_tf))
            test = np.vstack((test_tf, test_no_tf))

            # train2 = copy.deepcopy(np.vstack((train_tf, train_no_tf)))
            # val2 = copy.deepcopy(np.vstack((val_tf, val_no_tf)))
            # test2 = copy.deepcopy(np.vstack((test_tf, test_no_tf)))

            # np.random.shuffle(train)
            # np.random.shuffle(val)
            # np.random.shuffle(test)

            # assert set(map(tuple, train)) == set(map(tuple, train2))
            # assert set(map(tuple, val)) == set(map(tuple, val2))
            # assert set(map(tuple, test)) == set(map(tuple, test2))
        else:
            train = None
            val = None
            test = labels

        self.train_keys = train
        self.val_keys = val
        self.test_keys = test

    def setup(self, stage=None):
        # transforms
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # split dataset
        # if stage in (None, "fit"):
        self.trachoma_train = TrachomaDataset(self.img_dir, self.train_keys, transform=self.transforms_1)
        self.trachoma_val = TrachomaDataset(self.img_dir, self.val_keys, transform=self.transforms_0)
        # if stage == (None, "test"):
        self.trachoma_test = TrachomaDataset(self.img_dir, self.test_keys, transform=self.transforms_0)

        if self.norm:
            mean, std = self.normalize()
            trans = transforms.Normalize(mean=mean, std=std)
            self.transforms_0 = transforms.Compose([self.transforms_0, trans])
            self.transforms_1 = transforms.Compose([self.transforms_1, trans])

            self.trachoma_train.transform = self.transforms_1
            self.trachoma_val.transform = self.transforms_0
            self.trachoma_test.transform = self.transforms_0

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

    def test_dataloader(self):
        trachoma_test = DataLoader(self.trachoma_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=False)
        return trachoma_test


class TrachomaDataset(Dataset):
    def __init__(self, img_dir, img_keys, transform=None, name=False):
        super().__init__()
        self.img_dir = img_dir

        self.transform = transform

        self.img_keys = img_keys
        self.name = name

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.img_dir, 'image' + str(self.img_keys[item, 0])) + '.jpg'
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
    img_dir = 'm'
    img_keys = 'm/tfti.csv'

    trans_0 = transforms.Compose(
        [FollicleEnhance(addon=True), ToTensor(), #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

         transforms.Resize(256),
         transforms.CenterCrop(224)])
    trans_1 = transforms.Compose(
        [ToTensor(), transforms.Resize(256), transforms.RandomHorizontalFlip(),
         transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(15)])),
         transforms.RandomApply(nn.ModuleList([transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dm = TrachomaDataModule(img_dir, img_keys, 'imagename', 'ans_ground', transforms_0=trans_0,  transforms_1=trans_0,
                            batch_size=1, num_workers=1, oversample=False, split=False)

    dm.setup()
    data = dm.test_dataloader()
    print(len(data))
    for batch in data:
        # if batch['label'] == 1:
        img = batch['image'].squeeze()
        # print(img.size())
        img = img.permute(1, 2, 0)
        plt.imshow(img)
        plt.title(batch['label'])
        plt.show()
