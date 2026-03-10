import os, os.path
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
import random

np.random.seed(1234)


class TrachomaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir_m,
        img_dir_o,
        img_keys_csv_m,
        img_keys_csv_o,
        img_col,
        key_col,
        transforms_0=None,
        transforms_1=None,
        oversample=False,
        normalize=False,
        batch_size=32,
        num_workers=0,
        oversample_amt=0.2,
        dataPercent=1.0,
        mapp=None,
        split=True,
    ):
        super().__init__()
        self.img_dir_m = img_dir_m
        self.img_keys_csv_m = img_keys_csv_m
        self.img_dir_o = img_dir_o
        self.img_keys_csv_o = img_keys_csv_o
        self.img_col = img_col
        self.key_col = key_col
        self.map = {"No TF": 0, "TF": 1} if mapp is None else mapp
        self.dataPer = dataPercent

        self.transforms_0 = transforms_0
        self.transforms_1 = transforms_1 if not None else transforms_0
        self.oversample = oversample
        self.oversample_amt = oversample_amt
        self.norm = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

        self.get_keys()

        # self.img_dir = img_dir

        # # self.mask_dir = mask_dir

        # # self.mask_dir_test = mask_dir
        # self.img_dir_test = img_dir

        # # self.mask_dir_val = mask_dir
        # self.img_dir_val = img_dir

        # self.images = [name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(img_dir, name))]

        # self.img_ind = np.arange(len(self.images)).tolist()

        # #deterministic random split
        # self.test_ind = np.random.RandomState(8).choice(self.img_ind, int(len(self.img_ind) * .2), replace=False).tolist()
        # self.test_imgs = [self.images[i] for i in self.test_ind]
        # print('Number of test images: ', len(self.test_imgs))

        # if alternate_test_data_image_dir is not None:
        #     self.test_imgs = [name for name in os.listdir(alternate_test_data_image_dir) if os.path.isfile(os.path.join(alternate_test_data_image_dir, name))]

        #     # self.mask_dir_test = alternate_test_data_mask_dir
        #     self.img_dir_test = alternate_test_data_image_dir

        #     # self.mask_dir_val = alternate_test_data_mask_dir
        #     self.img_dir_val = alternate_test_data_image_dir

        # # self.train

        # self.train_ind = [i for i in self.img_ind if i not in self.test_ind]
        # print('Number of train images: ', len(self.train_ind))

        # self.dataPer = dataPercent

        # self.transforms_0 = transforms_0
        # self.transforms_1 = transforms_1 if not None else transforms_0
        # if oversample:
        #     self.train_ind = np.random.choice(self.train_ind, int(len(self.train_ind) * oversample_amt), replace=True).tolist()

        # self.train_imgs = [self.images[i] for i in self.train_ind]

        # self.norm = normalize
        # self.batch_size = batch_size
        # self.num_workers = num_workers

    def get_keys(self):
        labels = pd.read_csv(self.img_keys_csv_m, sep=",")

        # remove ti images
        labels = labels[~((labels["TI"] == 1) & (labels["TF"] == 0))][["key", "TF"]]
        labels["dataset"] = "m"
        labels["img_path"] = labels["key"].apply(
            lambda x: os.path.join(self.img_dir_m, "image" + str(x)) + ".jpg"
        )

        labels_o = pd.read_csv(
            self.img_keys_csv_o, sep=",", header=0, usecols=[self.img_col, self.key_col]
        )

        # changes labels from no TF and TF to 0 and 1 respectively
        labels_o = labels_o.replace({self.key_col: self.map})
        labels_o.rename(columns={"imagename": "key", "consensus": "TF"}, inplace=True)
        labels_o["dataset"] = "o"
        labels_o["img_path"] = labels_o["key"].apply(
            lambda x: os.path.join(self.img_dir_o, "0" + str(x)) + ".jpg"
        )
        labels = pd.concat([labels, labels_o])
        # labels.to_csv("labels.csv", index=False)
        with open("clean_image_paths.txt", "r") as f:
            clean_img_paths = f.read().splitlines()
        labels = labels[labels["img_path"].isin(clean_img_paths)]

        labels = labels.drop(columns=["img_path"])

        # changes labels from no TF and TF to 0 and 1 respectively
        # labels = labels.replace({self.key_col: self.map})
        labels = np.asarray(labels.values)

        labels = labels[: int(len(labels) * self.dataPer), :]

        # shuffle dataset
        np.random.seed(1234)
        np.random.shuffle(labels)
        if self.split:
            # splits No TF and TF for equal distribution in training and testing and validation
            no_tf = labels[labels[:, 1] == 0]
            tf = labels[labels[:, 1] == 1]

            # test train
            train_no_tf_num = int(len(no_tf) * 0.8)
            train_tf_num = int(len(tf) * 0.8)

            train_no_tf = copy.deepcopy(no_tf[:train_no_tf_num, :])
            train_tf = copy.deepcopy(tf[:train_tf_num, :])

            test_no_tf = copy.deepcopy(no_tf[train_no_tf_num:, :])
            test_tf = copy.deepcopy(tf[train_tf_num:, :])

            # train validate
            train_no_tf_num = int(len(train_no_tf) * 0.8)
            train_tf_num = int(len(train_tf) * 0.8)

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

    # @profile
    def setup(self, stage=None):
        # transforms
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # split dataset
        # if stage in (None, "fit"):

        # apply the test transform to train data because we aren't training the model
        self.trachoma_train = TrachomaDataset(
            self.img_dir_m, self.img_dir_o, self.train_keys, transform=self.transforms_0
        )
        self.trachoma_val = TrachomaDataset(
            self.img_dir_m, self.img_dir_o, self.val_keys, transform=self.transforms_0
        )
        self.trachoma_test = TrachomaDataset(
            self.img_dir_m, self.img_dir_o, self.test_keys, transform=self.transforms_0
        )

        if self.norm:
            mean, std = self.normalize()
            # print(f' Mean: {mean}\n STD: {std}')
            # found offline and hard-coded the normalization because it takes so long to compute
            # mean = torch.tensor([0.5097, 0.3696, 0.3377])
            # std = torch.tensor([0.2159, 0.1906, 0.1824])
            trans = transforms.Normalize(mean=mean, std=std)
            # self.transforms_0 = transforms.Compose([self.transforms_0, trans])
            self.transforms_0 = [self.transforms_0, trans]
            if isinstance(self.transforms_1, list):
                self.transforms_1[1] = transforms.Compose([self.transforms_1[1], trans])
            else:
                self.transforms_1 = [self.transforms_1, trans]

            self.trachoma_train.transform = self.transforms_0
            self.trachoma_val.transform = self.transforms_0
            self.trachoma_test.transform = self.transforms_0

    def normalize(self):
        train = self.train_dataloader()
        nimages = 0
        mean = 0.0
        var = 0.0
        for i_batch, batch_target in enumerate(train):
            # print(i_batch)
            batch = batch_target["image"]
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            nimages += batch.size(0)
            # Compute mean and std here
            mean += torch.mean(batch, 2).sum(0)
            var += torch.var(batch, 2).sum(0)

        mean /= nimages
        var /= nimages
        std = torch.sqrt(var)

        return mean, std

    # @profile
    # def normalize(self):
    #     loader = self.train_dataloader()
    #     channel_sum = 0.0
    #     channel_squared_sum = 0.0
    #     n_pixels = 0

    #     # Disable gradient calculation for speed
    #     with torch.no_grad():
    #         for batch in loader:
    #             imgs = batch['image']  # shape: (B, C, H, W)
    #             b, c, h, w = imgs.shape
    #             imgs = imgs.view(b, c, -1)  # (B, C, H*W)

    #             # Sum and squared sum for each channel
    #             channel_sum += imgs.sum(dim=[0, 2])
    #             channel_squared_sum += (imgs ** 2).sum(dim=[0, 2])

    #             n_pixels += b * h * w

    #     mean = channel_sum / n_pixels
    #     std = torch.sqrt(channel_squared_sum / n_pixels - mean ** 2)

    #     return mean, std

    # return the dataloader for each split
    def train_dataloader(self):
        trachoma_train = DataLoader(
            self.trachoma_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
        )
        return trachoma_train

    def val_dataloader(self):
        trachoma_val = DataLoader(
            self.trachoma_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
        )
        return trachoma_val

    def test_dataloader(self):
        trachoma_test = DataLoader(
            self.trachoma_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
        )
        return trachoma_test


class TrachomaDataset(Dataset):
    def __init__(self, img_dir_m, img_dir_o, img_keys, transform=None, name=False):
        super().__init__()
        self.img_dir_m = img_dir_m
        self.img_dir_o = img_dir_o

        self.transform = transform

        self.img_keys = img_keys
        self.name = name

    def __len__(self):
        return len(self.img_keys)

    # @profile
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self.img_keys[item, 2] == "m":
            img_path = (
                os.path.join(self.img_dir_m, "image" + str(self.img_keys[item, 0]))
                + ".jpg"
            )
        else:
            img_path = (
                os.path.join(self.img_dir_o, "0" + self.img_keys[item, 0]) + ".jpg"
            )

        image = io.imread(img_path)

        if self.transform is not None:
            if isinstance(self.transform, list):
                image = self.transform[0](image)
                image = self.transform[1](image)
            else:
                image = self.transform(image)
        sample = {"image": image, "label": self.img_keys[item, 1]}

        if self.name:
            sample["path"] = img_path
            if self.img_keys[item, 2] == "m":
                sample["name"] = "image" + str(self.img_keys[item, 0]) + ".jpg"

            else:
                sample["name"] = "0" + self.img_keys[item, 0] + ".jpg"
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
        img = img.transpose((2, 0, 1)) / 255
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

            crop_img = src[y : y + h, x : x + w]
        else:
            if self.rgb:
                src = cv.cvtColor(src, cv.COLOR_YCrCb2BGR)
            crop_img = src

        return crop_img


class FollicleEnhance(object):
    """Increases the contrast between the rest of the follicle and the eye"""

    def __init__(
        self, clipLimit=5.0, returnRGB=True, replace=None, sonly=False, addon=False
    ):
        self.clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        self.rgb = returnRGB
        self.r = replace
        self.add = addon
        self.s = sonly

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


# class RandomChoice(torch.nn.Module):
#     def __init__(self, transforms):
#        super().__init__()
#        self.transforms = transforms
#
#     def __call__(self, imgs):
#         t = random.choice(self.transforms)
#         return [t(img) for img in imgs]


if __name__ == "__main__":
    img_dir = "./MultipleFollicleImages/Images"
    # mask_dir = './MultipleFollicleImages/Masks'

    # trans_0 = transforms.Compose(
    #     [ToTensor(),
    #      transforms.Resize(105),
    #      transforms.CenterCrop(100)])
    # trans_1 = transforms.Compose(
    #     [ToTensor(), transforms.Resize(105),
    #      transforms.RandomApply(nn.ModuleList([transforms.RandomVerticalFlip(.5), transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(90), transforms.RandomPerspective(0.3)])), transforms.CenterCrop(100)
    #      ])

    # dm = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0,  transforms_1=trans_1,
    #                         batch_size=1, num_workers=1, oversample=True, oversample_amt=1, normalize=True)

    # dm.setup()
    # test_data = dm.val_dataloader()
    # print(len(test_data))
    # for batch in test_data:
    #     # if batch['label'] == 1:
    #     img = batch['image'].squeeze()
    #     mask = batch['label'].squeeze()
    #     # print(img.size())
    #     img = img.permute(1, 2, 0)
    #     tig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img)
    #     ax[1].imshow(mask)
    #     # plt.title(batch['label'])
    #     plt.show()
