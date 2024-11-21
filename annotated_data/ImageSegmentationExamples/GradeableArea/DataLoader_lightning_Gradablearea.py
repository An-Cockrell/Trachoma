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
    def __init__(self, img_dir, mask_dir, transforms_0=None, transforms_1=None, oversample=False, normalize=False, batch_size=32, num_workers=0, oversample_amt=0.2, dataPercent=1.0, alternate_test_data_image_dir=None, alternate_test_data_mask_dir=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.mask_dir_test = mask_dir
        self.img_dir_test = img_dir

        self.mask_dir_val = mask_dir
        self.img_dir_val = img_dir

        self.images = [name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(img_dir, name))]

        self.img_ind = np.arange(len(self.images)).tolist()

        self.test_ind = np.random.choice(self.img_ind, int(len(self.img_ind) * .2)).tolist()
        self.test_imgs = [self.images[i] for i in self.test_ind]

        if alternate_test_data_image_dir is not None:
            self.test_imgs = [name for name in os.listdir(alternate_test_data_image_dir) if os.path.isfile(os.path.join(alternate_test_data_image_dir, name))]

            self.mask_dir_test = alternate_test_data_mask_dir
            self.img_dir_test = alternate_test_data_image_dir

            self.mask_dir_val = alternate_test_data_mask_dir
            self.img_dir_val = alternate_test_data_image_dir

        # self.train

        self.train_ind = [i for i in self.img_ind if i not in self.test_ind]

        self.dataPer = dataPercent

        self.transforms_0 = transforms_0
        self.transforms_1 = transforms_1 if not None else transforms_0
        if oversample:
            self.train_ind = np.random.choice(self.train_ind, int(len(self.train_ind) * oversample_amt), replace=True).tolist()

        self.train_imgs = [self.images[i] for i in self.train_ind]

        self.norm = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        # transforms
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # split dataset
        # if stage in (None, "fit"):
        self.trachoma_train = TrachomaDataset(self.img_dir, self.mask_dir, self.train_imgs, transform=self.transforms_1)
        self.trachoma_val = TrachomaDataset(self.img_dir_val, self.mask_dir_val, self.test_imgs, transform=self.transforms_0)
        # if stage == (None, "test"):
        self.trachoma_test = TrachomaDataset(self.img_dir_test, self.mask_dir_test, self.test_imgs, transform=self.transforms_0)

        if self.norm:
            mean, std = self.normalize()
            trans = transforms.Normalize(mean=mean, std=std)
            # self.transforms_0 = transforms.Compose([self.transforms_0, trans])
            self.transforms_0 = [self.transforms_0, trans]
            self.transforms_1[1] = transforms.Compose([self.transforms_1[1], trans])

            self.trachoma_train.transform = self.transforms_1
            self.trachoma_val.transform = self.transforms_0
            self.trachoma_test.transform = self.transforms_0

    def normalize(self):
        train = self.train_dataloader()
        nimages = 0
        mean = 0.0
        var = 0.0
        for i_batch, batch_target in enumerate(train):
            # print(i_batch)
            batch = batch_target['image']
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
    def __init__(self, img_dir, mask_dir, imgs, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.transform = transform

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.img_dir, self.imgs[item])
        mask_path = os.path.join(self.mask_dir, self.imgs[item])
        # image = cv.imread(img_path)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # print(img_path)
        # print(mask_path)
        image = io.imread(img_path)
        mask = io.imread(mask_path)
        # assert image == image2
        # print('here')

        if self.transform is not None:
            # print('transformed', self.transform)
            temp = np.dstack((image, mask))

            if isinstance(self.transform, list):
                transformed_images = self.transform[0](temp)

                image = transformed_images[:3, :, :]
                mask = transformed_images[-1, :, :]

                image = self.transform[1](image)
            else:
                transformed_images = self.transform(temp)

                image = transformed_images[:3, :, :]
                mask = transformed_images[-1, :, :]

        sample = {'image': image, 'label': mask, 'name': self.imgs[item]}

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


# class RandomChoice(torch.nn.Module):
#     def __init__(self, transforms):
#        super().__init__()
#        self.transforms = transforms
#
#     def __call__(self, imgs):
#         t = random.choice(self.transforms)
#         return [t(img) for img in imgs]


if __name__ == '__main__':
    img_dir = './GradableAreaData/Images'
    mask_dir = './GradableAreaData/Mask'

    trans_0 = transforms.Compose(
        [ToTensor(),
         transforms.Resize(256),
         transforms.CenterCrop(224)])
    trans_1 = [transforms.Compose(
        [ToTensor(), transforms.Resize(256),
         transforms.RandomApply(nn.ModuleList([transforms.RandomVerticalFlip(.5), transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(90), transforms.RandomPerspective(0.3)])), transforms.CenterCrop(224)
         ]), transforms.ColorJitter(1, 1, 1, 0.5)]

    dm = TrachomaDataModule(img_dir, mask_dir, transforms_0=trans_0,  transforms_1=trans_1,
                            batch_size=1, num_workers=1, oversample=True, oversample_amt=1, normalize=True,
                            alternate_test_data_image_dir='GradableAreaData/Not_yet_used_in_training/Chris/Images',
                            alternate_test_data_mask_dir='GradableAreaData/Not_yet_used_in_training/Chris/Mask')

    dm.setup()
    test_data = dm.val_dataloader()
    print(len(test_data))
    for batch in test_data:
        # if batch['label'] == 1:
        img = batch['image'].squeeze()
        mask = batch['label'].squeeze()
        # print(img.size())
        img = img.permute(1, 2, 0)
        tig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(mask)
        # plt.title(batch['label'])
        plt.show()
