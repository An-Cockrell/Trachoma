import os, os.path
import copy
import json
import random
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
from PIL import Image

GA_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_GA_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    GA_NORM,
])


def _compute_ga_mask(ga_model, img_path):
    """Run ga_model on the image at img_path; return binary mask (uint8 0/255) at original resolution."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    device = next(ga_model.parameters()).device
    img_tensor = _GA_PREPROCESS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = ga_model(img_tensor)
    mask_512 = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
    return cv.resize(mask_512, (W, H), interpolation=cv.INTER_NEAREST)

# Global seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class TrachomaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir,
        mask_dir,
        transforms_0=None,
        transforms_1=None,
        oversample=False,
        normalize=False,
        batch_size=32,
        num_workers=0,
        oversample_amt=0.2,
        dataPercent=1.0,
        alternate_test_data_image_dir=None,
        alternate_test_data_mask_dir=None,
        ga_model=None,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ga_model = ga_model

        self.mask_dir_test = mask_dir
        self.img_dir_test = img_dir

        self.mask_dir_val = mask_dir
        self.img_dir_val = img_dir

        self.images = [
            name
            for name in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(img_dir, name))
        ]

        self.img_ind = np.arange(len(self.images)).tolist()

        # --- Three-way split: test (20%) / val (16%) / train (64%) ---
        # Test set: held out entirely, never used for early stopping or model selection.
        # Seed 8 preserved for backwards compatibility with previous runs.
        self.test_ind = (
            np.random.RandomState(8)
            .choice(self.img_ind, int(len(self.img_ind) * 0.2), replace=False)
            .tolist()
        )
        self.test_imgs = [self.images[i] for i in self.test_ind]
        print("Number of test images: ", len(self.test_imgs))

        if alternate_test_data_image_dir is not None:
            self.test_imgs = [
                name
                for name in os.listdir(alternate_test_data_image_dir)
                if os.path.isfile(os.path.join(alternate_test_data_image_dir, name))
            ]
            self.mask_dir_test = alternate_test_data_mask_dir
            self.img_dir_test = alternate_test_data_image_dir

        # Validation set: carved from remaining 80% (20% of remainder = ~16% of total).
        # Used for early stopping and LR scheduling. Never seen by the test set.
        remaining_ind = [i for i in self.img_ind if i not in self.test_ind]
        self.val_ind = (
            np.random.RandomState(SEED)
            .choice(remaining_ind, int(len(remaining_ind) * 0.2), replace=False)
            .tolist()
        )
        self.val_imgs = [self.images[i] for i in self.val_ind]
        print("Number of val images: ", len(self.val_imgs))

        self.train_ind = [i for i in remaining_ind if i not in self.val_ind]
        print("Number of train images: ", len(self.train_ind))

        self.dataPer = dataPercent

        self.transforms_0 = transforms_0
        self.transforms_1 = transforms_1 if not None else transforms_0
        if oversample:
            self.train_ind = np.random.choice(
                self.train_ind, int(len(self.train_ind) * oversample_amt), replace=True
            ).tolist()

        self.train_imgs = [self.images[i] for i in self.train_ind]

        self.norm = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Pre-compute GA crop masks once in the main process so workers never need the model.
        # ga_mask_cache maps filename -> uint8 mask (H, W) at original resolution.
        ga_mask_cache = {}
        if self.ga_model is not None:
            print("Pre-computing GA crop masks...")
            for fname in dict.fromkeys(self.train_imgs + self.val_imgs):
                ga_mask_cache[fname] = _compute_ga_mask(
                    self.ga_model, os.path.join(self.img_dir, fname)
                )
            for fname in dict.fromkeys(self.test_imgs):
                if fname not in ga_mask_cache:
                    ga_mask_cache[fname] = _compute_ga_mask(
                        self.ga_model, os.path.join(self.img_dir_test, fname)
                    )
            print(f"GA masks computed for {len(ga_mask_cache)} images.")

        self.trachoma_train = TrachomaDataset(
            self.img_dir, self.mask_dir, self.train_imgs, transform=self.transforms_1,
            augment_image=True, ga_mask_cache=ga_mask_cache,
        )
        # Validation uses its own held-out split (not the test set)
        self.trachoma_val = TrachomaDataset(
            self.img_dir,
            self.mask_dir,
            self.val_imgs,
            transform=self.transforms_0,
            ga_mask_cache=ga_mask_cache,
        )
        self.trachoma_test = TrachomaDataset(
            self.img_dir_test,
            self.mask_dir_test,
            self.test_imgs,
            transform=self.transforms_0,
            ga_mask_cache=ga_mask_cache,
        )

        if self.norm:
            mean, std = self.normalize()
            trans = transforms.Normalize(mean=mean, std=std)
            # self.transforms_0 = transforms.Compose([self.transforms_0, trans])
            self.transforms_0 = [self.transforms_0, trans]
            if isinstance(self.transforms_1, list):
                self.transforms_1[1] = transforms.Compose([self.transforms_1[1], trans])
            else:
                self.transforms_1 = [self.transforms_1, trans]

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

    # return the dataloader for each split
    def train_dataloader(self):
        trachoma_train = DataLoader(
            self.trachoma_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
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
    def __init__(self, img_dir, mask_dir, imgs, transform=None, augment_image=False, ga_mask_cache=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = imgs
        self.ga_mask_cache = ga_mask_cache or {}
        # Image-only augmentations (color/blur) applied after spatial transforms.
        # Must NOT go in the stacked transform pipeline as that would corrupt the binary mask.
        self._image_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
        ]) if augment_image else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        fname = self.imgs[item]
        img_path = os.path.join(self.img_dir, fname)
        mask_fname = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_fname)
        image = io.imread(img_path)
        mask = Image.open(mask_path).convert("1")
        mask = np.array(mask).astype(int)

        # Apply CLAHE to each channel for consistent contrast enhancement at all splits
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = np.stack([clahe.apply(image[:, :, c]) for c in range(image.shape[2])], axis=2)

        if self.transform is not None:
            # Scale mask from {0,1} to {0,255} so the /255 in ToTensor maps it back to {0,1}.
            mask_255 = (mask * 255).astype(np.uint8)

            # Stack GA crop mask as an extra channel so it undergoes the same spatial
            # transforms (flips, rotation, crop) as image and follicle mask.
            ga_mask = self.ga_mask_cache.get(fname)
            if ga_mask is not None:
                temp = np.dstack((image, mask_255, ga_mask))  # (H, W, 5)
            else:
                temp = np.dstack((image, mask_255))           # (H, W, 4)

            if isinstance(self.transform, list):
                transformed = self.transform[0](temp)
                image = transformed[:3, :, :]
                mask = transformed[3, :, :]
                if ga_mask is not None:
                    ga_mask_t = transformed[4, :, :]           # [0, 1] after ToTensor /255
                    # Apply GA crop to [0,1] image before normalization
                    image = image * (ga_mask_t > 0.5).float()
                image = self.transform[1](image)
            else:
                transformed = self.transform(temp)
                image = transformed[:3, :, :]
                mask = transformed[3, :, :]
                if ga_mask is not None:
                    ga_mask_t = transformed[4, :, :]
                    image = image * (ga_mask_t > 0.5).float()

        # Apply image-only augmentations (color jitter, blur) after spatial transforms
        if self._image_aug is not None:
            image = self._image_aug(image)

        sample = {"image": image, "label": mask, "name": fname}

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
    mask_dir = "./MultipleFollicleImages/Masks"

    trans_0 = transforms.Compose(
        [ToTensor(), transforms.Resize(105), transforms.CenterCrop(100)]
    )
    trans_1 = transforms.Compose(
        [
            ToTensor(),
            transforms.Resize(105),
            transforms.RandomApply(
                nn.ModuleList(
                    [
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomRotation(90),
                        transforms.RandomPerspective(0.3),
                    ]
                )
            ),
            transforms.CenterCrop(100),
        ]
    )

    dm = TrachomaDataModule(
        img_dir,
        mask_dir,
        transforms_0=trans_0,
        transforms_1=trans_1,
        batch_size=1,
        num_workers=1,
        oversample=True,
        oversample_amt=1,
        normalize=True,
    )

    dm.setup()
    test_data = dm.val_dataloader()
    print(len(test_data))
    for batch in test_data:
        # if batch['label'] == 1:
        img = batch["image"].squeeze()
        mask = batch["label"].squeeze()
        # print(img.size())
        img = img.permute(1, 2, 0)
        tig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(mask)
        # plt.title(batch['label'])
        plt.show()
