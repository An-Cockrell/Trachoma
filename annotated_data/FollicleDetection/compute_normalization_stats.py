"""
Compute per-channel mean and std on the training split of SAMFollicleMasks
using the same seed and 80/20 split as the DataLoader, but without oversampling.
Run from: annotated_data/FollicleDetection/
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import io
from torch.utils.data import Dataset, DataLoader


IMG_DIR = "./MultipleFollicleImages/SAMFollicleMasks/Images"


class ImageOnlyDataset(Dataset):
    def __init__(self, img_dir, img_names, transform):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = io.imread(os.path.join(self.img_dir, self.img_names[idx]))
        img = img.transpose((2, 0, 1)) / 255.0  # ToTensor
        img = torch.from_numpy(img).float()
        img = self.transform(img)
        return img


def main():
    images = [
        f for f in os.listdir(IMG_DIR)
        if os.path.isfile(os.path.join(IMG_DIR, f))
    ]
    img_ind = np.arange(len(images)).tolist()

    np.random.seed(1234)
    test_ind = np.random.choice(img_ind, int(len(img_ind) * 0.2), replace=False).tolist()
    train_ind = [i for i in img_ind if i not in test_ind]
    train_imgs = [images[i] for i in train_ind]

    print(f"Total images:    {len(images)}")
    print(f"Training images: {len(train_imgs)}")

    transform = transforms.Compose([
        transforms.Resize(540),
        transforms.CenterCrop(520),
    ])

    dataset = ImageOnlyDataset(IMG_DIR, train_imgs, transform)
    loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)

    mean = torch.zeros(3)
    var = torch.zeros(3)
    n = 0

    print("Computing stats...")
    for i, batch in enumerate(loader):
        b = batch.view(batch.size(0), batch.size(1), -1)
        mean += torch.mean(b, 2).sum(0)
        var += torch.var(b, 2).sum(0)
        n += batch.size(0)
        if (i + 1) % 10 == 0:
            print(f"  {n}/{len(train_imgs)} images processed")

    mean /= n
    std = torch.sqrt(var / n)

    print(f"\nmean = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"std  = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("\nAdd to trans_0 and trans_1 in the training script:")
    print(f"transforms.Normalize(mean=[{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}],")
    print(f"                     std=[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}])")


if __name__ == "__main__":
    main()
