from __future__ import annotations
from typing import Tuple
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from .utils import IMAGENET_MEAN, IMAGENET_STD

class EnsureRGB:
    """Make sure PIL image is RGB."""
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

class ToTensor01:
    """Convert PIL to float tensor in [0,1]."""
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return transforms.functional.to_tensor(img)

def build_train_transforms(img_size: int) -> transforms.Compose:
    # Augmentations chosen to mimic common field issues: blur, exposure, glare-like contrast shifts, mild geometry.
    return transforms.Compose([
        EnsureRGB(),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.2),
        transforms.RandomApply([transforms.RandomAutocontrast()], p=0.2),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.15),
        transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.3),
        ToTensor01(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_eval_transforms(img_size: int) -> transforms.Compose:
    # Deterministic resize+center crop
    return transforms.Compose([
        EnsureRGB(),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        ToTensor01(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
