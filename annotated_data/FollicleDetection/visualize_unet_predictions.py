"""
Visualize FollicleUNet predictions vs ground-truth masks for the test set.

Saves a grid of (image | GT mask | predicted mask) for every test image.
Usage:
    cd annotated_data/FollicleDetection
    python visualize_unet_predictions.py
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../GradableAreaData"))
from Training_lightning_GradableArea_UNet import GradableAreaUNet

from DataLoader_lightning_MultipleFollicle import TrachomaDataModule, ToTensor
from Training_lightning_MultipleFollicle_UNet import FollicleUNet

# ── Paths ──────────────────────────────────────────────────────────────────────

CKPT_PATH = "./Checkpoints/MultipleFollicle_EfficientNetB4_UNet_GACrop/last.ckpt"
GA_CKPT   = "../GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"

IMG_DIR  = "./MultipleFollicleImages/RawImages"
MASK_DIR = "./MultipleFollicleImages/RawMasks"

OUT_DIR  = "./visualizations/unet_test_predictions"
THRESHOLD = 0.5

IMG_SIZE = 512
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load models ────────────────────────────────────────────────────────────────

def load_ga_model():
    ga = GradableAreaUNet.load_from_checkpoint(GA_CKPT, encoder="mobilenet_v2", lr=1e-3)
    ga.eval().to(DEVICE)
    return ga


def load_follicle_model():
    model = FollicleUNet.load_from_checkpoint(CKPT_PATH, encoder="efficientnet-b4", lr=1e-3)
    model.eval().to(DEVICE)
    return model


# ── Inverse-normalise for display ─────────────────────────────────────────────

_MEAN = torch.tensor(IMG_MEAN).view(3, 1, 1)
_STD  = torch.tensor(IMG_STD).view(3, 1, 1)

def denorm(t):
    """Return a HxWx3 uint8 numpy array from a normalised CHW tensor."""
    t = t.cpu() * _STD + _MEAN
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    trans_eval = [
        transforms.Compose([ToTensor(), transforms.Resize((IMG_SIZE, IMG_SIZE))]),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ]

    ga_model = load_ga_model()

    num_workers = min(max(os.cpu_count() - 2, 1), 4)
    dm = TrachomaDataModule(
        IMG_DIR,
        MASK_DIR,
        transforms_0=trans_eval,
        transforms_1=trans_eval,
        batch_size=4,
        num_workers=num_workers,
        oversample=False,
        normalize=False,
        ga_model=ga_model,
    )
    dm.setup()

    follicle_model = load_follicle_model()

    test_loader = dm.test_dataloader()
    n_test = len(dm.trachoma_test)
    print(f"Visualising {n_test} test images → {OUT_DIR}")

    img_idx = 0
    for batch in test_loader:
        images  = batch["image"].to(DEVICE)   # (B, 3, H, W)  normalised
        targets = batch["label"]               # (B, 1, H, W)  or (B, H, W)
        names   = batch["name"]

        with torch.no_grad():
            logits = follicle_model(images).squeeze(1)   # (B, H, W)
            preds  = (torch.sigmoid(logits) > THRESHOLD).float().cpu()

        targets = targets.squeeze(1).cpu()  # (B, H, W)

        for i in range(images.size(0)):
            img_np  = denorm(images[i].cpu())
            gt_np   = targets[i].numpy()
            pred_np = preds[i].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Image")

            axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth")

            axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"Prediction (t={THRESHOLD})")

            for ax in axes:
                ax.axis("off")

            fig.suptitle(names[i], fontsize=9)
            fig.tight_layout()

            stem = os.path.splitext(names[i])[0]
            out_path = os.path.join(OUT_DIR, f"{img_idx:04d}_{stem}.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            img_idx += 1

    print(f"Saved {img_idx} visualisations to {OUT_DIR}/")

    # Also save a single summary grid (first 16 images, up to 4 per row)
    _save_summary_grid(OUT_DIR, n_cols=4, max_imgs=16)


def _save_summary_grid(src_dir, n_cols=4, max_imgs=16):
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png") and f[0].isdigit())[:max_imgs]
    if not files:
        return
    n = len(files)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = np.array(axes).reshape(-1)
    for ax, fname in zip(axes, files):
        img = plt.imread(os.path.join(src_dir, fname))
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    grid_path = os.path.join(src_dir, "_summary_grid.png")
    fig.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary grid saved to {grid_path}")


if __name__ == "__main__":
    main()
