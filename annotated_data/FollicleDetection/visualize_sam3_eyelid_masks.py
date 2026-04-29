"""
Visualize SAM3 eyelid masks for every image seen by MultipleFollicleImageGeneration.

For each image, saves a 3-panel figure:
  Left:   original image
  Center: SAM3 mask overlaid in red
  Right:  masked crop (non-eyelid pixels zeroed)

Output goes to:
  MultipleFollicleImages/SAMCrop/vis/<annotator>/<filename>.png

Run from annotated_data/FollicleDetection/:
    python visualize_sam3_eyelid_masks.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/Trachoma/annotated_data/GradableAreaData")
from eyelid_masker import EyelidMasker

# ── same source dirs as MultipleFollicleImageGeneration ──────────────────────
IMAGE_DIRS = [
    "/home/Trachoma/annotated_data/vgg_new_data/manny_images/",
    "/home/Trachoma/annotated_data/vgg_new_data/chris_images/",
    "/home/Trachoma/annotated_data/vgg_new_data/clara_images/",
]
ANNOTATION_FILES = [
    "/home/Trachoma/annotated_data/vgg_new_data/manny_images_annotations.json",
    "/home/Trachoma/annotated_data/vgg_new_data/chris_images_annotations.json",
    "/home/Trachoma/annotated_data/vgg_new_data/clara_images_annotations.json",
]
ANNOTATOR_NAMES = ["manny", "chris", "clara"]

VIS_ROOT = "/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/SAMCrop/vis/"

# ─────────────────────────────────────────────────────────────────────────────

masker = EyelidMasker()

for image_dir, ann_file, annotator in zip(IMAGE_DIRS, ANNOTATION_FILES, ANNOTATOR_NAMES):
    vis_dir = os.path.join(VIS_ROOT, annotator)
    os.makedirs(vis_dir, exist_ok=True)

    with open(ann_file, "r") as f:
        ann_dict = json.load(f)

    for name, annotation in ann_dict.items():
        im_filename = annotation["filename"]
        img_path = os.path.join(image_dir, im_filename)

        if not os.path.isfile(img_path):
            print(f"  [SKIP] not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)

        mask = masker.get_mask(img_np)          # (H, W) bool
        masked_np = img_np * mask[:, :, None]   # (H, W, 3)

        # ── 3-panel figure ────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(img_np)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask, :] = [0, 1, 0, 0.4]      # green = eyelid region
        overlay[~mask, :] = [1, 0, 0, 0.25]    # red   = masked out
        axes[1].imshow(overlay)
        n_det = 0 if mask.all() else 1          # crude: all-True means fallback
        axes[1].set_title(f"SAM3 mask  ({'detected' if not mask.all() else 'NO DETECTION – fallback'})")
        axes[1].axis("off")

        axes[2].imshow(masked_np)
        axes[2].set_title("Masked crop")
        axes[2].axis("off")

        plt.suptitle(f"{annotator} / {im_filename}", fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(vis_dir, os.path.splitext(im_filename)[0] + ".png")
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close(fig)

        print(f"  saved: {save_path}")

print("\nDone.")
