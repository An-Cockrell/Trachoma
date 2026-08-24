"""
Adds confirmed zero-follicle images to the SAMFollicleMasks training set.

Reads keep.csv (human-reviewed TF-negative images), copies each to
SAMFollicleMasks/Images/ and creates a corresponding all-zero mask in
SAMFollicleMasks/Masks/.  Skips missing source files and already-added images.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

KEEP_CSV  = "MultipleFollicleImages/zero_follicles/keep.csv"
IMG_DIR   = "MultipleFollicleImages/SAMFollicleMasks/Images/"
MASK_DIR  = "MultipleFollicleImages/SAMFollicleMasks/Masks/"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

df = pd.read_csv(KEEP_CSV)
paths = df["image_path"].tolist()

added = skipped_exists = skipped_missing = 0

for src_path in paths:
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_name = f"neg_{stem}.png"
    img_out  = os.path.join(IMG_DIR, out_name)
    mask_out = os.path.join(MASK_DIR, out_name)

    if os.path.exists(img_out):
        skipped_exists += 1
        continue

    if not os.path.exists(src_path):
        print(f"  MISSING: {src_path}")
        skipped_missing += 1
        continue

    try:
        img = Image.open(src_path).convert("RGB")
        W, H = img.size
        img.save(img_out)

        mask = Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L")
        mask.save(mask_out)
        added += 1
    except Exception as e:
        print(f"  ERROR {src_path}: {e}")
        skipped_missing += 1

print(f"\nDone. Added: {added}  Already present: {skipped_exists}  Missing/error: {skipped_missing}")
print(f"SAMFollicleMasks now has {len(os.listdir(IMG_DIR))} images total.")
