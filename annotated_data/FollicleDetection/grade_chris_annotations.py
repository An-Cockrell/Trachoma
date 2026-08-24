"""
Count follicles in Chris's manually annotated masks and assign TF grades.
WHO standard: TF positive = 5 or more follicles.

Input:  chris_regraded_fn_fp/  (PNG masks, one per image)
Output: chris_tf_grades.csv    (image_name, follicle_count, tf_grade)
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects

MASK_DIR = os.path.join(os.path.dirname(__file__), "chris_regraded_fn_fp")
OUT_CSV = os.path.join(os.path.dirname(__file__), "chris_tf_grades.csv")
TF_THRESHOLD = 5
MIN_BLOB_SIZE = 20  # same as the rest of the pipeline


def count_follicles(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    binary = (mask > 127).astype(bool)
    binary = remove_small_objects(binary, max_size=MIN_BLOB_SIZE - 1)
    n, _ = cv2.connectedComponents(binary.astype(np.uint8))
    return n - 1  # subtract background component


records = []
for fname in sorted(os.listdir(MASK_DIR)):
    if not fname.lower().endswith(".png"):
        continue
    stem = os.path.splitext(fname)[0]
    count = count_follicles(os.path.join(MASK_DIR, fname))
    grade = int(count >= TF_THRESHOLD)
    records.append({"image_name": stem, "follicle_count": count, "tf_grade": grade})

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

print(f"Graded {len(df)} masks  →  {OUT_CSV}")
print(f"TF positive: {df['tf_grade'].sum()}  |  TF negative: {(df['tf_grade'] == 0).sum()}")
print(df.to_string(index=False))
