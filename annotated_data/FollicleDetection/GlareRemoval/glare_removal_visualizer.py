import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGES_DIR     = "../MultipleFollicleImages/SAMFollicleMasks/Images"
N_IMAGES       = 20
SEED           = 42
V_THRESH       = 210
S_THRESH       = 45
MIN_BLOB       = 100
MAX_BLOB_PCT   = 3.0
INPAINT_RADIUS = 7


def detect_and_inpaint(img_bgr):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    raw  = ((hsv[:, :, 2] >= V_THRESH) & (hsv[:, :, 1] <= S_THRESH)).astype(np.uint8)
    k3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw  = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k3)
    raw  = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  k3)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    mask = np.zeros_like(raw)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if MIN_BLOB <= area <= (MAX_BLOB_PCT / 100.0) * img_area:
            mask[labels == i] = 255
    k5       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated  = cv2.dilate(mask, k5, iterations=1)
    inpainted = cv2.inpaint(img_bgr, dilated, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
    return mask, inpainted


def overlay_mask(img_bgr, mask):
    red = img_bgr.copy()
    red[mask > 0] = [0, 0, 220]
    return cv2.addWeighted(img_bgr, 0.45, red, 0.55, 0)


def to_rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


rng      = random.Random(SEED)
files    = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
selected = rng.sample(files, min(N_IMAGES, len(files)))

for fname in selected:
    img             = cv2.imread(os.path.join(IMAGES_DIR, fname))
    mask, inpainted = detect_and_inpaint(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(fname, fontsize=9)
    axes[0].imshow(to_rgb(img));                     axes[0].axis("off"); axes[0].set_title("Original")
    axes[1].imshow(to_rgb(overlay_mask(img, mask))); axes[1].axis("off"); axes[1].set_title("Glare region (red)")
    axes[2].imshow(to_rgb(inpainted));               axes[2].axis("off"); axes[2].set_title("Inpainted")
    plt.tight_layout()
    plt.show()
