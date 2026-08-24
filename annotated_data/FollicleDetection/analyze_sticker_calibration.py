"""
Derives MIN_FOLLICLE_RATIO from the sticker_pixel7 reference images.

Each image contains an orange sticker on the grader's thumb with 5 white dots
that physically represent the 0.5 mm minimum follicle diameter (WHO standard).

The script:
  1. Detects the 5 reference dots for each image and records their pixel area.
  2. Runs the GradableArea UNet to segment the eyelid and records its pixel area.
  3. Computes ratio = dot_area_512 / eyelid_area_512 across quality images.
  4. Writes the median ratio to sticker_calibration.json.

Outputs
-------
  sticker_calibration.json      — calibration constant for production use
  sticker_calibration_vis/      — debug images with detected dots + eyelid overlay
"""

import os, sys, json
import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../GradableAreaData"))
from Training_lightning_GradableArea_UNet import GradableAreaUNet

GA_PATH    = "../GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"
GA_NORM    = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
STICKER_DIR = "sticker_pixel7"
OUT_DIR    = "sticker_calibration_vis"
OUT_JSON   = "sticker_calibration.json"

FULL_W, FULL_H = 4080, 3072
MODEL_SIZE = 512
AREA_SCALE = (MODEL_SIZE / FULL_W) * (MODEL_SIZE / FULL_H)  # full-res → 512×512

# ── Sticker / dot detection ────────────────────────────────────────────────────

def detect_sticker_dots(img_bgr):
    """
    Returns (sticker_area_fullres, blobs) where blobs = [(area_px, circularity, cx, cy), ...].
    The blobs are the 5 reference dots inside the orange sticker, sorted largest-first.
    Returns (None, []) if the sticker or dots cannot be found.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Orange/red hue range
    mask1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([20,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
    orange = (mask1 | mask2).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(orange)
    if n <= 1:
        return None, []
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    sticker_area = int(stats[best, cv2.CC_STAT_AREA])

    # Fill the sticker outline to find holes (= reference dots)
    sticker_bin = (labels == best).astype(np.uint8)
    cnts, _ = cv2.findContours(sticker_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(sticker_bin)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    holes = ((filled > 0) & (orange == 0)).astype(np.uint8)

    nd, ld, sd, cents = cv2.connectedComponentsWithStats(holes)
    blobs = []
    for i in range(1, nd):
        area = int(sd[i, cv2.CC_STAT_AREA])
        # Size gate: 0.1 % – 3 % of sticker area
        if not (sticker_area * 0.001 < area < sticker_area * 0.03):
            continue
        b_mask = (ld == i).astype(np.uint8)
        c2, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not c2:
            continue
        peri = cv2.arcLength(c2[0], True)
        circ = (4 * np.pi * area / (peri ** 2 + 1e-9)) if peri > 0 else 0
        if circ < 0.5:
            continue
        blobs.append((area, circ, float(cents[i][0]), float(cents[i][1])))
    blobs.sort(key=lambda b: -b[0])
    return sticker_area, blobs[:5]


# ── GA model ──────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ga_model = GradableAreaUNet.load_from_checkpoint(GA_PATH, encoder="mobilenet_v2", lr=1e-3)
ga_model.eval().to(device)

_resize512 = transforms.Resize((MODEL_SIZE, MODEL_SIZE))


def get_eyelid_area_512(img_bgr):
    """
    Runs the GA UNet on img_bgr and returns (eyelid_area_px, mask_512).
    The mask is a boolean array at 512×512.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    tensor = _resize512(tensor)
    tensor_norm = GA_NORM(tensor)
    with torch.no_grad():
        logits = ga_model(tensor_norm)
        mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(bool)
    return int(mask.sum()), mask


# ── Main loop ─────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

fnames = sorted(f for f in os.listdir(STICKER_DIR) if f.endswith(".jpg"))

ratios = []
skipped = []

for fname in fnames:
    img = cv2.imread(os.path.join(STICKER_DIR, fname))
    if img is None:
        skipped.append((fname, "read error"))
        continue

    sticker_area, blobs = detect_sticker_dots(img)

    if sticker_area is None or len(blobs) != 5:
        skipped.append((fname, f"dots={len(blobs) if blobs is not None else 0}"))
        continue

    sticker_diam = 2 * np.sqrt(sticker_area / np.pi)
    if sticker_diam < 450:
        skipped.append((fname, f"sticker too small ({sticker_diam:.0f}px)"))
        continue

    dot_areas_fullres = [b[0] for b in blobs]
    dot_median_512 = float(np.median(dot_areas_fullres)) * AREA_SCALE

    eyelid_area_512, eyelid_mask = get_eyelid_area_512(img)
    if eyelid_area_512 < 1000:
        skipped.append((fname, "GA model found no eyelid"))
        continue

    ratio = dot_median_512 / eyelid_area_512
    ratios.append(ratio)

    # ── Debug visualisation ──────────────────────────────────────────────────
    vis = cv2.resize(img, (800, 600))
    sx, sy = 800 / img.shape[1], 600 / img.shape[0]

    for area, circ, cx, cy in blobs:
        r = max(2, int(np.sqrt(area / np.pi) * sx))
        cv2.circle(vis, (int(cx * sx), int(cy * sy)), r, (0, 255, 0), 2)

    mask_up = cv2.resize(eyelid_mask.astype(np.uint8) * 80, (800, 600))
    vis[:, :, 1] = np.clip(vis[:, :, 1].astype(int) + mask_up, 0, 255).astype(np.uint8)

    label = f"ratio={ratio:.5f}  eyelid={eyelid_area_512}px"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, fname), vis)


# ── Summarise ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Quality images used : {len(ratios)} / {len(fnames)}")
print(f"Skipped             : {len(skipped)}")
for fname, reason in skipped:
    print(f"  {fname}: {reason}")

if not ratios:
    print("ERROR: no usable images — cannot compute calibration constant.")
    sys.exit(1)

ratios_arr = np.array(ratios)
q1, q3 = np.percentile(ratios_arr, [25, 75])
iqr = q3 - q1
robust = ratios_arr[(ratios_arr >= q1 - 1.5 * iqr) & (ratios_arr <= q3 + 1.5 * iqr)]

print(f"\nRatio (dot_area_512 / eyelid_area_512):")
print(f"  all    : mean={ratios_arr.mean():.6f}  median={np.median(ratios_arr):.6f}  std={ratios_arr.std():.6f}")
print(f"  IQR-filt({len(robust)}): mean={robust.mean():.6f}  median={np.median(robust):.6f}  std={robust.std():.6f}")

calibration_ratio = float(np.median(robust))
print(f"\nCalibration constant (IQR-filtered median): {calibration_ratio:.6f}")

result = {
    "min_follicle_ratio": calibration_ratio,
    "description": "min_follicle_area_px = min_follicle_ratio * eyelid_area_px  (both at 512x512)",
    "dot_physical_diameter_mm": 0.5,
    "n_images_used": int(len(robust)),
    "n_images_total": int(len(fnames)),
    "ratio_mean": float(robust.mean()),
    "ratio_std": float(robust.std()),
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUT_JSON}")
