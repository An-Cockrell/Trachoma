"""
Batch follicle pseudo-label generation: GradableArea UNet eyelid crop → SAM3 "dots"/"spots".

For each image in all_metadata.csv:
  1. GradableArea UNet → eyelid mask → zero non-eyelid pixels
  2. SAM3 text prompts "dots" AND "spots" → union of all instances from both prompts
  3. If any detection → save to SAMFollicleMasks/manual_check/{Images,Mask}/

All output goes to manual_check/ (no automatic_keep split — all pairs need human review).
Resume-safe: skips images whose output file already exists.

Usage:
    python generate_sam3_follicle_masks.py
"""

import os
import sys
import csv

import cv2 as cv
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
METADATA_CSV   = "/home/Trachoma/data/all_metadata.csv"
BASE_OUT       = "/home/Trachoma/annotated_data/FollicleDetection/SAMFollicleMasks/manual_check"
GA_CHECKPOINT  = "/home/Trachoma/annotated_data/GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"
SAM3_CHECKPOINT = "/home/manny/sam3_checkpoints/sam3.pt"

for sub in ("Images", "Mask"):
    os.makedirs(os.path.join(BASE_OUT, sub), exist_ok=True)

# ── GradableArea UNet ──────────────────────────────────────────────────────────
sys.path.insert(0, "/home/Trachoma/annotated_data/GradableAreaData")
from Training_lightning_GradableArea_UNet import GradableAreaUNet

# GA UNet runs on CPU so the full GPU is free for SAM3 (3.3 GB checkpoint).
# MobileNetV2 is fast enough on CPU for single-image inference.
device = torch.device("cpu")
print("[GA UNet] Loading checkpoint on CPU (GPU reserved for SAM3)...")
ga_model = GradableAreaUNet.load_from_checkpoint(
    GA_CHECKPOINT, encoder="mobilenet_v2", lr=1e-3, map_location="cpu"
)
ga_model.eval()

_ga_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_eyelid_masked(img_pil):
    """Apply GradableArea UNet; return uint8 (H,W,3) with non-eyelid pixels zeroed."""
    img_tensor = _ga_preprocess(img_pil).unsqueeze(0)
    with torch.no_grad():
        logits = ga_model(img_tensor)
    mask_512 = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    W, H = img_pil.size
    mask_orig = cv.resize(mask_512, (W, H), interpolation=cv.INTER_NEAREST)
    img_np = np.asarray(img_pil, dtype=np.uint8).copy()
    img_np[mask_orig == 0] = 0
    return img_np


# ── SAM3 ───────────────────────────────────────────────────────────────────────
# Patch torch.autocast before importing sam3: sam3_video_inference.py applies
# @torch.autocast(dtype=torch.bfloat16) at class-definition time, which raises
# on pre-Ampere GPUs if the patch isn't already in place.
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    _orig_autocast = torch.autocast

    class _Float16Autocast:
        def __new__(cls, device_type="cuda", dtype=None, **kwargs):
            if dtype == torch.bfloat16:
                dtype = torch.float16
            return _orig_autocast(device_type=device_type, dtype=dtype, **kwargs)

    torch.autocast = _Float16Autocast

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

SAM3_CONFIDENCE = 0.5
FOLLICLE_PROMPTS = ["dots", "spots"]

sam3_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SAM3] Loading checkpoint on {sam3_device}...")
_sam3_model = build_sam3_image_model(
    checkpoint_path=SAM3_CHECKPOINT,
    load_from_HF=False,
    device=sam3_device,
    enable_inst_interactivity=True,
)
sam3_processor = Sam3Processor(_sam3_model, confidence_threshold=SAM3_CONFIDENCE, device=sam3_device)


def get_union_mask(img_np, prompt):
    """Return union of all SAM3 detections above confidence threshold, or None."""
    pil_image = Image.fromarray(img_np)
    state = sam3_processor.set_image(pil_image)
    sam3_processor.reset_all_prompts(state)
    state = sam3_processor.set_text_prompt(prompt=prompt, state=state)
    masks  = state.get("masks")
    scores = state.get("scores")
    if masks is None or masks.shape[0] == 0:
        return None
    scores_np = scores.cpu().numpy()
    union = np.zeros(masks.shape[-2:], dtype=bool)
    for i, score in enumerate(scores_np):
        if score >= SAM3_CONFIDENCE:
            union |= masks[i].squeeze(0).cpu().numpy() > 0.5
    return union if union.any() else None


def get_follicle_union_mask(img_np):
    """Run SAM3 with all follicle prompts; return union (H,W) bool mask or None."""
    union     = np.zeros(img_np.shape[:2], dtype=bool)
    any_found = False
    for prompt in FOLLICLE_PROMPTS:
        pmask = get_union_mask(img_np, prompt)
        if pmask is not None:
            union    |= pmask
            any_found = True
    return union if any_found else None


# ── Processing ─────────────────────────────────────────────────────────────────
image_paths = []
with open(METADATA_CSV, newline="") as f:
    for row in csv.DictReader(f):
        if row["label"] == "1":
            image_paths.append(row["image_path"])

print(f"TF-positive images to process: {len(image_paths)}")

counts = {"saved": 0, "skipped": 0, "no_detection": 0, "missing": 0, "failed": 0}

for idx, img_path in enumerate(image_paths):
    img_name  = os.path.basename(img_path)
    out_stem  = os.path.splitext(img_name)[0]
    out_name  = out_stem + ".png"
    out_img   = os.path.join(BASE_OUT, "Images", out_name)
    out_mask  = os.path.join(BASE_OUT, "Mask",   out_name)

    if os.path.exists(out_img):
        counts["skipped"] += 1
        continue

    if not os.path.isfile(img_path):
        counts["missing"] += 1
        continue

    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  [{idx+1}] READ ERROR {img_name}: {e}")
        counts["failed"] += 1
        continue

    # Step 1: GradableArea UNet eyelid crop
    eyelid_np = get_eyelid_masked(img_pil)

    # Step 2: SAM3 follicle detection (union of "dots" + "spots")
    follicle_mask = get_follicle_union_mask(eyelid_np)

    if follicle_mask is None:
        follicle_mask = np.zeros(eyelid_np.shape[:2], dtype=bool)
        counts["no_detection"] += 1
        print(f"  [{idx+1}/{len(image_paths)}] BLANK  {img_name}")
    else:
        counts["saved"] += 1
        print(f"  [{idx+1}/{len(image_paths)}] SAVED  {img_name}")

    Image.fromarray(eyelid_np).save(out_img)
    Image.fromarray((follicle_mask.astype(np.uint8) * 255), mode="L").save(out_mask)

print("\n── Done ─────────────────────────────────────────")
print(f"  saved (SAM3 mask)  : {counts['saved']}")
print(f"  saved (blank mask) : {counts['no_detection']}")
print(f"  skipped (exists)   : {counts['skipped']}")
print(f"  missing            : {counts['missing']}")
print(f"  failed             : {counts['failed']}")
print(f"  total        : {sum(counts.values())}")
