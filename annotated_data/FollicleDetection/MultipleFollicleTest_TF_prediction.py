import os
import sys
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from torchvision import transforms, models
from scipy import ndimage
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

plt.ioff()

from skimage.morphology import remove_small_objects

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../GradableAreaData"))
from Training_lightning_GradableArea_UNet import GradableAreaUNet

from DataLoader_lightning_MultipleFollicle_all_metadata import (
    TrachomaMetaDataModule,
    ToTensor,
)

from Training_lightning_MultipleFollicle_UNet_Pretrained import FollicleUNet
# from Training_lightning_MultipleFollicle_UNet_Scratch import FollicleUNet
# from training_multiple_follicle_sanity_check import TrachomaGradableArea

# ── Config ─────────────────────────────────────────────────────────────────────

METADATA_CSV = "/home/Trachoma/data/all_metadata.csv"
GA_PATH = "../GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"
FOLLICLE_PATH = "Checkpoints/MultipleFollicle_EfficientNetB4_UNet_Pretrained/last.ckpt"
# FOLLICLE_PATH = "Checkpoints/MultipleFollicle_EfficientNetB4_UNet_Scratch/last.ckpt"
# FOLLICLE_PATH = "Checkpoints/old_script_SAMFollicleMask/last.ckpt"
OUT_DIR = "follicle_detection_figs/EfficientNetB4_UNet_Pretrained_TF_eval"
# OUT_DIR = "follicle_detection_figs/EfficientNetB4_UNet_Scratch_TF_eval"
# OUT_DIR = "follicle_detection_figs/old_script_SAMFollicleMask_TF_eval"

# GA UNet was trained with ImageNet normalization — must match training exactly
GA_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
FOLLICLE_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # UNet Pretrained (ImageNet)
# FOLLICLE_NORM = transforms.Normalize(mean=[0.147780, 0.084905, 0.079360], std=[0.255321, 0.155361, 0.147369])  # UNet Scratch (dataset-specific)
# FOLLICLE_NORM = transforms.Normalize(mean=[0.147780, 0.084905, 0.079360], std=[0.255321, 0.155361, 0.147369])  # old_script_SAMFollicleMask (dataset-specific)

# ── Helpers ────────────────────────────────────────────────────────────────────

MIN_BLOB_SIZE = 20  # pixels — blobs smaller than this are treated as noise

GLARE_V_THRESH = 210  # HSV Value floor for glare detection
GLARE_S_THRESH = 45  # HSV Saturation ceiling (sclera sits ~80–130, safely above this)
GLARE_MIN_BLOB = 100  # minimum glare blob area in pixels
GLARE_MAX_BLOB_PCT = 3.0  # maximum glare blob area as % of image (rejects sclera)


def detect_glare_mask(images_raw):
    """
    Detects specular glare in a (1, 3, H, W) float [0,1] tensor.
    Returns a (H, W) boolean numpy array — True where glare was detected.
    """
    img_np = (images_raw.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(
        np.uint8
    )
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    raw = ((hsv[:, :, 2] >= GLARE_V_THRESH) & (hsv[:, :, 1] <= GLARE_S_THRESH)).astype(
        np.uint8
    )
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k3)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k3)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    mask = np.zeros_like(raw, dtype=bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if GLARE_MIN_BLOB <= area <= (GLARE_MAX_BLOB_PCT / 100.0) * img_area:
            mask[labels == i] = True
    return mask


def get_follicle_mask(outputs, crop_mask_np):
    """
    Threshold the follicle UNet sigmoid output at 0.5 (same as MultipleFollicleTest.py),
    restrict to the eyelid region, and remove noise blobs below MIN_BLOB_SIZE.

    Returns (outMask, num_follicles).
    """
    if crop_mask_np.size == 0:
        return np.zeros_like(outputs, dtype=bool), 0
    outMask = (outputs >= 0.5) & crop_mask_np
    outMask = remove_small_objects(outMask, min_size=MIN_BLOB_SIZE)
    _, num_follicles = ndimage.label(outMask)
    return outMask, num_follicles


def param_sweep(results, save_path=f"{OUT_DIR}/parameter_sweep.png"):
    paths           = [r[0] for r in results]
    true_labels     = np.array([r[1] for r in results])
    follicle_counts = np.array([r[2] for r in results])

    thresholds = range(0, max(follicle_counts) + 1)
    metrics_dict = {
        "threshold": [],
        "kappa":     [],
        "f1":        [],
        "accuracy":  [],
        "precision": [],
        "recall":    [],
    }

    for thresh in thresholds:
        preds = (follicle_counts >= thresh).astype(int)
        # kappa is undefined when all predictions are the same class
        try:
            kappa = cohen_kappa_score(true_labels, preds)
        except ValueError:
            kappa = 0.0
        metrics_dict["threshold"].append(thresh)
        metrics_dict["kappa"].append(kappa)
        metrics_dict["f1"].append(f1_score(true_labels, preds, zero_division=0))
        metrics_dict["accuracy"].append(accuracy_score(true_labels, preds))
        metrics_dict["precision"].append(precision_score(true_labels, preds, zero_division=0))
        metrics_dict["recall"].append(recall_score(true_labels, preds, zero_division=0))

    df_metrics = pd.DataFrame(metrics_dict)
    best_idx      = np.argmax(df_metrics["kappa"])
    best_threshold = int(df_metrics["threshold"][best_idx])
    best_row       = df_metrics.iloc[best_idx]

    print(f"\nBest threshold by Kappa: {best_threshold}")
    print(f"  Kappa:     {best_row['kappa']:.3f}")
    print(f"  F1:        {best_row['f1']:.3f}")
    print(f"  Accuracy:  {best_row['accuracy']:.3f}")
    print(f"  Precision: {best_row['precision']:.3f}")
    print(f"  Recall:    {best_row['recall']:.3f}")

    # FP / FN lists at best threshold
    preds_best = (follicle_counts >= best_threshold).astype(int)
    fp_paths = [paths[i] for i in range(len(results)) if preds_best[i] == 1 and true_labels[i] == 0]
    fn_paths = [paths[i] for i in range(len(results)) if preds_best[i] == 0 and true_labels[i] == 1]

    print(f"\nFalse Positives ({len(fp_paths)}):")
    for p in fp_paths:
        print(f"  {p}")
    print(f"\nFalse Negatives ({len(fn_paths)}):")
    for p in fn_paths:
        print(f"  {p}")

    with open(f"{OUT_DIR}/false_positives.txt", "w") as f:
        f.write("\n".join(fp_paths))
    with open(f"{OUT_DIR}/false_negatives.txt", "w") as f:
        f.write("\n".join(fn_paths))

    with open(f"{OUT_DIR}/model_summary.txt", "w") as f:
        f.write(f"Best threshold (by Kappa): {best_threshold}\n")
        f.write(f"  Kappa:     {best_row['kappa']:.4f}\n")
        f.write(f"  F1:        {best_row['f1']:.4f}\n")
        f.write(f"  Accuracy:  {best_row['accuracy']:.4f}\n")
        f.write(f"  Precision: {best_row['precision']:.4f}\n")
        f.write(f"  Recall:    {best_row['recall']:.4f}\n")
        f.write(f"\nFalse Positives: {len(fp_paths)}\n")
        f.write(f"False Negatives: {len(fn_paths)}\n")
        f.write(f"Total test images: {len(results)}\n")

    # Confusion matrix at best threshold
    cm = confusion_matrix(true_labels, preds_best)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["No TF", "TF"]).plot(ax=ax_cm)
    ax_cm.set_title(f"Threshold = {best_threshold}  (κ = {best_row['kappa']:.3f})")
    fig_cm.savefig(
        save_path.replace(".png", f"_cm_thresh{best_threshold}.png"),
        bbox_inches="tight",
    )
    plt.close(fig_cm)

    # Sweep plot
    fig, ax = plt.subplots()
    ax.plot(df_metrics["threshold"], df_metrics["kappa"],    label="Kappa", linewidth=2)
    ax.plot(df_metrics["threshold"], df_metrics["f1"],       label="F1",        linestyle="--")
    ax.plot(df_metrics["threshold"], df_metrics["accuracy"], label="Accuracy",   linestyle=":")
    ax.plot(df_metrics["threshold"], df_metrics["precision"],label="Precision",  linestyle="-.")
    ax.plot(df_metrics["threshold"], df_metrics["recall"],   label="Recall",     linestyle=(0, (3, 1, 1, 1)))
    ax.axvline(best_threshold, color="red", linestyle="--", alpha=0.6,
               label=f"Best thresh={best_threshold}")
    ax.set_xlabel("Threshold (# follicles)")
    ax.set_ylabel("Score")
    ax.set_title("TF Detection Threshold Sweep")
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return best_threshold, df_metrics


# ── Data ───────────────────────────────────────────────────────────────────────

trans_eval = transforms.Compose([ToTensor(), transforms.Resize((512, 512))])  # UNet
# trans_eval = transforms.Compose([ToTensor(), transforms.Resize(540), transforms.CenterCrop(520)])  # old_script_SAMFollicleMask

dm = TrachomaMetaDataModule(
    metadata_csv=METADATA_CSV,
    transforms_0=trans_eval,
    batch_size=1,
    num_workers=4,
)

dm.setup()
test_data = dm.test_dataloader()
print(f"Test batches: {len(test_data)}")

# ── Models ─────────────────────────────────────────────────────────────────────

ga_model = GradableAreaUNet.load_from_checkpoint(
    GA_PATH, encoder="mobilenet_v2", lr=1e-3
)
ga_model.eval()

model = FollicleUNet.load_from_checkpoint(FOLLICLE_PATH)  # UNet Pretrained or Scratch
# fcn = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
# model = TrachomaGradableArea.load_from_checkpoint(FOLLICLE_PATH, model=fcn)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ga_model = ga_model.to(device)
model = model.to(device)

# ── Inference ──────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# (path, true_label, num_follicles) — collected for all test images
results = []

with torch.no_grad():
    for batch in test_data:
        images_raw = batch["image"].to(device)  # (1, 3, 520, 520) in [0, 1]
        true_label = batch["label"].item()
        img_path = batch["path"][0]

        # GA model: ImageNet-normalized, 520×520
        images_ga = GA_NORM(images_raw)
        crop_logits = ga_model(images_ga)
        crop_mask = torch.sigmoid(crop_logits) > 0.5  # (1, 1, 520, 520)

        # Follicle UNet: feed eyelid-masked image, matching training distribution
        images_follicle = FOLLICLE_NORM(images_raw * crop_mask)

        outputs = torch.sigmoid(model(images_follicle)).squeeze().detach().cpu().numpy()  # FollicleUNet
        # outputs = torch.sigmoid(model(images_follicle)["out"]).squeeze().detach().cpu().numpy()  # TrachomaGradableArea
        crop_mask_np = crop_mask.squeeze().cpu().numpy().astype(bool)
        outputs[detect_glare_mask(images_raw)] = 0

        outMask, num_follicles = get_follicle_mask(outputs, crop_mask_np)

        results.append((img_path, true_label, num_follicles, outMask))


# ── Evaluate ───────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Evaluated {len(results)} test images")
best_thresh, sweep_df = param_sweep(results)

# ── Save FP / FN outmasks ──────────────────────────────────────────────────────

os.makedirs(f"{OUT_DIR}/fp_masks", exist_ok=True)
os.makedirs(f"{OUT_DIR}/fn_masks", exist_ok=True)

for img_path, true_label, num_follicles, outMask in results:
    pred = int(num_follicles >= best_thresh)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if pred == 1 and true_label == 0:
        Image.fromarray(outMask.astype(np.uint8) * 255).save(f"{OUT_DIR}/fp_masks/{stem}.png")
    elif pred == 0 and true_label == 1:
        Image.fromarray(outMask.astype(np.uint8) * 255).save(f"{OUT_DIR}/fn_masks/{stem}.png")

# ── Visualise 25 positive + 25 negative random samples ────────────────────────

rng = np.random.default_rng(seed=42)
pos_indices = [i for i, (_, lbl, _, _) in enumerate(results) if lbl == 1]
neg_indices = [i for i, (_, lbl, _, _) in enumerate(results) if lbl == 0]

viz_pos = rng.choice(
    pos_indices, size=min(25, len(pos_indices)), replace=False
).tolist()
viz_neg = rng.choice(
    neg_indices, size=min(25, len(neg_indices)), replace=False
).tolist()
viz_indices = set(viz_pos + viz_neg)

viz_paths = [results[i][0] for i in viz_indices]  # path is still index 0

from torch.utils.data import DataLoader as _DL
from DataLoader_lightning_MultipleFollicle_all_metadata import TrachomaMetaDataset

viz_df = dm.test_df[dm.test_df["image_path"].isin(viz_paths)].reset_index(drop=True)
viz_dataset = TrachomaMetaDataset(viz_df, transform=trans_eval)
viz_loader = _DL(viz_dataset, batch_size=1, num_workers=0, shuffle=False)

os.makedirs(f"{OUT_DIR}/positive", exist_ok=True)
os.makedirs(f"{OUT_DIR}/negative", exist_ok=True)

with torch.no_grad():
    for batch in viz_loader:
        images_raw = batch["image"].to(device)
        true_label = batch["label"].item()
        img_path = batch["path"][0]

        images_ga = GA_NORM(images_raw)
        crop_mask = torch.sigmoid(ga_model(images_ga)) > 0.5  # (1, 1, 520, 520)

        images_follicle = FOLLICLE_NORM(images_raw * crop_mask)

        outputs = torch.sigmoid(model(images_follicle)).squeeze().detach().cpu().numpy()  # FollicleUNet
        # outputs = torch.sigmoid(model(images_follicle)["out"]).squeeze().detach().cpu().numpy()  # TrachomaGradableArea
        crop_mask_np = crop_mask.squeeze().cpu().numpy().astype(bool)
        outputs[detect_glare_mask(images_raw)] = 0
        outMask, num_follicles = get_follicle_mask(outputs, crop_mask_np)
        outputs_masked = outputs * crop_mask_np

        im = images_raw.squeeze().permute(1, 2, 0).cpu().numpy()
        cropped_im = (images_raw * crop_mask).squeeze().permute(1, 2, 0).cpu().numpy()
        pred_tf = int(num_follicles >= best_thresh)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        subdir = "positive" if true_label == 1 else "negative"

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(im)
        ax[0].axis("off")
        ax[0].set_title("Input")
        ax[1].imshow(cropped_im)
        ax[1].axis("off")
        ax[1].set_title("GA Crop")
        ax[2].imshow(outputs_masked)
        ax[2].axis("off")
        ax[2].set_title("Follicle Output")
        ax[3].imshow(outMask)
        ax[3].axis("off")
        ax[3].set_title("Follicle Mask")
        fig.suptitle(
            f"{stem} | Follicles: {num_follicles} | Pred TF: {pred_tf} | True TF: {true_label}"
        )
        fig.savefig(f"{OUT_DIR}/{subdir}/{stem}.png", bbox_inches="tight", dpi=100)
        plt.close(fig)
