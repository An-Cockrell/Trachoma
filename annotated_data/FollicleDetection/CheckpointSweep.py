import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from scipy import ndimage
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

plt.ioff()

from skimage.morphology import remove_small_objects

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../GradableAreaData"))
from Training_lightning_GradableArea_UNet import GradableAreaUNet

from DataLoader_lightning_MultipleFollicle_all_metadata import (
    TrachomaMetaDataModule,
    ToTensor,
)
from Training_lightning_MultipleFollicle import (
    TrachomaGradableArea as SegmentationModel,
)

# ── Config ─────────────────────────────────────────────────────────────────────

METADATA_CSV = "/home/Trachoma/data/all_metadata.csv"
GA_PATH = "../GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"
CHECKPOINTS_DIR = "Checkpoints"
OUT_BASE = "follicle_detection_figs/checkpoint_sweep"

GA_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

STD_MULTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MIN_BLOB_SIZE = 20

# ── Helpers ────────────────────────────────────────────────────────────────────


def get_follicle_mask(outputs, crop_mask_np, std_mult=1.0):
    """
    Threshold follicle model outputs and remove small blobs.

    Background (outside gradable area) is zeroed first so that arbitrary FCN
    activations there don't affect the threshold or appear in the mask.
    """
    outputs = outputs * crop_mask_np  # zero background before any computation
    eyelid_pixels = outputs[crop_mask_np]
    if eyelid_pixels.size == 0:
        return np.zeros_like(outputs, dtype=bool), 0
    threshold = eyelid_pixels.mean() + std_mult * eyelid_pixels.std()
    outMask = outputs > threshold
    outMask = remove_small_objects(outMask, min_size=MIN_BLOB_SIZE)
    _, num_follicles = ndimage.label(outMask)
    return outMask, num_follicles


def param_sweep_2d(raw_results, out_dir, ckpt_label):
    """
    Sweep (std_mult × num_fol_thresh) and save a heatmap + confusion matrix.

    raw_results: list of (outputs_np, crop_mask_np, true_label)
    Returns (best_std_mult, best_num_thresh, best_metrics_series).
    """
    true_labels = np.array([r[2] for r in raw_results])

    # Compute follicle counts for every std_mult once
    counts_by_std = {}
    for std_mult in STD_MULTS:
        counts = [
            get_follicle_mask(outputs, crop_mask, std_mult)[1]
            for outputs, crop_mask, _ in raw_results
        ]
        counts_by_std[std_mult] = np.array(counts)

    # Build 2-D metric grid
    rows = []
    for std_mult in STD_MULTS:
        counts = counts_by_std[std_mult]
        for thresh in range(0, int(counts.max()) + 2):
            preds = (counts >= thresh).astype(int)
            rows.append(
                {
                    "std_mult": std_mult,
                    "num_fol_thresh": thresh,
                    "f1": f1_score(true_labels, preds, zero_division=0),
                    "accuracy": accuracy_score(true_labels, preds),
                    "precision": precision_score(true_labels, preds, zero_division=0),
                    "recall": recall_score(true_labels, preds, zero_division=0),
                }
            )

    df = pd.DataFrame(rows)
    best_idx = df["f1"].idxmax()
    best = df.loc[best_idx]
    best_std = float(best["std_mult"])
    best_thresh = int(best["num_fol_thresh"])

    print(f"  Best: std_mult={best_std}, num_fol_thresh={best_thresh}")
    print(
        f"    F1={best['f1']:.3f}  Acc={best['accuracy']:.3f}"
        f"  Prec={best['precision']:.3f}  Rec={best['recall']:.3f}"
    )

    # ── F1 heatmap ────────────────────────────────────────────────────────────
    pivot = df.pivot_table(index="std_mult", columns="num_fol_thresh", values="f1")
    pivot = pivot.loc[:, pivot.columns <= 20]  # cap columns for readability

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)), 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("Num-follicle threshold")
    ax.set_ylabel("Std multiplier")
    ax.set_title(f"F1 sweep — {ckpt_label}")

    if best_thresh in pivot.columns and best_std in pivot.index:
        col_i = list(pivot.columns).index(best_thresh)
        row_i = list(pivot.index).index(best_std)
        ax.add_patch(
            plt.Rectangle(
                (col_i - 0.5, row_i - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=2,
            )
        )

    plt.colorbar(im, ax=ax, label="F1")
    fig.savefig(os.path.join(out_dir, "f1_heatmap.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)

    # ── Confusion matrix at best params ───────────────────────────────────────
    preds_best = (counts_by_std[best_std] >= best_thresh).astype(int)
    cm = confusion_matrix(true_labels, preds_best)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["No TF", "TF"]).plot(ax=ax_cm)
    ax_cm.set_title(f"{ckpt_label}\nstd_mult={best_std}, thresh={best_thresh}")
    fig_cm.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig_cm)

    df.to_csv(os.path.join(out_dir, "sweep_metrics.csv"), index=False)
    return best_std, best_thresh, best


# ── Data ───────────────────────────────────────────────────────────────────────

trans_eval = transforms.Compose(
    [ToTensor(), transforms.Resize(540), transforms.CenterCrop(520)]
)

dm = TrachomaMetaDataModule(
    metadata_csv=METADATA_CSV,
    transforms_0=trans_eval,
    batch_size=1,
    num_workers=4,
)
dm.setup()
test_loader = dm.test_dataloader()
print(f"Test batches: {len(test_loader)}")

# ── GA model ──────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading GA model...")
ga_model = GradableAreaUNet.load_from_checkpoint(
    GA_PATH, encoder="mobilenet_v2", lr=1e-3
)
ga_model.eval().to(device)

# ── GA inference (run once, cache results for all checkpoints) ─────────────────

print("Running GA inference on test set...")
ga_cache = []  # (cropped_image_cpu, crop_mask_np, true_label, img_path)

with torch.no_grad():
    for batch in test_loader:
        images_raw = batch["image"].to(device)
        true_label = batch["label"].item()
        img_path = batch["path"][0]

        crop_mask = torch.sigmoid(ga_model(GA_NORM(images_raw))) > 0.5
        cropped_image = images_raw * crop_mask
        crop_mask_np = crop_mask.squeeze().cpu().numpy().astype(bool)

        ga_cache.append((cropped_image.cpu(), crop_mask_np, true_label, img_path))

print(f"Cached {len(ga_cache)} images.")

# ── Discover checkpoints ──────────────────────────────────────────────────────

ckpt_files = sorted(
    glob.glob(os.path.join(CHECKPOINTS_DIR, "**/last.ckpt"), recursive=True)
)
print(f"\nFound {len(ckpt_files)} checkpoints:")
for p in ckpt_files:
    print(f"  {os.path.relpath(p, CHECKPOINTS_DIR)}")

fcn_template = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)

# ── Per-checkpoint sweep ───────────────────────────────────────────────────────

summary_rows = []

for ckpt_path in ckpt_files:
    rel = os.path.relpath(os.path.dirname(ckpt_path), CHECKPOINTS_DIR)
    print(f"\n{'='*60}")
    print(f"Checkpoint: {rel}")

    try:
        fol_model = SegmentationModel.load_from_checkpoint(
            ckpt_path, model=fcn_template, strict=True
        )
    except Exception as e:
        print(f"  [SKIP] Failed to load: {e}")
        continue

    fol_model.eval().to(device)

    raw_results = []
    with torch.no_grad():
        for cropped_image, crop_mask_np, true_label, _ in ga_cache:
            try:
                outputs = (
                    fol_model(cropped_image.to(device))["out"]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
                raw_results.append((outputs, crop_mask_np, true_label))
            except Exception as e:
                print(f"  [WARN] {e}")

    out_dir = os.path.join(OUT_BASE, rel)
    os.makedirs(out_dir, exist_ok=True)

    best_std, best_thresh, best_row = param_sweep_2d(raw_results, out_dir, ckpt_label=rel)

    summary_rows.append(
        {
            "checkpoint": rel,
            "best_std_mult": best_std,
            "best_num_fol_thresh": best_thresh,
            "f1": best_row["f1"],
            "accuracy": best_row["accuracy"],
            "precision": best_row["precision"],
            "recall": best_row["recall"],
            "n_test": len(raw_results),
        }
    )

    del fol_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────

summary_df = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
os.makedirs(OUT_BASE, exist_ok=True)
summary_path = os.path.join(OUT_BASE, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"\n{'='*60}")
print("SUMMARY (sorted by F1):")
print(summary_df.to_string(index=False))
print(f"\nSaved summary to {summary_path}")
