import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from skimage.morphology import remove_small_objects

from Training_lightning_MultipleFollicle_UNet_Pretrained import (
    FollicleUNet,
    TrachomaDataModule,
    ToTensor,
    MIN_BLOB_SIZE,
)

img_dir = "./MultipleFollicleImages/SAMFollicleMasks/Images"
mask_dir = "./MultipleFollicleImages/SAMFollicleMasks/Masks"
RUN_NAME = "MultipleFollicle_EfficientNetB4_UNet_Pretrained_v2"
save_dir = f"follicle_detection_figs/{RUN_NAME}/"
images_dir = os.path.join(save_dir, "images")
targets_dir = os.path.join(save_dir, "targets")
outputs_dir = os.path.join(save_dir, "outputs")
for d in (save_dir, images_dir, targets_dir, outputs_dir):
    os.makedirs(d, exist_ok=True)

path = "Checkpoints/MultipleFollicle_EfficientNetB4_UNet_Pretrained_v2/last.ckpt"

FOLLICLE_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
trans_eval = [
    transforms.Compose([ToTensor(), transforms.Resize((512, 512))]),
    FOLLICLE_NORM,
]

dm = TrachomaDataModule(
    img_dir,
    mask_dir,
    trans_train=trans_eval,
    trans_eval=trans_eval,
    batch_size=1,
    num_workers=1,
)
dm.setup()
test_data = dm.test_dataloader()
print("test", len(test_data))
print("training", len(dm.train_dataloader()))

model = FollicleUNet.load_from_checkpoint(path)

if torch.cuda.is_available():
    accelerator, num_devices = "gpu", torch.cuda.device_count()
else:
    accelerator, num_devices = "cpu", 1

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=num_devices,
    log_every_n_steps=2,
    max_epochs=1,
    default_root_dir="Training_Checkpoints",
)

model.eval()


def compare_follicles_by_centroid(target_mask, pred_mask):
    """
    Match predicted follicle blobs to target blobs using centroid-in-bounding-box.

    A predicted blob is a true positive if its centroid falls inside a target blob's
    bounding box. Unmatched target blobs are false negatives; unmatched predicted
    blobs are false positives.

    Returns lists of (cx, cy, radius) tuples for TP, FN, FP, plus raw blob counts.
    """
    target_u8 = remove_small_objects(
        target_mask.astype(bool), min_size=MIN_BLOB_SIZE
    ).astype(np.uint8)
    pred_u8 = remove_small_objects(
        pred_mask.astype(bool), min_size=MIN_BLOB_SIZE
    ).astype(np.uint8)

    n_target, _, stats_t, centroids_t = cv2.connectedComponentsWithStats(target_u8)
    n_pred, _, stats_p, centroids_p = cv2.connectedComponentsWithStats(pred_u8)

    target_bboxes = {i: stats_t[i][:4] for i in range(1, n_target)}
    pred_bboxes = {i: stats_p[i][:4] for i in range(1, n_pred)}

    true_positives = []
    false_negatives = []
    false_positives = []
    matched_pred = set()

    for t_idx, t_cent in enumerate(centroids_t[1:], 1):
        cx_t, cy_t = t_cent
        x, y, w, h = target_bboxes[t_idx]
        r = max(int(min(w, h) / 2), 3)
        match_found = False
        for p_idx, p_cent in enumerate(centroids_p[1:], 1):
            cx_p, cy_p = p_cent
            if x <= cx_p <= x + w and y <= cy_p <= y + h:
                true_positives.append((cx_p, cy_p, r))
                matched_pred.add(p_idx)
                match_found = True
                break
        if not match_found:
            false_negatives.append((cx_t, cy_t, r))

    for p_idx, p_cent in enumerate(centroids_p[1:], 1):
        if p_idx not in matched_pred:
            cx_p, cy_p = p_cent
            px, py, pw, ph = pred_bboxes[p_idx]
            r = max(int(min(pw, ph) / 2), 3)
            false_positives.append((cx_p, cy_p, r))

    return true_positives, false_negatives, false_positives, n_target - 1, n_pred - 1


def make_detection_overlay(shape, true_positives, false_negatives, false_positives):
    """RGB canvas with green=TP, red=FN, blue=FP filled circles."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for cx, cy, r in true_positives:
        cv2.circle(canvas, (int(cx), int(cy)), r, (0, 255, 0), -1)
    for cx, cy, r in false_negatives:
        cv2.circle(canvas, (int(cx), int(cy)), r, (255, 0, 0), -1)
    for cx, cy, r in false_positives:
        cv2.circle(canvas, (int(cx), int(cy)), r, (0, 0, 255), -1)
    return canvas


_mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

follicle_records = []

for batch in test_data:
    images, targets = batch["image"], batch["label"]
    targets_np = targets.squeeze().numpy()

    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        outputs = torch.sigmoid(model(images)).squeeze().detach().cpu().numpy()

    im = (
        (images.squeeze().cpu() * _std_t + _mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
    )
    out_mask = (outputs >= 0.5).astype(np.uint8)
    tgt_u8 = targets_np.astype(np.uint8)

    name = batch["name"][0].split(".")[0]

    Image.fromarray((im * 255).astype(np.uint8)).save(f"{images_dir}/{name}.png")
    Image.fromarray((tgt_u8 * 255).astype(np.uint8), mode="L").save(
        f"{targets_dir}/{name}.png"
    )
    Image.fromarray((out_mask * 255).astype(np.uint8), mode="L").save(
        f"{outputs_dir}/{name}.png"
    )

    tp, fn, fp, n_target, n_pred = compare_follicles_by_centroid(tgt_u8, out_mask)
    overlay = make_detection_overlay(tgt_u8.shape, tp, fn, fp)

    follicle_records.append(
        {
            "image_name": name,
            "n_target": n_target,
            "n_pred": n_pred,
            "true_pos": len(tp),
            "false_neg": len(fn),
            "false_pos": len(fp),
        }
    )

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(name, fontsize=10)
    ax[0].imshow(im)
    ax[0].axis("off")
    ax[0].set_title("Image")
    ax[1].imshow(tgt_u8)
    ax[1].axis("off")
    ax[1].set_title("Target mask")
    ax[2].imshow(outputs)
    ax[2].axis("off")
    ax[2].set_title("Model output")
    ax[3].imshow(overlay)
    ax[3].axis("off")
    ax[3].set_title("TP=green  FN=red  FP=blue")
    plt.tight_layout()
    plt.savefig(f"{save_dir}{name}.png", bbox_inches="tight")
    plt.close(fig)


# ── Summary ───────────────────────────────────────────────────────────────────

df = pd.DataFrame(follicle_records)
csv_path = f"{save_dir}follicle_counts.csv"
df.to_csv(csv_path, index=False)
print(f"Saved per-image counts → {csv_path}")

total_target = df["n_target"].sum()
total_pred = df["n_pred"].sum()
total_tp = df["true_pos"].sum()
total_fn = df["false_neg"].sum()
total_fp = df["false_pos"].sum()

lines = [
    f"--- Follicle detection summary ({len(df)} images) ---",
    f"Unweighted avg correct (TP) per image: {df['true_pos'].mean():.3f}",
    f"Unweighted avg missed  (FN) per image: {df['false_neg'].mean():.3f}",
    f"Unweighted avg extra   (FP) per image: {df['false_pos'].mean():.3f}",
]
if total_target > 0:
    lines += [
        f"Weighted correct (TP / total targets): {total_tp / total_target:.3f}  ({total_tp}/{total_target})",
        f"Weighted missed  (FN / total targets): {total_fn / total_target:.3f}  ({total_fn}/{total_target})",
    ]
if total_pred > 0:
    lines.append(
        f"Weighted extra   (FP / total preds):   {total_fp / total_pred:.3f}  ({total_fp}/{total_pred})"
    )

for line in lines:
    print(line)

summary_path = f"{save_dir}follicle_summary.txt"
with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary → {summary_path}")
