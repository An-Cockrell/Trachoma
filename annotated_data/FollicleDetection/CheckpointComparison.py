import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from skimage import io
from scipy import ndimage

plt.ioff()

from skimage.morphology import remove_small_objects

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../GradableAreaData"))
from Training_lightning_GradableArea_UNet import GradableAreaUNet
from DataLoader_lightning_MultipleFollicle_no_masks import ToTensor
from Training_lightning_MultipleFollicle import TrachomaGradableArea as SegmentationModel

MIN_BLOB_SIZE = 20  # pixels — blobs smaller than this are treated as noise


def get_follicle_mask(outputs, crop_mask_np):
    eyelid_pixels = outputs[crop_mask_np]
    if eyelid_pixels.size == 0:
        return np.zeros_like(outputs, dtype=bool), 0
    threshold = eyelid_pixels.mean() + eyelid_pixels.std()
    outMask = (outputs > threshold) & crop_mask_np
    outMask = remove_small_objects(outMask, min_size=MIN_BLOB_SIZE)
    _, num_follicles = ndimage.label(outMask)
    return outMask, num_follicles


# ── Config ─────────────────────────────────────────────────────────────────────

CHECKPOINTS_DIR = "Checkpoints"
GA_PATH = "../GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"
METADATA_CSV = "/home/Trachoma/data/all_metadata.csv"
REF_DIR = "follicle_detection_figs/new_ga_run_0"
OUT_BASE = "follicle_detection_figs/checkpoint_comparison"

GA_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

trans_eval = transforms.Compose(
    [ToTensor(), transforms.Resize(540), transforms.CenterCrop(520)]
)

# Directories to sweep — for those with run_* subdirs, each run is tested separately
CHECKPOINT_ROOTS = [
    "MultipleFollicle_all_available_data",
    "MultipleFollicle_all_data_cropped_images",
    "MultipleFollicle_CCM_data_all_follicles",
    "MultipleFollicle_CCM_data_def_follicles",
    "Old_script_new_data",
    "Old_script_new_data_softmax",
    "Sanity_Check",
    "Sanity_Check_sigmoid",
]

# ── Collect checkpoints ────────────────────────────────────────────────────────
# Returns list of (ckpt_path, output_subdir_relative_to_OUT_BASE)

def collect_checkpoints():
    entries = []
    for root in CHECKPOINT_ROOTS:
        root_path = os.path.join(CHECKPOINTS_DIR, root)
        # Gather all last.ckpt files under this root
        ckpts = sorted(glob.glob(os.path.join(root_path, "**/last.ckpt"), recursive=True))
        for ckpt in ckpts:
            # Output subdir mirrors the checkpoint path relative to CHECKPOINTS_DIR,
            # dropping the trailing /last.ckpt
            rel = os.path.relpath(os.path.dirname(ckpt), CHECKPOINTS_DIR)
            entries.append((ckpt, rel))
    return entries


# ── Recover original image paths from the reference run viz filenames ──────────

def load_sample_paths():
    meta = pd.read_csv(METADATA_CSV)
    # stem = image_name without extension
    meta["stem"] = meta["image_name"].str.rsplit(".", n=1).str[0]
    stem_to_path = dict(zip(meta["stem"], meta["image_path"]))

    samples = {}  # path -> "positive" | "negative"
    for label in ("positive", "negative"):
        for png in glob.glob(os.path.join(REF_DIR, label, "*.png")):
            stem = os.path.splitext(os.path.basename(png))[0]
            if stem in stem_to_path:
                samples[stem_to_path[stem]] = label
            else:
                print(f"[WARN] stem '{stem}' not found in metadata — skipping")

    print(f"Loaded {sum(v == 'positive' for v in samples.values())} positive and "
          f"{sum(v == 'negative' for v in samples.values())} negative reference images")
    return samples


# ── Inference for one image ────────────────────────────────────────────────────

def run_image(img_path, ga_model, fol_model, device):
    image = io.imread(img_path)
    tensor = trans_eval(image).unsqueeze(0).to(device)  # (1, 3, 520, 520)

    images_ga  = GA_NORM(tensor)
    crop_mask    = torch.sigmoid(ga_model(images_ga)) > 0.5
    crop_mask_np = crop_mask.squeeze().cpu().numpy().astype(bool)

    cropped      = tensor * crop_mask
    outputs      = fol_model(cropped)["out"].squeeze().detach().cpu().numpy()
    outputs_masked = outputs * crop_mask_np
    out_mask, n_fol = get_follicle_mask(outputs, crop_mask_np)

    im         = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    cropped_im = cropped.squeeze().permute(1, 2, 0).cpu().numpy()
    return im, cropped_im, outputs_masked, out_mask, n_fol


def save_viz(im, cropped_im, outputs_masked, out_mask, n_fol, stem, out_path):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(im);             ax[0].axis("off"); ax[0].set_title("Input")
    ax[1].imshow(cropped_im);     ax[1].axis("off"); ax[1].set_title("GA Crop")
    ax[2].imshow(outputs_masked); ax[2].axis("off"); ax[2].set_title("Follicle Output")
    ax[3].imshow(out_mask);       ax[3].axis("off"); ax[3].set_title("Follicle Mask")
    fig.suptitle(f"{stem} | Detected blobs: {n_fol}")
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading GA model...")
ga_model = GradableAreaUNet.load_from_checkpoint(GA_PATH, encoder="mobilenet_v2", lr=1e-3)
ga_model.eval().to(device)

samples = load_sample_paths()
checkpoints = collect_checkpoints()
print(f"\nCheckpoints to evaluate: {len(checkpoints)}")
for ckpt, subdir in checkpoints:
    print(f"  {subdir}")

fcn_template = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)

for ckpt_path, out_subdir in checkpoints:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {out_subdir}")

    try:
        fol_model = SegmentationModel.load_from_checkpoint(
            ckpt_path, model=fcn_template, strict=True
        )
    except Exception as e:
        print(f"  [SKIP] Failed to load: {e}")
        continue

    fol_model.eval().to(device)

    pos_dir = os.path.join(OUT_BASE, out_subdir, "positive")
    neg_dir = os.path.join(OUT_BASE, out_subdir, "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    with torch.no_grad():
        for img_path, label in samples.items():
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = pos_dir if label == "positive" else neg_dir
            out_path = os.path.join(out_dir, f"{stem}.png")

            try:
                im, cropped_im, outputs_masked, out_mask, n_fol = run_image(
                    img_path, ga_model, fol_model, device
                )
                save_viz(im, cropped_im, outputs_masked, out_mask, n_fol, stem, out_path)
            except Exception as e:
                print(f"  [WARN] {stem}: {e}")

    print(f"  Saved to {os.path.join(OUT_BASE, out_subdir)}")
    del fol_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nDone.")
