"""
Offline eyelid masking pre-processor for the trachoma_resnet pipeline.

Uses the trained GradableAreaUNet (MobileNetV2 encoder) to segment the everted
eyelid in each image.  Non-eyelid pixels are zeroed; masked images are saved to
an output directory.  A new CSV is written with updated image_path values so the
existing train.py / eval.py can be called without any changes.

The GA model was trained on 512x512 inputs, so each image is resized to 512x512
for masking, then saved at that size.  The resnet training pipeline resizes again
to its own img_size (e.g. 384), so no resolution is wasted.

Usage:
    cd /home/Trachoma/blackbox/trachoma_resnet
    python preprocess_eyelid_mask.py \
        --csv /home/Trachoma/data/all_metadata.csv \
        --out_dir /home/Trachoma/data/masked_images \
        --out_csv /home/Trachoma/data/all_metadata_masked.csv

Options:
    --ga_ckpt     Path to GradableAreaUNet checkpoint (default below).
    --resume      Skip images whose output file already exists.
    --batch_size  Images processed per GPU batch (default: 16).
    --num_workers DataLoader workers for image loading (default: 4).
    --fallback    When GA finds no eyelid (all-zero mask), keep the
                  original image instead of saving a blank. (default: True)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Locate GradableAreaUNet ────────────────────────────────────────────────────
_GRADABLE_AREA_DIR = Path(__file__).resolve().parents[2] / "annotated_data" / "GradableAreaData"
sys.path.insert(0, str(_GRADABLE_AREA_DIR))
from Training_lightning_GradableArea_UNet import GradableAreaUNet  # noqa: E402

# ── Constants (must match GA training) ────────────────────────────────────────
GA_CKPT_DEFAULT = str(_GRADABLE_AREA_DIR / "Checkpoints" / "GradableArea_MobileNetV2_UNet" / "last.ckpt")
GA_SIZE = 512
GA_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_to_tensor = transforms.ToTensor()
_resize512 = transforms.Resize((GA_SIZE, GA_SIZE))


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawImageDataset(Dataset):
    """Loads images as float [0,1] tensors resized to 512x512, returns path."""

    def __init__(self, df: pd.DataFrame):
        self.paths = df["image_path"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = _to_tensor(img)          # (3, H, W) float [0,1]
            tensor = _resize512(tensor)       # (3, 512, 512)
            ok = True
        except Exception:
            tensor = torch.zeros(3, GA_SIZE, GA_SIZE)
            ok = False
        return tensor, path, ok


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _out_path(out_dir: Path, row: pd.Series) -> Path:
    return out_dir / _sanitize(str(row["source"])) / str(row["image_name"])


def _save_tensor_as_jpeg(tensor: torch.Tensor, dest: Path, quality: int = 95) -> None:
    """Save a (3, H, W) float [0,1] tensor as a JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(dest), quality=quality)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--ga_ckpt", default=GA_CKPT_DEFAULT)
    p.add_argument("--resume", action="store_true", help="Skip already-saved images.")
    p.add_argument("--no_fallback", action="store_true",
                   help="Save a blank image when GA finds no eyelid (default: keep original).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    fallback_orig = not args.no_fallback
    out_dir = Path(args.out_dir).resolve()

    df = pd.read_csv(args.csv)
    for col in ("source", "image_name", "image_path"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # Pre-compute output paths
    df["_out_path"] = df.apply(lambda r: str(_out_path(out_dir, r)), axis=1)

    # If resuming, skip rows whose output already exists
    if args.resume:
        already_done = df["_out_path"].apply(lambda p: Path(p).exists())
        to_process = df[~already_done].reset_index(drop=True)
        print(f"Resuming: {already_done.sum()} already done, {len(to_process)} remaining.")
    else:
        to_process = df.reset_index(drop=True)

    if len(to_process) == 0:
        print("Nothing to process. Writing output CSV.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading GA model from {args.ga_ckpt} on {device} ...")
        ga_model = GradableAreaUNet.load_from_checkpoint(
            args.ga_ckpt, encoder="mobilenet_v2", lr=1e-3
        )
        ga_model.eval().to(device)

        dataset = RawImageDataset(to_process)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        try:
            from tqdm import tqdm
            batches = tqdm(loader, desc="Masking")
        except ImportError:
            batches = loader

        no_mask_count = 0
        error_count = 0

        # Build a path→out_path lookup for this subset
        path_to_out = dict(zip(to_process["image_path"], to_process["_out_path"]))

        with torch.no_grad():
            for images, paths, oks in batches:
                images = images.to(device)              # (B, 3, 512, 512) [0,1]
                images_norm = GA_NORM(images)           # ImageNet normalize
                logits = ga_model(images_norm)          # (B, 1, 512, 512)
                masks = (torch.sigmoid(logits) > 0.5)   # (B, 1, 512, 512) bool

                for i, (path, ok) in enumerate(zip(paths, oks)):
                    dest = Path(path_to_out[path])
                    raw = images[i].cpu()               # (3, 512, 512) [0,1]
                    mask = masks[i].cpu()               # (1, 512, 512) bool

                    if not ok.item():
                        # Image failed to load — write a blank placeholder
                        error_count += 1
                        _save_tensor_as_jpeg(torch.zeros_like(raw), dest)
                        continue

                    mask_coverage = mask.float().mean().item()
                    if mask_coverage < 0.005:
                        # GA found essentially nothing
                        no_mask_count += 1
                        out_tensor = raw if fallback_orig else torch.zeros_like(raw)
                    else:
                        out_tensor = raw * mask         # zero background

                    _save_tensor_as_jpeg(out_tensor, dest)

        print(f"\nDone. {len(to_process)} images processed.")
        print(f"  No-mask fallbacks : {no_mask_count}")
        print(f"  Load errors       : {error_count}")

    # Write output CSV — use masked path for processed rows, original path for skipped
    df_out = df.copy()
    df_out["image_path"] = df_out["_out_path"]
    df_out = df_out.drop(columns=["_out_path"])
    df_out.to_csv(args.out_csv, index=False)
    print(f"Output CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
