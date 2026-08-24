"""Grad-CAM figure for the paper.

Pulls representative TP / TN / FP / FN exemplars from the held-out test set
(using ensemble probabilities + the kappa-optimal threshold), generates a
Grad-CAM overlay using the median-AUROC fold (fold 3, AUROC = 0.9705) of the
ResNet-50, and lays the results out as a 4-row x 4-col grid figure.

Usage:
    pip install grad-cam
    python generate_gradcam_fig.py
        [--fold 3] [--objective kappa|youden]
        [--n_per_cell 4] [--out FIGURES_DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model import TrachomaLitModel
from src.utils import IMAGENET_MEAN, IMAGENET_STD


IMG_SIZE = 512
RESIZE_SIZE = int(IMG_SIZE * 1.15)


# ---------------------------------------------------------------------------
# Model + transforms
# ---------------------------------------------------------------------------

class _SqueezeToTwoDim(torch.nn.Module):
    """Wrap the Lightning module so the output is (B, 1) — the shape the
    pytorch_grad_cam target API expects for binary classifiers."""

    def __init__(self, lit: TrachomaLitModel):
        super().__init__()
        self.net = lit.net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out  # (B, 1)


class _PositiveLogitTarget:
    """Target callable: return the (single) positive-class logit."""
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[0]


def load_model(ckpt_path: Path, device: torch.device) -> _SqueezeToTwoDim:
    lit = TrachomaLitModel.load_from_checkpoint(
        str(ckpt_path), map_location=device, strict=False
    )
    lit.eval()
    wrapped = _SqueezeToTwoDim(lit).to(device).eval()
    return wrapped


def prepare_input_and_underlay(
    image_pil: Image.Image,
) -> tuple[torch.Tensor, np.ndarray]:
    """Replicate build_eval_transforms, but split into (model_input, underlay).

    underlay: HxWx3 float32 in [0,1], the exact crop the model saw.
    """
    common = T.Compose([
        T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
        T.Resize(RESIZE_SIZE),
        T.CenterCrop(IMG_SIZE),
    ])
    cropped = common(image_pil)

    x = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(cropped).unsqueeze(0)

    underlay = np.asarray(cropped, dtype=np.float32) / 255.0  # (H, W, 3)
    return x, underlay


# ---------------------------------------------------------------------------
# Exemplar selection
# ---------------------------------------------------------------------------

def source_from_path(p: str) -> str:
    return Path(p).parent.name


def select_exemplars(
    test_index: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    n_per_cell: int,
) -> dict[str, list[int]]:
    """For each of TP/TN/FP/FN, pick n_per_cell exemplars favoring source diversity."""
    labels = test_index["label"].to_numpy().astype(int)
    preds = (probs >= threshold).astype(int)
    sources = test_index["image_path"].map(source_from_path).to_numpy()

    cells_rank: dict[str, np.ndarray] = {
        # TP: highest prob among true positives (most confident correct calls)
        "TP": _rank(labels, preds, probs, gt=1, pr=1, descending=True),
        # TN: lowest prob among true negatives
        "TN": _rank(labels, preds, probs, gt=0, pr=0, descending=False),
        # FP: highest prob among false positives (most confident errors)
        "FP": _rank(labels, preds, probs, gt=0, pr=1, descending=True),
        # FN: lowest prob among false negatives (most badly missed positives)
        "FN": _rank(labels, preds, probs, gt=1, pr=0, descending=False),
    }

    chosen: dict[str, list[int]] = {}
    for cell, idxs in cells_rank.items():
        picked: list[int] = []
        seen: set[str] = set()
        for i in idxs:
            s = sources[i]
            if s not in seen:
                picked.append(int(i))
                seen.add(s)
            if len(picked) == n_per_cell:
                break
        # fill remainder from the top of the list
        if len(picked) < n_per_cell:
            for i in idxs:
                if int(i) not in picked:
                    picked.append(int(i))
                if len(picked) == n_per_cell:
                    break
        chosen[cell] = picked
    return chosen


def _rank(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray,
    gt: int, pr: int, descending: bool,
) -> np.ndarray:
    mask = (labels == gt) & (preds == pr)
    idx = np.where(mask)[0]
    if descending:
        order = np.argsort(-probs[idx])
    else:
        order = np.argsort(probs[idx])
    return idx[order]


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

CELL_LABEL = {
    "TP": "True\npositive",
    "TN": "True\nnegative",
    "FP": "False\npositive",
    "FN": "False\nnegative",
}

# Pretty-print source folder names → labels used in the rest of the paper.
SOURCE_PRETTY = {
    "SOCIT_R": "SOCIT",
    "CC_EA2017": "CC_EA2017",
    "allTZphotos": "ICAPS",
    "m": "Kim et al.",
    "Solomons2": "Trop. Data (Solomons)",
    "Australia": "Trop. Data (Australia)",
    "Gambia PRET 18m": "Trop. Data (Gambia)",
    "TANA II study,  Ethiopia, Goncha Siso Enesie woreda, "
    "Amhara Region, Nov 2011": "Trop. Data (TANA II)",
}


def render_grid(
    chosen: dict[str, list[int]],
    test_index: pd.DataFrame,
    probs: np.ndarray,
    cam: GradCAM,
    device: torch.device,
    target: _PositiveLogitTarget,
    out_png: Path,
    dpi: int,
    exemplars_dir: Path | None = None,
) -> None:
    rows = ["TP", "TN", "FP", "FN"]
    ncols = max(len(v) for v in chosen.values())

    fig, axes = plt.subplots(
        len(rows), ncols,
        figsize=(2.6 * ncols + 1.0, 2.85 * len(rows)),
        gridspec_kw={"wspace": 0.05, "hspace": 0.30},
    )

    manifest_rows: list[dict] = []
    if exemplars_dir is not None:
        exemplars_dir.mkdir(parents=True, exist_ok=True)

    for r, cell in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c >= len(chosen[cell]):
                ax.axis("off")
                continue

            idx = chosen[cell][c]
            row = test_index.iloc[idx]
            src_folder = source_from_path(row["image_path"])
            img = Image.open(row["image_path"])
            x, underlay = prepare_input_and_underlay(img)
            x = x.to(device)

            grayscale_cam = cam(input_tensor=x, targets=[target])[0]
            overlay = show_cam_on_image(underlay, grayscale_cam, use_rgb=True)
            ax.imshow(overlay)

            src = SOURCE_PRETTY.get(src_folder, src_folder)
            ax.set_title(
                f"{src}\np = {probs[idx]:.2f}",
                fontsize=8.5, pad=4,
            )

            if exemplars_dir is not None:
                stem = f"{cell}_{c}_{src_folder}_{Path(row['image_path']).stem}"
                orig_dst = exemplars_dir / f"{stem}__original{Path(row['image_path']).suffix.lower()}"
                crop_dst = exemplars_dir / f"{stem}__crop512.png"
                cam_dst = exemplars_dir / f"{stem}__gradcam.png"
                shutil.copy2(row["image_path"], orig_dst)
                Image.fromarray((underlay * 255).astype(np.uint8)).save(crop_dst)
                Image.fromarray(overlay).save(cam_dst)
                manifest_rows.append({
                    "cell": cell,
                    "rank_in_cell": c,
                    "test_index_row": int(idx),
                    "source_folder": src_folder,
                    "source_pretty": src,
                    "image_path": row["image_path"],
                    "label": int(row["label"]),
                    "prob_ensemble": float(probs[idx]),
                    "original_file": orig_dst.name,
                    "crop_file": crop_dst.name,
                    "gradcam_file": cam_dst.name,
                })

        # Row label on the far-left axis
        axes[r, 0].set_ylabel(
            CELL_LABEL[cell],
            fontsize=11, fontweight="bold", rotation=0,
            labelpad=42, ha="right", va="center",
        )

    fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    if exemplars_dir is not None and manifest_rows:
        manifest_path = exemplars_dir / "manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"Wrote {manifest_path} ({len(manifest_rows)} exemplars)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_root", default="eval_outputs/resnet_cv5")
    ap.add_argument("--runs_root", default="runs/resnet_cv5")
    ap.add_argument("--objective", choices=["kappa", "youden"], default="kappa")
    ap.add_argument("--fold", type=int, default=3,
                    help="Which fold to Grad-CAM (default 3 = median AUROC).")
    ap.add_argument("--n_per_cell", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    tier = "tier_a" if args.objective == "kappa" else "tier_a_youden"
    tier_dir = eval_root / tier

    summary = json.loads((tier_dir / "cv_summary.json").read_text())
    threshold = float(summary["ensemble_threshold_oof"])

    test_index = pd.read_csv(tier_dir / "test_index.csv")
    probs = np.load(tier_dir / "test_probs_ensemble.npy")
    assert len(probs) == len(test_index), \
        f"probs ({len(probs)}) / test_index ({len(test_index)}) length mismatch"

    chosen = select_exemplars(test_index, probs, threshold, args.n_per_cell)
    for cell, idxs in chosen.items():
        srcs = [source_from_path(test_index.iloc[i]["image_path"]) for i in idxs]
        print(f"{cell}: {len(idxs)} picks across sources {srcs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.runs_root) / f"fold_{args.fold}" / "checkpoints" / "best.ckpt"
    model = load_model(ckpt, device)

    target_layers = [model.net.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    target = _PositiveLogitTarget()

    out_png = Path(args.out) if args.out else (
        eval_root / "figures_paper" / f"fig_gradcam_fold{args.fold}.png"
    )
    exemplars_dir = out_png.parent / f"gradcam_exemplars_fold{args.fold}"
    render_grid(
        chosen, test_index, probs, cam, device, target, out_png, args.dpi,
        exemplars_dir=exemplars_dir,
    )


if __name__ == "__main__":
    main()
