from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
)

from .data import build_legacy_dataframe, TrachomaDataModule, ImageCSVData
from .transforms import build_eval_transforms, build_train_transforms
from .model import TrachomaLitModel
from .utils import ensure_dir, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to .ckpt file (best.ckpt or last.ckpt).",
    )
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--image_col", type=str, default="image_path")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--group_col", type=str, default=None)
    p.add_argument(
        "--group_cols",
        nargs="+",
        default=None,
        help="Optional: multiple columns for leakage-safe grouped splitting (e.g. source id).",
    )
    p.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Path to a splits/ directory containing train.csv/val.csv/test.csv. "
        "If provided, eval will use these exact files and will NOT resplit.",
    )

    # legacy inputs
    p.add_argument("--img_dir_m", type=str, default=None)
    p.add_argument("--img_keys_csv_m", type=str, default=None)
    p.add_argument("--img_dir_o", type=str, default=None)
    p.add_argument("--img_keys_csv_o", type=str, default=None)
    p.add_argument("--img_col_o", type=str, default="imagename")
    p.add_argument("--label_col_o", type=str, default="consensus")
    p.add_argument("--clean_paths_txt", type=str, default=None)

    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="eval_outputs")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If not provided, we sweep to maximize kappa on val split.",
    )
    return p.parse_args()


def _load_df(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.image_col != "image_path":
            df = df.rename(columns={args.image_col: "image_path"})
        if args.label_col != "label":
            df = df.rename(columns={args.label_col: "label"})
        df["label"] = df["label"].astype(int)
        return df
    required = [
        args.img_dir_m,
        args.img_keys_csv_m,
        args.img_dir_o,
        args.img_keys_csv_o,
    ]
    if any(x is None for x in required):
        raise ValueError(
            "Provide either --csv OR all legacy args: --img_dir_m --img_keys_csv_m --img_dir_o --img_keys_csv_o"
        )
    return build_legacy_dataframe(
        img_dir_m=args.img_dir_m,
        img_keys_csv_m=args.img_keys_csv_m,
        img_dir_o=args.img_dir_o,
        img_keys_csv_o=args.img_keys_csv_o,
        img_col_o=args.img_col_o,
        label_col_o=args.label_col_o,
        clean_paths_txt=args.clean_paths_txt,
    )


def sweep_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_k = 0.5, -1.0
    for t in thresholds:
        k = cohen_kappa_score(y, (probs >= t).astype(int))
        if k > best_k:
            best_k, best_t = float(k), float(t)
    return best_t, best_k


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    
    df = _load_df(args)
    eval_tfms = build_eval_transforms(args.img_size)
    train_tfms = build_train_transforms(
        args.img_size
    )  # not used; datamodule expects it

    val_dl = test_dl = None

    if args.split_dir:
        split_dir = Path(args.split_dir)

        val_split = pd.read_csv(split_dir / "val.csv")
        test_split = pd.read_csv(split_dir / "test.csv")

        # Use image_path as the stable key
        val_df = df.merge(val_split[["image_path"]], on="image_path", how="inner")
        test_df = df.merge(test_split[["image_path"]], on="image_path", how="inner")

        # Sanity checks
        if len(val_df) != len(val_split):
            missing = set(val_split["image_path"]) - set(val_df["image_path"])
            raise RuntimeError(
                f"val split contains {len(missing)} paths not found in --csv (showing up to 5): {list(sorted(missing))[:5]}"
            )
        if len(test_df) != len(test_split):
            missing = set(test_split["image_path"]) - set(test_df["image_path"])
            raise RuntimeError(
                f"test split contains {len(missing)} paths not found in --csv (showing up to 5): {list(sorted(missing))[:5]}"
            )

        overlap = set(val_df["image_path"]) & set(test_df["image_path"])
        if overlap:
            raise RuntimeError(
                f"Split leakage: {len(overlap)} overlapping image_path entries between val and test."
            )

        val_ds = ImageCSVData(val_df, transform=eval_tfms, return_paths=False)
        test_ds = ImageCSVData(test_df, transform=eval_tfms, return_paths=True)

        dl_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
        )

        val_dl = torch.utils.data.DataLoader(val_ds, **dl_kwargs)
        test_dl = torch.utils.data.DataLoader(test_ds, **dl_kwargs)

    else:
        dm = TrachomaDataModule(
            df=df,
            train_tfms=train_tfms,
            eval_tfms=eval_tfms,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            oversample=False,
            group_col=args.group_col,
            group_cols=args.group_cols,
        )
        dm.setup()

        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()

    model = TrachomaLitModel.load_from_checkpoint(args.ckpt)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- get probs on val and test ---
    def infer(dl):
        probs, y, paths = [], [], []
        with torch.no_grad():
            for batch in dl:
                x = batch["image"].to(device)
                logits = model(x)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(p)
                y.append(batch["label"].cpu().numpy())
                if "image_path" in batch:
                    paths.extend(batch["image_path"])
        probs = np.concatenate(probs, axis=0)
        y = np.concatenate(y, axis=0).astype(int)
        return probs, y, paths

    val_probs, val_y, _ = infer(val_dl)
    test_probs, test_y, test_paths = infer(test_dl)

    if args.threshold is None:
        thr, val_k = sweep_threshold(val_probs, val_y)
    else:
        thr, val_k = float(args.threshold), cohen_kappa_score(
            val_y, (val_probs >= args.threshold).astype(int)
        )

    test_pred = (test_probs >= thr).astype(int)
    test_kappa = cohen_kappa_score(test_y, test_pred)

    test_acc = accuracy_score(test_y, test_pred)
    test_prec = precision_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred)

    cm = confusion_matrix(test_y, test_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Test Confusion Matrix (thr={thr:.2f})")
    plt.savefig(Path(out_dir) / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    prec, rec, pr_thr = precision_recall_curve(test_y, test_probs)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (test)")
    plt.savefig(
        Path(out_dir) / "precision_recall_curve.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    summary = {
        "ckpt": args.ckpt,
        "threshold": thr,
        "val_kappa_at_threshold": float(val_k),
        "test_kappa": float(test_kappa),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_f1": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(test_y)),
    }
    save_json(Path(out_dir) / "eval_summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import json

    main()
