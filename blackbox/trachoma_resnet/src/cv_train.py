"""
Train a single fold for the 5-fold full-data cross-validation experiment.

Reads pre-computed splits from --splits_dir/fold_<i>/{train,val}.csv (produced
by src/cv_split.py). Trains a ResNet-50 from ImageNet init at 512px input with
the canonical hyperparameters (same as resnet_01's grid-search optimum:
lr=1e-4, wd=1e-4, freeze_backbone_epochs=1, max_epochs=30, patience=7),
batch size 12 with grad accumulation 2 (effective batch 24).

Each fold uses seed = base_seed + fold_idx so per-fold randomness is
deterministic but independent.

Outputs (under --out_dir/fold_<i>/):
  checkpoints/best.ckpt   -- best by val_kappa
  val_predictions.csv     -- (image_path, label, prob) on the fold's val split
  threshold.json          -- val-selected kappa-optimal & Youden thresholds
  metrics.json            -- per-fold summary (n_train, pos_weight, thresholds, ...)

Run from blackbox/trachoma_resnet/ in the project venv:

  /home/Trachoma/venv/bin/python -m src.cv_train \
      --csv /home/Trachoma/data/all_metadata.csv \
      --splits_dir runs/resnet_cv5/splits \
      --out_dir runs/resnet_cv5 \
      --fold 0

Smoke test (1 epoch, few batches):

  /home/Trachoma/venv/bin/python -m src.cv_train \
      --csv /home/Trachoma/data/all_metadata.csv \
      --splits_dir runs/resnet_cv5/splits \
      --out_dir runs/resnet_cv5_smoke \
      --fold 0 --max_epochs 1 \
      --limit_train_batches 4 --limit_val_batches 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

from .data import ImageCSVData
from .transforms import build_train_transforms, build_eval_transforms
from .model import TrachomaLitModel
from .utils import ensure_dir, save_json, seed_everything
from .tier_a_analysis import infer, sweep_threshold


def _pos_weight(labels: np.ndarray) -> float:
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    return n_neg / max(n_pos, 1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="Full metadata CSV (used only to merge labels if splits lack them).")
    p.add_argument("--splits_dir", required=True,
                   help="Directory with fold_<i>/train.csv, val.csv from src/cv_split.py.")
    p.add_argument("--out_dir", required=True,
                   help="Per-fold output dir; fold_<i>/ will be created here.")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_backbone_epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234,
                   help="Base seed; the fold actually runs at seed + fold.")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--skip_if_done", action="store_true",
                   help="Skip training if best.ckpt + val_predictions.csv both exist.")
    return p.parse_args()


def _load_fold_csv(splits_dir: Path, fold: int, name: str, meta_csv: str) -> pd.DataFrame:
    path = splits_dir / f"fold_{fold}" / f"{name}.csv"
    df = pd.read_csv(path)
    if "label" not in df.columns:
        meta = pd.read_csv(meta_csv)[["image_path", "label"]]
        df = df.merge(meta, on="image_path", how="left")
    if df["label"].isna().any():
        n_missing = int(df["label"].isna().sum())
        raise RuntimeError(f"fold_{fold}/{name}: {n_missing} rows missing label after merge")
    df["label"] = df["label"].astype(int)
    return df


def main():
    args = parse_args()
    fold_seed = args.seed + args.fold
    seed_everything(fold_seed)
    pl.seed_everything(fold_seed, workers=True)

    fold_out = ensure_dir(Path(args.out_dir) / f"fold_{args.fold}")
    ckpt_dir = ensure_dir(fold_out / "checkpoints")
    best_path = ckpt_dir / "best.ckpt"
    preds_path = fold_out / "val_predictions.csv"

    if args.skip_if_done and best_path.exists() and preds_path.exists():
        print(f"[cv_train] fold_{args.fold}: already done, skipping "
              f"(found {best_path} and {preds_path})")
        return

    splits_dir = Path(args.splits_dir)
    train_df = _load_fold_csv(splits_dir, args.fold, "train", args.csv)
    val_df = _load_fold_csv(splits_dir, args.fold, "val", args.csv)

    print(f"[cv_train] fold={args.fold} seed={fold_seed} "
          f"train={len(train_df)} val={len(val_df)} img_size={args.img_size} "
          f"BS={args.batch_size} accum={args.accumulate_grad_batches} "
          f"effective_BS={args.batch_size * args.accumulate_grad_batches}")

    train_tfms = build_train_transforms(args.img_size)
    eval_tfms = build_eval_transforms(args.img_size)
    train_ds = ImageCSVData(train_df, transform=train_tfms, return_paths=False)
    val_ds = ImageCSVData(val_df, transform=eval_tfms, return_paths=False)

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    pos_weight = _pos_weight(train_df["label"].to_numpy().astype(int))
    model = TrachomaLitModel(
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val_kappa",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val_kappa", mode="max", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        default_root_dir=str(fold_out),
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=20,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else best_path
    print(f"[cv_train] best checkpoint: {best_ckpt}")

    # Reload best, infer on val to cache OOF predictions for ensemble threshold sweep.
    model = TrachomaLitModel.load_from_checkpoint(str(best_ckpt))
    val_probs, val_y, val_paths = infer(
        model, val_df, args.img_size, args.batch_size, args.num_workers,
    )
    thr_kappa = sweep_threshold(val_probs, val_y, "kappa")
    thr_youden = sweep_threshold(val_probs, val_y, "youden")

    pd.DataFrame({
        "image_path": val_paths,
        "label": val_y,
        "prob": val_probs,
    }).to_csv(preds_path, index=False)

    save_json(fold_out / "threshold.json", {
        "kappa": float(thr_kappa),
        "youden": float(thr_youden),
    })
    save_json(fold_out / "metrics.json", {
        "fold": int(args.fold),
        "seed": int(fold_seed),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "pos_weight": float(pos_weight),
        "img_size": int(args.img_size),
        "batch_size": int(args.batch_size),
        "accumulate_grad_batches": int(args.accumulate_grad_batches),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
        "best_ckpt": str(best_ckpt),
        "val_threshold_kappa": float(thr_kappa),
        "val_threshold_youden": float(thr_youden),
    })
    print(f"[cv_train] fold_{args.fold} done. thr_k={thr_kappa:.2f} thr_y={thr_youden:.2f}")


if __name__ == "__main__":
    main()
