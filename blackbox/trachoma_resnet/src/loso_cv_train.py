"""
Tier C — Leave-One-Source-Out cross-region generalization with 5-fold CV.

For each held-out source S:
  * the ENTIRE source S is held out as the test set (held_out_test.csv),
  * the remaining 7 sources form the training pool,
  * a leakage-safe StratifiedKFold(5) on (source, id) groups carves five
    disjoint train/val partitions out of that pool,
  * a fresh ResNet-50 (512px, canonical HPs) is trained on each fold,
  * for each fold we save the best.ckpt + val_predictions.csv + threshold.json.

40 trainings total (8 sources x 5 folds). Resumable: any fold whose
best.ckpt + val_predictions.csv already exist is skipped, so a crash mid-run
doesn't lose completed folds.

This mirrors Yazbeck multiregion 2026's cross-region protocol (their
single-region models are evaluated against every other region with subject-level
splitting; we extend that to 8 sources with 5-fold internal CV).

Run in the project venv:

  /home/Trachoma/venv/bin/python -m src.loso_cv_train \
      --csv /home/Trachoma/data/all_metadata.csv \
      --out_dir runs/loso_cv5 \
      --max_epochs 30

Smoke test (1 source, 1 fold, 1 epoch, few batches):

  /home/Trachoma/venv/bin/python -m src.loso_cv_train \
      --csv /home/Trachoma/data/all_metadata.csv \
      --out_dir runs/loso_cv5_smoke \
      --sources ICAPS --folds 0 --max_epochs 1 \
      --limit_train_batches 4 --limit_val_batches 4

Outputs (per held-out source S and fold i) under <out_dir>/<S>/fold_<i>/:
  checkpoints/best.ckpt, val_predictions.csv, threshold.json, metrics.json
Per source: <out_dir>/<S>/held_out_test.csv, <out_dir>/<S>/splits/fold_<i>/{train,val}.csv
Top-level: <out_dir>/training_progress.csv (incremental status of all 40 folds)
"""

from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .data import ImageCSVData
from .transforms import build_train_transforms, build_eval_transforms
from .model import TrachomaLitModel
from .utils import ensure_dir, save_json, seed_everything
from .tier_a_analysis import infer, sweep_threshold


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def _pos_weight(labels: np.ndarray) -> float:
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    return n_neg / max(n_pos, 1.0)


def _collapse_to_groups(df: pd.DataFrame, group_cols: Sequence[str],
                        label_col: str = "label") -> pd.DataFrame:
    return df.groupby(list(group_cols), as_index=False)[label_col].max()


def _kfold_pool(pool_df: pd.DataFrame, group_cols: Sequence[str], n_folds: int,
                label_col: str, seed: int
                ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    gc = list(group_cols)
    g = _collapse_to_groups(pool_df, gc, label_col).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(g, g[label_col])):
        g_tr = g.iloc[tr_idx]
        g_va = g.iloc[va_idx]
        train_df = pool_df.merge(g_tr[gc], on=gc, how="inner").reset_index(drop=True)
        val_df = pool_df.merge(g_va[gc], on=gc, how="inner").reset_index(drop=True)
        tg = set(map(tuple, g_tr[gc].to_numpy()))
        vg = set(map(tuple, g_va[gc].to_numpy()))
        assert tg.isdisjoint(vg), f"Group leakage in pool fold {fi}"
        out.append((train_df, val_df))
    return out


def _train_one(train_df: pd.DataFrame, val_df: pd.DataFrame, fold_out: Path,
               args, fold_seed: int) -> dict:
    """Trains one (source, fold) combo. Returns summary dict."""
    seed_everything(fold_seed)
    pl.seed_everything(fold_seed, workers=True)

    ckpt_dir = ensure_dir(fold_out / "checkpoints")
    best_path = ckpt_dir / "best.ckpt"
    preds_path = fold_out / "val_predictions.csv"

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
    # Reload best and cache OOF val predictions for downstream ensemble threshold sweep
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

    summary = {
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
    }
    save_json(fold_out / "threshold.json", {
        "kappa": float(thr_kappa),
        "youden": float(thr_youden),
    })
    save_json(fold_out / "metrics.json", summary)

    # Free GPU mem before next fold
    del model, trainer, train_ds, val_ds, train_dl, val_dl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Metadata CSV (id, label, source, image_path).")
    p.add_argument("--out_dir", default="runs/loso_cv5")
    p.add_argument("--sources", nargs="+", default=None,
                   help="Restrict to these held-out sources (default: all).")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="Restrict to these fold indices (default: all 5).")
    p.add_argument("--n_folds", type=int, default=5)
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
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--skip_if_done", action="store_true",
                   help="Skip a fold if its best.ckpt + val_predictions.csv both exist.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv)
    for col in ("id", "label", "source", "image_path"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df["label"] = df["label"].astype(int)

    all_sources = sorted(df["source"].unique())
    sources = args.sources if args.sources else all_sources
    unknown = [s for s in sources if s not in all_sources]
    if unknown:
        raise ValueError(f"Unknown source(s): {unknown}. Available: {all_sources}")

    folds = args.folds if args.folds is not None else list(range(args.n_folds))

    print(f"[loso_cv] {len(sources)} held-out source(s) x {len(folds)} fold(s)")
    print(f"[loso_cv] sources: {sources}")
    print(f"[loso_cv] folds:   {folds}")

    progress_rows = []

    for s in sources:
        tag = _sanitize(s)
        src_dir = ensure_dir(out_dir / tag)
        print("\n" + "=" * 70)
        print(f"[loso_cv] HELD-OUT SOURCE: {s}")
        print("=" * 70)

        held_out = df[df["source"] == s].reset_index(drop=True)
        pool = df[df["source"] != s].reset_index(drop=True)
        held_out.to_csv(src_dir / "held_out_test.csv", index=False)
        print(f"[loso_cv] pool n={len(pool)}  held-out test n={len(held_out)}  "
              f"test prev={held_out['label'].mean():.4f}")

        try:
            pool_folds = _kfold_pool(
                pool, group_cols=("source", "id"), n_folds=args.n_folds,
                label_col="label", seed=args.seed,
            )
        except Exception as exc:
            print(f"[loso_cv] ERROR splitting pool for {s}: {exc}")
            traceback.print_exc()
            continue

        splits_dir = ensure_dir(src_dir / "splits")
        for fi, (tr_df, va_df) in enumerate(pool_folds):
            fdir = ensure_dir(splits_dir / f"fold_{fi}")
            tr_df.to_csv(fdir / "train.csv", index=False)
            va_df.to_csv(fdir / "val.csv", index=False)

        for fi in folds:
            tr_df, va_df = pool_folds[fi]
            fold_out = ensure_dir(src_dir / f"fold_{fi}")
            best = fold_out / "checkpoints" / "best.ckpt"
            preds = fold_out / "val_predictions.csv"
            if args.skip_if_done and best.exists() and preds.exists():
                print(f"[loso_cv] {s} fold_{fi}: already done, skipping")
                progress_rows.append({
                    "source": s, "fold": fi, "status": "skipped",
                    "n_train": int(len(tr_df)), "n_val": int(len(va_df)),
                })
                continue

            fold_seed = args.seed + 1000 * (all_sources.index(s) + 1) + fi
            print(f"\n[loso_cv] {s} fold_{fi}: train={len(tr_df)} val={len(va_df)} "
                  f"seed={fold_seed}")
            try:
                summary = _train_one(tr_df, va_df, fold_out, args, fold_seed)
                progress_rows.append({
                    "source": s, "fold": fi, "status": "ok",
                    "n_train": int(len(tr_df)), "n_val": int(len(va_df)),
                    "val_threshold_kappa": summary["val_threshold_kappa"],
                    "val_threshold_youden": summary["val_threshold_youden"],
                })
            except Exception as exc:
                print(f"[loso_cv] ERROR on {s} fold_{fi}: {exc}")
                traceback.print_exc()
                progress_rows.append({
                    "source": s, "fold": fi, "status": "error",
                    "n_train": int(len(tr_df)), "n_val": int(len(va_df)),
                    "error": str(exc),
                })

            # Incremental save of progress after each fold
            pd.DataFrame(progress_rows).to_csv(out_dir / "training_progress.csv", index=False)

    print("\n" + "=" * 70)
    print("LOSO-CV TRAINING COMPLETE")
    print("=" * 70)
    print(pd.DataFrame(progress_rows).to_string())
    print(f"\nProgress log: {out_dir/'training_progress.csv'}")


if __name__ == "__main__":
    main()
