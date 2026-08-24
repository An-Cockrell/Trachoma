from __future__ import annotations

import argparse
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from .utils import ensure_dir, save_json, seed_everything
from .transforms import build_train_transforms, build_eval_transforms
from .data import build_legacy_dataframe, TrachomaDataModule
from .model import TrachomaLitModel
from .grid_search import run_grid_search, GridSpec

def parse_args():
    p = argparse.ArgumentParser()
    # Either provide a single CSV...
    p.add_argument("--csv", type=str, default=None, help="CSV with columns: image_path,label (and optional subject_id).")
    p.add_argument("--image_col", type=str, default="image_path")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--group_col", type=str, default=None, help="Optional column for leakage-safe grouped splitting (e.g., subject_id).")
    p.add_argument(
    "--group_cols",
    nargs="+",
    default=None,
    help="Optional: multiple columns for leakage-safe grouped splitting (e.g. source id).",)

    # ...or use your legacy two-CSV setup
    p.add_argument("--img_dir_m", type=str, default=None)
    p.add_argument("--img_keys_csv_m", type=str, default=None)
    p.add_argument("--img_dir_o", type=str, default=None)
    p.add_argument("--img_keys_csv_o", type=str, default=None)
    p.add_argument("--img_col_o", type=str, default="imagename")
    p.add_argument("--label_col_o", type=str, default="consensus")
    p.add_argument("--clean_paths_txt", type=str, default=None)

    # Train params
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=30)
    #parameter sweep args
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--freeze_backbone_epochs", type=int, default=None)
    p.add_argument("--oversample", action="store_true", help="Use WeightedRandomSampler to balance classes in training.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")

    p.add_argument("--auto_tune", action="store_true",
              help="Run grid search at start IF any of lr/weight_decay/freeze_backbone_epochs were not explicitly provided.")

    p.add_argument("--tune_epochs", type=int, default=5)
    p.add_argument("--tune_limit_train_batches", type=int, default=None)
    p.add_argument("--tune_limit_val_batches", type=int, default=None)

    p.add_argument("--grid_lr", nargs="+", type=float, default=[1e-4, 3e-4, 6e-4])
    p.add_argument("--grid_wd", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--grid_freeze", nargs="+", type=int, default=[0, 1, 2])
    return p.parse_args()

def _load_df(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv)
        # normalize expected column names internally
        if args.image_col != "image_path":
            df = df.rename(columns={args.image_col: "image_path"})
        if args.label_col != "label":
            df = df.rename(columns={args.label_col: "label"})
        df["label"] = df["label"].astype(int)
        return df
    # legacy
    required = [args.img_dir_m, args.img_keys_csv_m, args.img_dir_o, args.img_keys_csv_o]
    if any(x is None for x in required):
        raise ValueError("Provide either --csv OR all legacy args: --img_dir_m --img_keys_csv_m --img_dir_o --img_keys_csv_o")
    return build_legacy_dataframe(
        img_dir_m=args.img_dir_m,
        img_keys_csv_m=args.img_keys_csv_m,
        img_dir_o=args.img_dir_o,
        img_keys_csv_o=args.img_keys_csv_o,
        img_col_o=args.img_col_o,
        label_col_o=args.label_col_o,
        clean_paths_txt=args.clean_paths_txt,
    )

def _compute_pos_weight(train_labels: np.ndarray) -> float:
    # pos_weight = N_neg / N_pos for BCEWithLogitsLoss
    n_pos = float((train_labels == 1).sum())
    n_neg = float((train_labels == 0).sum())
    if n_pos <= 0:
        return 1.0
    return n_neg / max(n_pos, 1.0)

def _save_split_csvs(run_dir: Path, dm, keys: list[str] | None = None) -> None:
    """
    Save deterministic split membership for later eval.
    If keys is None, we save by image_path only (works in most cases).
    If keys is provided (e.g. ['source','id']), we save those too for auditing.
    """
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    cols = ["image_path"]
    if keys:
        cols = keys + cols

    dm.train_df[cols].to_csv(splits_dir / "train.csv", index=False)
    dm.val_df[cols].to_csv(splits_dir / "val.csv", index=False)
    dm.test_df[cols].to_csv(splits_dir / "test.csv", index=False)


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)

    df = _load_df(args)

    run_name = args.run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path("runs") / run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    train_tfms = build_train_transforms(args.img_size)
    eval_tfms = build_eval_transforms(args.img_size)

        # ---- AUTO GRID SEARCH (only if user didn't pass some hyperparams) ----
    need_tune = (args.lr is None) or (args.weight_decay is None) or (args.freeze_backbone_epochs is None)

    if args.auto_tune and need_tune:
        # lock any hyperparams the user DID pass, and tune only the missing ones
        fixed = {}
        if args.lr is not None:
            fixed["lr"] = float(args.lr)
        if args.weight_decay is not None:
            fixed["weight_decay"] = float(args.weight_decay)
        if args.freeze_backbone_epochs is not None:
            fixed["freeze_backbone_epochs"] = int(args.freeze_backbone_epochs)

        grid = GridSpec(
            lr=args.grid_lr,
            weight_decay=args.grid_wd,
            freeze_backbone_epochs=args.grid_freeze,
        )

        tune_dir = ensure_dir(run_dir / "tuning")
        best = run_grid_search(
            df=df,
            train_tfms=train_tfms,
            eval_tfms=eval_tfms,
            out_dir=tune_dir,
            grid=grid,
            seed=args.seed,
            group_cols=(args.group_cols if args.group_cols else ([args.group_col] if args.group_col else None)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tune_epochs=args.tune_epochs,
            limit_train_batches=args.tune_limit_train_batches,
            limit_val_batches=args.tune_limit_val_batches,
            fixed_params=fixed,
        )

        # Fill in any missing hyperparams from tuned best
        if args.lr is None:
            args.lr = float(best["lr"])
        if args.weight_decay is None:
            args.weight_decay = float(best["weight_decay"])
        if args.freeze_backbone_epochs is None:
            args.freeze_backbone_epochs = int(best["freeze_backbone_epochs"])

        print("AUTO-TUNE selected:", {"lr": args.lr, "weight_decay": args.weight_decay,
                                     "freeze_backbone_epochs": args.freeze_backbone_epochs})
    else:
        # If user didn't enable auto_tune, but also didn't pass values, pick sensible defaults
        if args.lr is None:
            args.lr = 3e-4
        if args.weight_decay is None:
            args.weight_decay = 1e-2
        if args.freeze_backbone_epochs is None:
            args.freeze_backbone_epochs = 1

    dm = TrachomaDataModule(
        df=df,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        oversample=args.oversample,
        seed=args.seed,
        group_col=args.group_col,
        group_cols=args.group_cols,
    )
    dm.setup()
    # Save deterministic splits so eval uses the exact same samples
    _save_split_csvs(run_dir, dm, keys=args.group_cols if args.group_cols else ([args.group_col] if args.group_col else None))


    pos_weight = _compute_pos_weight(dm.train_df["label"].values.astype(int))

    model = TrachomaLitModel(
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val_kappa",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(monitor="val_kappa", mode="max", patience=7),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision = "16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=20,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)
    test_out = trainer.test(model, datamodule=dm, ckpt_path="best")

    # Save summary
    save_json(run_dir / "metrics.json", {
        "run_name": run_name,
        "pos_weight": float(pos_weight),
        "n_train": int(len(dm.train_df)),
        "n_val": int(len(dm.val_df)),
        "n_test": int(len(dm.test_df)),
        "test": test_out,
    })

    # Save chosen threshold (from last val sweep logged)
    # The best checkpoint logs its own val_best_threshold during training; for convenience, also store model.threshold here.
    save_json(run_dir / "threshold.json", {"threshold": float(model.threshold)})

    print(f"Done. Best checkpoint: {ckpt_dir/'best.ckpt'}")

if __name__ == "__main__":
    main()
