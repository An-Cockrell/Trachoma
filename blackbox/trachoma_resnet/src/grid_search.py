from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from .data import ImageCSVData, group_stratified_split, stratified_split
from .model import TrachomaLitModel
from .utils import save_json


@dataclass(frozen=True)
class GridSpec:
    lr: Sequence[float]
    weight_decay: Sequence[float]
    freeze_backbone_epochs: Sequence[int]


def _make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_tfms,
    eval_tfms,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = ImageCSVData(train_df, transform=train_tfms, return_paths=False)
    val_ds = ImageCSVData(val_df, transform=eval_tfms, return_paths=False)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **dl_kwargs)
    return train_dl, val_dl


def run_grid_search(
    df: pd.DataFrame,
    train_tfms,
    eval_tfms,
    out_dir: Path,
    *,
    grid: GridSpec,
    seed: int = 1234,
    group_cols: Optional[Sequence[str]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    tune_epochs: int = 5,
    # Optional speed knobs
    limit_train_batches: Optional[int] = None,
    limit_val_batches: Optional[int] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs a small grid search and returns + saves the best hyperparameters.

    - Splits ONCE (grouped if group_cols provided), then reuses that same split for all grid points.
    - Scores by val_kappa (from your LightningModule logging).
    - Saves a JSON summary to out_dir / "grid_search_best.json".
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_params = fixed_params or {}

    # ---- deterministic split ONCE ----
    if group_cols:
        train_df, val_df, _ = group_stratified_split(df, group_cols=group_cols, seed=seed)
    else:
        train_df, val_df, _ = stratified_split(df, seed=seed)

    # prebuild loaders once per config? transforms vary? (not here) so we can rebuild per config cheaply
    best = {"val_kappa": -1e9, "params": None}

    # Build the actual grid, but allow fixed params to lock dimensions
    lr_list = [fixed_params["lr"]] if "lr" in fixed_params else list(grid.lr)
    wd_list = [fixed_params["weight_decay"]] if "weight_decay" in fixed_params else list(grid.weight_decay)
    fr_list = [fixed_params["freeze_backbone_epochs"]] if "freeze_backbone_epochs" in fixed_params else list(grid.freeze_backbone_epochs)

    results = []

    for lr, wd, freeze in product(lr_list, wd_list, fr_list):
        pl.seed_everything(seed, workers=True)

        train_dl, val_dl = _make_loaders(
            train_df, val_df, train_tfms, eval_tfms,
            batch_size=batch_size, num_workers=num_workers
        )

        model = TrachomaLitModel(
            lr=float(lr),
            weight_decay=float(wd),
            pos_weight=None,  # you compute this in train.py; omit for tuning or compute similarly if you want
            freeze_backbone_epochs=int(freeze),
        )

        trainer = pl.Trainer(
            max_epochs=int(tune_epochs),
            accelerator="auto",
            devices="auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=50,
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            deterministic=True,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
        )

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Pull the epoch-level val_kappa
        metrics = trainer.callback_metrics
        val_kappa = metrics.get("val_kappa", None)
        val_kappa_f = float(val_kappa.detach().cpu().item()) if val_kappa is not None else float("nan")

        rec = {
            "lr": float(lr),
            "weight_decay": float(wd),
            "freeze_backbone_epochs": int(freeze),
            "val_kappa": val_kappa_f,
        }
        results.append(rec)

        if np.isfinite(val_kappa_f) and val_kappa_f > best["val_kappa"]:
            best["val_kappa"] = val_kappa_f
            best["params"] = {"lr": float(lr), "weight_decay": float(wd), "freeze_backbone_epochs": int(freeze)}

    payload = {
        "best": best,
        "fixed_params": fixed_params,
        "grid": {
            "lr": list(map(float, grid.lr)),
            "weight_decay": list(map(float, grid.weight_decay)),
            "freeze_backbone_epochs": list(map(int, grid.freeze_backbone_epochs)),
        },
        "tune_epochs": int(tune_epochs),
        "group_cols": list(group_cols) if group_cols else None,
        "results": results,
    }

    save_json(out_dir / "grid_search_best.json", payload)
    return payload["best"]["params"] or {}