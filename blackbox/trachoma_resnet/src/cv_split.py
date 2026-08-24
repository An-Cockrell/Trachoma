"""
Generate canonical CV splits for the 5-fold full-data experiment.

Pipeline (mirrors Yazbeck multiregion 2026: 20% held out, 5-fold CV on the rest):

  1. Read all_metadata.csv (all 23,104 imgs, Gambia included now that ids are fixed).
  2. Group-stratified 80/20 split by (source, id):
       80% -> CV pool, 20% -> held-out test set.
  3. Group-stratified 5-fold split on the 80%:
       Each fold produces train.csv + val.csv (each fold's val ~ 20% of the 80%).
  4. Save under <out_dir>/splits/{test.csv, fold_<i>/{train,val}.csv}.
  5. Sanity-assert no group leakage anywhere.

Group definition: unique (source, id) combination. Mirrors group_stratified_split()
in data.py. Folds are seeded deterministically.

Splits include source, id, image_path, label columns so downstream scripts don't
need to re-merge from the metadata CSV.

Usage (run from blackbox/trachoma_resnet/ in the project venv):

  /home/Trachoma/venv/bin/python -m src.cv_split \
      --csv /home/Trachoma/data/all_metadata.csv \
      --out_dir runs/resnet_cv5 \
      --n_folds 5 --test_frac 0.2 --seed 1234
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from .utils import ensure_dir, save_json


def _collapse_to_groups(df: pd.DataFrame, group_cols: Sequence[str],
                        label_col: str = "label") -> pd.DataFrame:
    """One row per group; group-level label = max(label within group)."""
    return df.groupby(list(group_cols), as_index=False)[label_col].max()


def _hold_out_test(df: pd.DataFrame, group_cols: Sequence[str],
                   test_frac: float, label_col: str, seed: int
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-stratified 80/20: returns (pool_rows, test_rows, pool_groups)."""
    gc = list(group_cols)
    g = _collapse_to_groups(df, gc, label_col)
    g_pool, g_test = train_test_split(
        g, test_size=test_frac, stratify=g[label_col], random_state=seed,
    )
    pool_df = df.merge(g_pool[gc], on=gc, how="inner")
    test_df = df.merge(g_test[gc], on=gc, how="inner")
    # No group leakage between pool and test
    pool_groups = set(map(tuple, g_pool[gc].to_numpy()))
    test_groups = set(map(tuple, g_test[gc].to_numpy()))
    assert pool_groups.isdisjoint(test_groups), "Group leakage between pool and test"
    return pool_df.reset_index(drop=True), test_df.reset_index(drop=True), g_pool


def _kfold_on_pool(pool_df: pd.DataFrame, g_pool: pd.DataFrame,
                   group_cols: Sequence[str], n_folds: int,
                   label_col: str, seed: int
                   ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """StratifiedKFold at the GROUP level; expand each fold to rows."""
    gc = list(group_cols)
    g_pool = g_pool.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fi, (train_idx, val_idx) in enumerate(skf.split(g_pool, g_pool[label_col])):
        g_train = g_pool.iloc[train_idx]
        g_val = g_pool.iloc[val_idx]
        train_df = pool_df.merge(g_train[gc], on=gc, how="inner")
        val_df = pool_df.merge(g_val[gc], on=gc, how="inner")
        tg = set(map(tuple, g_train[gc].to_numpy()))
        vg = set(map(tuple, g_val[gc].to_numpy()))
        assert tg.isdisjoint(vg), f"Group leakage in fold {fi}"
        folds.append((train_df.reset_index(drop=True), val_df.reset_index(drop=True)))
    return folds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Metadata CSV with id, label, source, image_path.")
    p.add_argument("--out_dir", default="runs/resnet_cv5",
                   help="Splits will be saved under <out_dir>/splits/.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--group_cols", nargs="+", default=["source", "id"])
    p.add_argument("--label_col", default="label")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    for c in list(args.group_cols) + [args.label_col, "image_path"]:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")
    df[args.label_col] = df[args.label_col].astype(int)

    splits_dir = ensure_dir(Path(args.out_dir) / "splits")
    save_cols = list(args.group_cols) + ["image_path", args.label_col]

    # 80/20 group-stratified hold-out
    pool_df, test_df, g_pool = _hold_out_test(
        df, args.group_cols, args.test_frac, args.label_col, args.seed
    )
    test_df[save_cols].to_csv(splits_dir / "test.csv", index=False)
    print(f"[cv_split] total n={len(df)}  pool n={len(pool_df)}  test n={len(test_df)}")
    print(f"[cv_split] test prevalence = {test_df[args.label_col].mean():.4f}")
    print(f"[cv_split] pool prevalence = {pool_df[args.label_col].mean():.4f}")
    print(f"[cv_split] per-source in test:")
    for s, n in test_df["source"].value_counts().items():
        tf = int((test_df[test_df["source"] == s][args.label_col] == 1).sum())
        print(f"  {str(s)[:50]:50s} n={n:5d}  TF={tf}")

    # 5-fold StratifiedKFold on groups inside the pool
    folds = _kfold_on_pool(
        pool_df, g_pool, args.group_cols, args.n_folds, args.label_col, args.seed
    )
    fold_info = []
    val_path_sets = []
    for i, (train_df, val_df) in enumerate(folds):
        fold_dir = ensure_dir(splits_dir / f"fold_{i}")
        train_df[save_cols].to_csv(fold_dir / "train.csv", index=False)
        val_df[save_cols].to_csv(fold_dir / "val.csv", index=False)
        info = {
            "fold": i,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "train_prevalence": float(train_df[args.label_col].mean()),
            "val_prevalence": float(val_df[args.label_col].mean()),
        }
        fold_info.append(info)
        val_path_sets.append(set(val_df["image_path"]))
        print(f"[cv_split] fold_{i}: train={info['n_train']:5d} val={info['n_val']:5d} "
              f"train_prev={info['train_prevalence']:.4f} val_prev={info['val_prevalence']:.4f}")

    # Cross-fold sanity: val folds are disjoint at the image level
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            assert val_path_sets[i].isdisjoint(val_path_sets[j]), \
                f"Val folds {i} and {j} share images — group split is broken"

    # Test set disjoint from pool at the image level
    pool_paths = set(pool_df["image_path"])
    test_paths = set(test_df["image_path"])
    assert pool_paths.isdisjoint(test_paths), "Test set overlaps with CV pool"

    save_json(splits_dir / "split_info.json", {
        "csv": args.csv,
        "n_total": int(len(df)),
        "n_pool": int(len(pool_df)),
        "n_test": int(len(test_df)),
        "test_prevalence": float(test_df[args.label_col].mean()),
        "pool_prevalence": float(pool_df[args.label_col].mean()),
        "n_folds": int(args.n_folds),
        "test_frac": float(args.test_frac),
        "seed": int(args.seed),
        "group_cols": list(args.group_cols),
        "folds": fold_info,
    })
    print(f"\n[cv_split] done. wrote {splits_dir}")


if __name__ == "__main__":
    main()
