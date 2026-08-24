from __future__ import annotations

import os
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, List, Dict

import numpy as np
import pandas as pd
from PIL import Image


import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

def _read_clean_paths(clean_paths_txt: Optional[str]) -> Optional[set[str]]:
    if not clean_paths_txt:
        return None
    with open(clean_paths_txt, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f.read().splitlines() if line.strip()])

def build_legacy_dataframe(
    img_dir_m: str,
    img_keys_csv_m: str,
    img_dir_o: str,
    img_keys_csv_o: str,
    img_col_o: str = "imagename",
    label_col_o: str = "consensus",
    clean_paths_txt: Optional[str] = None,
) -> pd.DataFrame:
    """Recreate your older 'm' + 'o' combined dataframe, but in a single consistent schema.

    Output columns: image_path, label, dataset, key
    label is 0/1
    """
    # 'm' csv expected to have columns: key, TF, TI (based on your older code)
    m = pd.read_csv(img_keys_csv_m)
    # remove TI-only images (as in your previous pipeline)
    if {"TI", "TF"}.issubset(set(m.columns)):
        m = m[~((m["TI"] == 1) & (m["TF"] == 0))]
    # keep consistent columns
    if "key" not in m.columns:
        raise ValueError("img_keys_csv_m must contain a 'key' column.")
    if "TF" not in m.columns:
        raise ValueError("img_keys_csv_m must contain a 'TF' column with 0/1 labels.")
    m = m[["key", "TF"]].copy()
    m["dataset"] = "m"
    m["image_path"] = m["key"].apply(lambda x: os.path.join(img_dir_m, f"image{int(x)}.jpg"))

    # 'o' csv expected to have imagename + consensus with 'TF'/'No TF'
    o = pd.read_csv(img_keys_csv_o, usecols=[img_col_o, label_col_o])
    o = o.rename(columns={img_col_o: "key", label_col_o: "label_text"})
    o["dataset"] = "o"
    # map text to 0/1 (matches your older mapping)
    o["TF"] = o["label_text"].replace({"No TF": 0, "TF": 1})
    o["image_path"] = o["key"].apply(lambda x: os.path.join(img_dir_o, f"0{str(x)}.jpg"))
    o = o[["key", "TF", "dataset", "image_path"]].copy()

    df = pd.concat([m.rename(columns={"TF": "label"}), o.rename(columns={"TF": "label"})], ignore_index=True)

    clean = _read_clean_paths(clean_paths_txt)
    if clean is not None:
        df = df[df["image_path"].isin(clean)].copy()

    df = df.dropna(subset=["image_path", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return df

def stratified_split(df: pd.DataFrame, label_col: str = "label", seed: int = 1234) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 split, stratified by label."""
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def group_stratified_split(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("source", "id"),
    label_col: str = "label",
    seed: int = 1234,
    test_size: float = 0.2,
    val_size: float = 0.1,   # fraction of full dataset (train = 1 - test - val)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe approximate split: groups stay together.

    Groups are defined by the unique combination of `group_cols`
    (e.g., ('source','ID') where IDs repeat across sources).

    Strategy:
    - Collapse to group-level labels (max label in group).
    - Split groups stratified by group label.
    - Pull rows for each split via merge on group_cols.
    """
    from sklearn.model_selection import train_test_split

    if not (0 < test_size < 1) or not (0 <= val_size < 1) or (test_size + val_size) >= 1:
        raise ValueError("Require 0<test_size<1, 0<=val_size<1, and test_size+val_size<1.")

    group_cols = list(group_cols)

    # 1) group-level labels
    g = df.groupby(group_cols, as_index=False)[label_col].max()

    # 2) stratified group split (train / temp)
    g_train, g_temp = train_test_split(
        g,
        test_size=(test_size + val_size),
        stratify=g[label_col],
        random_state=seed,
    )

    # 3) stratified split of temp into val/test
    if val_size == 0:
        g_val = g_temp.iloc[0:0].copy()
        g_test = g_temp
    else:
        # fraction of temp that should become test
        test_frac_of_temp = test_size / (test_size + val_size)
        g_val, g_test = train_test_split(
            g_temp,
            test_size=test_frac_of_temp,
            stratify=g_temp[label_col],
            random_state=seed,
        )

    # 4) pull full rows by merging on the group keys
    train_df = df.merge(g_train[group_cols], on=group_cols, how="inner")
    val_df   = df.merge(g_val[group_cols],   on=group_cols, how="inner")
    test_df  = df.merge(g_test[group_cols],  on=group_cols, how="inner")

    # 5) optional sanity checks: no group overlap
    train_groups = set(map(tuple, g_train[group_cols].to_numpy()))
    val_groups   = set(map(tuple, g_val[group_cols].to_numpy()))
    test_groups  = set(map(tuple, g_test[group_cols].to_numpy()))
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

class ImageCSVData(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, return_paths: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        sample = {"image": img, "label": torch.tensor(label, dtype=torch.long)}
        if self.return_paths:
            sample["image_path"] = path
        return sample

class TrachomaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        train_tfms,
        eval_tfms,
        batch_size: int = 16,
        num_workers: int = 4,
        oversample: bool = False,
        seed: int = 1234,
        group_col: Optional[str] = None,
        group_cols: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.df = df
        self.train_tfms = train_tfms
        self.eval_tfms = eval_tfms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.oversample = oversample
        self.seed = seed
        self.group_col = group_col
        self.group_cols = group_cols

    def setup(self, stage: Optional[str] = None):
        if self.group_cols is not None:
            gcols = list(self.group_cols)
        elif self.group_col:
            gcols = [self.group_col]
        else:
            gcols = None

        if gcols is not None:
            missing = [c for c in gcols if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing group columns in df: {missing}. Available: {list(self.df.columns)}")

        if gcols is not None:
            train_df, val_df, test_df = group_stratified_split(
                self.df,
                group_cols=gcols,
                seed=self.seed,
            )
        else:
            train_df, val_df, test_df = stratified_split(self.df, seed=self.seed)

        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df
        

        self.train_ds = ImageCSVData(train_df, transform=self.train_tfms, return_paths=False)
        self.val_ds   = ImageCSVData(val_df, transform=self.eval_tfms, return_paths=False)
        self.test_ds  = ImageCSVData(test_df, transform=self.eval_tfms, return_paths=True)

        # Compute weights for oversampling (if requested)
        if self.oversample:
            labels = train_df["label"].values.astype(int)
            class_counts = np.bincount(labels, minlength=2)
            # inverse frequency
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[labels]
            self._sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
        else:
            self._sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(self._sampler is None),
            sampler=self._sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )
