import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedGroupKFold

from DataLoader_lightning_MultipleFollicle_no_masks import ToTensor, CustomCrop, FollicleEnhance  # noqa: F401


class TrachomaMetaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_csv,
        transforms_0=None,
        transforms_1=None,
        batch_size=32,
        num_workers=0,
        dataPercent=1.0,
    ):
        super().__init__()
        self.metadata_csv = metadata_csv
        self.transforms_0 = transforms_0
        self.transforms_1 = transforms_1 if transforms_1 is not None else transforms_0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataPer = dataPercent
        self._split_data()

    def _split_data(self):
        df = pd.read_csv(self.metadata_csv)
        df = df.iloc[: int(len(df) * self.dataPer)].copy()

        # Patient-level groups prevent leakage across splits
        groups = (df["source"].astype(str) + "_" + df["id"].astype(str)).values
        labels = df["label"].values
        indices = np.arange(len(df))

        # 80/20 train+val vs test
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1234)
        train_val_idx, test_idx = next(sgkf.split(indices, labels, groups))

        # 80/20 train vs val from train+val
        tv_labels = labels[train_val_idx]
        tv_groups = groups[train_val_idx]
        tv_indices = np.arange(len(train_val_idx))

        sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_rel, val_rel = next(sgkf2.split(tv_indices, tv_labels, tv_groups))

        train_abs = train_val_idx[train_rel]
        val_abs = train_val_idx[val_rel]

        self.train_df = df.iloc[train_abs].reset_index(drop=True)
        self.val_df = df.iloc[val_abs].reset_index(drop=True)
        self.test_df = df.iloc[test_idx].reset_index(drop=True)

        print(
            f"[Split] Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}"
        )
        print(
            f"[Split] TF%  — Train: {self.train_df['label'].mean():.3f}, "
            f"Val: {self.val_df['label'].mean():.3f}, "
            f"Test: {self.test_df['label'].mean():.3f}"
        )

    def setup(self, stage=None):
        self.trachoma_train = TrachomaMetaDataset(
            self.train_df, transform=self.transforms_0
        )
        self.trachoma_val = TrachomaMetaDataset(
            self.val_df, transform=self.transforms_0
        )
        self.trachoma_test = TrachomaMetaDataset(
            self.test_df, transform=self.transforms_0
        )

    def _make_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            shuffle=False,
        )

    def train_dataloader(self):
        return self._make_loader(self.trachoma_train)

    def val_dataloader(self):
        return self._make_loader(self.trachoma_val)

    def test_dataloader(self):
        return self._make_loader(self.trachoma_test)


class TrachomaMetaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])

        image = io.imread(img_path)

        if self.transform is not None:
            if isinstance(self.transform, list):
                for t in self.transform:
                    image = t(image)
            else:
                image = self.transform(image)

        return {"image": image, "label": label, "path": img_path}
