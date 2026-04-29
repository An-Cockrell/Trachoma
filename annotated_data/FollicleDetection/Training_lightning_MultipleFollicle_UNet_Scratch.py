import gc
import os
import random
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import (
    BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex,
)
import segmentation_models_pytorch as smp
from skimage import io
from PIL import Image


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IMG_SIZE = 512
# Dataset-specific normalization stats computed from SAMFollicleMasks images
IMG_MEAN = [0.147780, 0.084905, 0.079360]
IMG_STD  = [0.255321, 0.155361, 0.147369]


# ── Data ───────────────────────────────────────────────────────────────────────

class ToTensor:
    def __call__(self, img):
        # img: H x W x C  →  C x H x W, float [0, 1]
        return torch.from_numpy(img.transpose((2, 0, 1)) / 255).float()


class TrachomaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, imgs, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        image = io.imread(os.path.join(self.img_dir, name))
        mask = np.array(
            Image.open(os.path.join(self.mask_dir, name)).convert("1")
        ).astype(int)

        # Stack image + mask so spatial transforms are applied identically to both
        stacked = np.dstack((image, mask))  # H x W x 4

        if isinstance(self.transform, list):
            # transform[0]: geometric transforms applied to stacked (image+mask)
            # transform[1]: normalization applied to image channels only
            stacked = self.transform[0](stacked)  # → (4, H, W) tensor
            image = self.transform[1](stacked[:3])
            mask  = stacked[3]
        elif self.transform is not None:
            stacked = self.transform(stacked)
            image = stacked[:3]
            mask  = stacked[3]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1) / 255).float()
            mask  = torch.from_numpy(mask).float()

        mask = (mask > 0).to(torch.int)
        return {"image": image, "label": mask, "name": name}


class TrachomaDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, mask_dir, trans_train, trans_eval,
                 batch_size=6, num_workers=4):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.trans_train = trans_train
        self.trans_eval = trans_eval
        self.batch_size = batch_size
        self.num_workers = num_workers

        all_imgs = sorted([
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
        ])

        rng = np.random.default_rng(SEED)
        idx = np.arange(len(all_imgs))
        rng.shuffle(idx)

        n_test = int(len(idx) * 0.15)
        n_val  = int(len(idx) * 0.15)

        self.test_imgs  = [all_imgs[i] for i in idx[:n_test]]
        self.val_imgs   = [all_imgs[i] for i in idx[n_test:n_test + n_val]]
        self.train_imgs = [all_imgs[i] for i in idx[n_test + n_val:]]

        print(
            f"Split → train: {len(self.train_imgs)}, "
            f"val: {len(self.val_imgs)}, test: {len(self.test_imgs)}"
        )

    def setup(self, stage=None):
        self.ds_train = TrachomaDataset(
            self.img_dir, self.mask_dir, self.train_imgs, transform=self.trans_train
        )
        self.ds_val = TrachomaDataset(
            self.img_dir, self.mask_dir, self.val_imgs, transform=self.trans_eval
        )
        self.ds_test = TrachomaDataset(
            self.img_dir, self.mask_dir, self.test_imgs, transform=self.trans_eval
        )

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self): return self._loader(self.ds_train, shuffle=True)
    def val_dataloader(self):   return self._loader(self.ds_val,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.ds_test,  shuffle=False)


# ── Loss ───────────────────────────────────────────────────────────────────────

def f_score(pr, gt, beta=1, eps=1e-7):
    pr = torch.sigmoid(pr)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    return ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)


class BCEDiceLoss(nn.Module):
    def __init__(self, eps=1e-7, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__()
        self.eps = eps
        self.lambda_dice = lambda_dice
        self.lambda_bce  = lambda_bce
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, y_pr, y_gt):
        y_gt = y_gt.float()
        dice = 1 - f_score(y_pr, y_gt, eps=self.eps)
        bce  = self.bce(y_pr, y_gt)
        return self.lambda_dice * dice + self.lambda_bce * bce


# ── Lightning module ───────────────────────────────────────────────────────────

class FollicleUNet(pl.LightningModule):
    def __init__(self, encoder="efficientnet-b4", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,  # training from scratch — no pretrained weights
            in_channels=3,
            classes=1,
        )
        self.loss = BCEDiceLoss()
        self.lr   = lr

        self.train_precision = BinaryPrecision(threshold=0.5)
        self.val_precision   = BinaryPrecision(threshold=0.5)
        self.test_precision  = BinaryPrecision(threshold=0.5)

        self.train_recall = BinaryRecall(threshold=0.5)
        self.val_recall   = BinaryRecall(threshold=0.5)
        self.test_recall  = BinaryRecall(threshold=0.5)

        self.train_f1 = BinaryF1Score(threshold=0.5)
        self.val_f1   = BinaryF1Score(threshold=0.5)
        self.test_f1  = BinaryF1Score(threshold=0.5)

        self.val_iou  = BinaryJaccardIndex(threshold=0.5)
        self.test_iou = BinaryJaccardIndex(threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        images, targets = batch["image"], batch["label"]
        logits  = self.model(images).squeeze(1)          # (B, H, W)
        targets = targets.squeeze(1).to(dtype=logits.dtype)
        loss    = self.loss(logits, targets)
        preds   = torch.sigmoid(logits).flatten()
        tgts    = targets.long().flatten()
        return loss, preds, tgts

    def training_step(self, batch, batch_idx):
        loss, preds, tgts = self._shared_step(batch)
        bs = batch["image"].size(0)
        self.log("train_loss",      loss,                  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True, batch_size=bs)
        self.train_precision(preds, tgts)
        self.train_recall(preds, tgts)
        self.train_f1(preds, tgts)
        self.log("train_precision", self.train_precision,  on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_recall",    self.train_recall,     on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_f1",        self.train_f1,         on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, tgts = self._shared_step(batch)
        bs = batch["image"].size(0)
        self.log("val_loss",      loss,               prog_bar=True, sync_dist=True, batch_size=bs)
        self.val_precision(preds, tgts)
        self.val_recall(preds, tgts)
        self.val_f1(preds, tgts)
        self.val_iou(preds, tgts)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_recall",    self.val_recall,    on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1",        self.val_f1,        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_iou",       self.val_iou,       on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, tgts = self._shared_step(batch)
        bs = batch["image"].size(0)
        self.log("test_loss",      loss,               sync_dist=True, batch_size=bs)
        self.test_precision(preds, tgts)
        self.test_recall(preds, tgts)
        self.test_f1(preds, tgts)
        self.test_iou(preds, tgts)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_recall",    self.test_recall,    on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_f1",        self.test_f1,        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_iou",       self.test_iou,       on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # All weights randomly initialized — uniform LR across the whole network.
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", patience=3),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ── Experiment runner ──────────────────────────────────────────────────────────

def run_experiment(run_name, datamodule, model, project="Trachoma_FollicleDetection_UNet"):
    print(f"Running: {run_name}")
    wandb_logger = WandbLogger(project=project, name=run_name)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=f"Checkpoints/{run_name}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=50,
        log_every_n_steps=20,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir="Checkpoints",
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()

    del trainer, model, datamodule, wandb_logger
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    img_dir  = "./MultipleFollicleImages/SAMFollicleMasks/Images"
    mask_dir = "./MultipleFollicleImages/SAMFollicleMasks/Masks"

    norm = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)

    trans_eval = [
        transforms.Compose([
            ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]),
        norm,
    ]
    trans_train = [
        transforms.Compose([
            ToTensor(),
            transforms.Resize((int(IMG_SIZE * 1.05), int(IMG_SIZE * 1.05))),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.3),
            transforms.CenterCrop(IMG_SIZE),
        ]),
        norm,
    ]

    num_workers = min(max(os.cpu_count() - 2, 1), 8)

    dm = TrachomaDataModule(
        img_dir,
        mask_dir,
        trans_train=trans_train,
        trans_eval=trans_eval,
        batch_size=6,
        num_workers=num_workers,
    )

    model = FollicleUNet(encoder="efficientnet-b4", lr=1e-3)
    run_experiment("MultipleFollicle_EfficientNetB4_UNet_Scratch", dm, model)
