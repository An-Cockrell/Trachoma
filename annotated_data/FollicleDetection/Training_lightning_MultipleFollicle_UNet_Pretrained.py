import gc
import json
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
from scipy import ndimage
from skimage import io
from skimage.morphology import remove_small_objects
from PIL import Image


MIN_BLOB_SIZE = 20  # pixels — blobs smaller than this are treated as noise

_calib_path = os.path.join(os.path.dirname(__file__), "sticker_calibration.json")
with open(_calib_path) as _f:
    MIN_FOLLICLE_RATIO = float(json.load(_f)["min_follicle_ratio"])
# min_follicle_area_px = MIN_FOLLICLE_RATIO * eyelid_area_px  (both at 512×512)
# Matches the post-processing filter in MultipleFollicleTest_TF_prediction.py exactly.

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IMG_SIZE = 512
# ImageNet normalization — required to match the pretrained encoder's expected input distribution
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ── Data ───────────────────────────────────────────────────────────────────────

class ToTensor:
    def __call__(self, img):
        # img: H x W x C  →  C x H x W, float [0, 1]
        return torch.from_numpy(img.transpose((2, 0, 1)) / 255).float()


class TrachomaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, imgs, transform=None, image_aug=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = imgs
        self.transform = transform
        self.image_aug = image_aug

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
            stacked = self.transform[0](stacked)  # → (4, H, W) tensor, [0,1] unnormalized
            eyelid_area = int((stacked[:3].sum(0) > 0).sum().item())
            img_channels = self.image_aug(stacked[:3]) if self.image_aug is not None else stacked[:3]
            image = self.transform[1](img_channels)  # normalize after augmentation
            mask  = stacked[3]
        elif self.transform is not None:
            stacked = self.transform(stacked)
            eyelid_area = int((stacked[:3].sum(0) > 0).sum().item())
            img_channels = self.image_aug(stacked[:3]) if self.image_aug is not None else stacked[:3]
            image = img_channels
            mask  = stacked[3]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1) / 255).float()
            mask  = torch.from_numpy(mask).float()
            eyelid_area = int((image.sum(0) > 0).sum().item())
            if self.image_aug is not None:
                image = self.image_aug(image)

        # Filter ground-truth mask blobs below the WHO 0.5 mm threshold.
        # Uses the same ratio and resolution (512×512) as inference-time filtering.
        min_area = max(MIN_BLOB_SIZE, int(MIN_FOLLICLE_RATIO * eyelid_area))
        mask_np = remove_small_objects((mask.numpy() > 0), max_size=min_area)
        mask = torch.from_numpy(mask_np).to(torch.int)
        return {"image": image, "label": mask, "name": name}


class TrachomaDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, mask_dir, trans_train, trans_eval,
                 batch_size=6, num_workers=4, image_aug=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.trans_train = trans_train
        self.trans_eval = trans_eval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_aug = image_aug

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
            self.img_dir, self.mask_dir, self.train_imgs,
            transform=self.trans_train, image_aug=self.image_aug,
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


def count_follicles(binary_mask):
    """Count connected follicle blobs, ignoring noise below MIN_BLOB_SIZE."""
    clean = remove_small_objects(binary_mask.astype(bool), max_size=MIN_BLOB_SIZE)
    _, n = ndimage.label(clean)
    return n


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
            encoder_weights="imagenet",
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
        # Encoder pretrained on ImageNet: fine-tune at 0.1x LR.
        # Decoder + head initialized randomly: train at full LR.
        optimizer = optim.Adam([
            {"params": self.model.encoder.parameters(),           "lr": self.lr * 0.1},
            {"params": self.model.decoder.parameters(),           "lr": self.lr},
            {"params": self.model.segmentation_head.parameters(), "lr": self.lr},
        ])
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

    # Image-only augmentations applied before normalization (ColorJitter needs [0,1] input).
    # Kept moderate so the distribution shift post-normalization is small (~1σ vs the
    # 3σ shift the original gamma lambda caused). Targets domain gap from CC_EA2017 and
    # allTZphotos (warmer tones, older cameras, finger eversion). Never applied to val/test.
    image_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.4),
    ])

    num_workers = min(max(os.cpu_count() - 2, 1), 8)

    dm = TrachomaDataModule(
        img_dir,
        mask_dir,
        trans_train=trans_train,
        trans_eval=trans_eval,
        batch_size=6,
        num_workers=num_workers,
        image_aug=image_aug,
    )

    model = FollicleUNet(encoder="efficientnet-b4", lr=1e-3)
    run_experiment("MultipleFollicle_EfficientNetB4_UNet_Pretrained_v3", dm, model)
