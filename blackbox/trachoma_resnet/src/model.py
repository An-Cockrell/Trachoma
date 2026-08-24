from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchvision import models

from .utils import save_json


def build_resnet(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    m = models.resnet50(weights=weights)
    # Replace classifier head to output 1 logit
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, 1)
    return m


def _sweep_thresholds_for_kappa(
    probs: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float]:
    # returns best_thr, best_kappa
    from sklearn.metrics import cohen_kappa_score

    best_thr = 0.5
    best_k = -1.0
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        k = cohen_kappa_score(y_true, y_pred)
        if k > best_k:
            best_k = k
            best_thr = float(t)
    return best_thr, float(best_k)


class TrachomaLitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        pos_weight: Optional[float] = None,
        initial_threshold: float = 0.5,
        freeze_backbone_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = build_resnet(pretrained=True)

        # loss
        if pos_weight is None:
            self.pos_weight = None
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # metrics at a fixed threshold (updated each epoch by sweep)
        self.threshold = initial_threshold
        self.acc = torchmetrics.classification.BinaryAccuracy(threshold=self.threshold)
        self.prec = torchmetrics.classification.BinaryPrecision(
            threshold=self.threshold
        )
        self.rec = torchmetrics.classification.BinaryRecall(threshold=self.threshold)
        self.f1 = torchmetrics.classification.BinaryF1Score(threshold=self.threshold)
        self.kappa = torchmetrics.classification.BinaryCohenKappa(
            threshold=self.threshold
        )

        self._freeze_backbone_epochs = freeze_backbone_epochs

        # store val probs/labels for threshold sweep
        self._val_probs = []
        self._val_labels = []
        self._test_probs = []
        self._test_labels = []

    def forward(self, x):
        return self.net(x).squeeze(1)

    def on_fit_start(self):
        # move pos_weight to device (if used)
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def on_train_epoch_start(self):
        # optional: freeze backbone for first N epochs to stabilize head
        if self.current_epoch < self._freeze_backbone_epochs:
            for name, p in self.net.named_parameters():
                # keep classifier trainable
                if name.startswith("fc."):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for p in self.net.parameters():
                p.requires_grad = True

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"].float()
        logits = self(x)
        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"].float()
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)

        # accumulate for sweep
        self._val_probs.append(probs.detach().cpu())
        self._val_labels.append(y.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        if len(self._val_probs) == 0:
            return
        probs = torch.cat(self._val_probs).detach().float().cpu().numpy()
        labels = torch.cat(self._val_labels).detach().cpu().numpy().astype(int)

        thresholds = np.linspace(0.05, 0.95, 19)
        best_thr, best_kappa = _sweep_thresholds_for_kappa(probs, labels, thresholds)

        # update thresholded metrics modules
        self.threshold = best_thr
        self.acc.threshold = best_thr
        self.prec.threshold = best_thr
        self.rec.threshold = best_thr
        self.f1.threshold = best_thr
        self.kappa.threshold = best_thr

        # compute metrics at best threshold
        preds = (probs >= best_thr).astype(int)
        # manual compute in numpy then log
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            cohen_kappa_score,
        )

        self.log("val_kappa", cohen_kappa_score(labels, preds), prog_bar=True)
        self.log("val_acc", accuracy_score(labels, preds))
        self.log("val_precision", precision_score(labels, preds, zero_division=0))
        self.log("val_recall", recall_score(labels, preds, zero_division=0))
        self.log("val_f1", f1_score(labels, preds, zero_division=0))
        self.log("val_best_threshold", best_thr)

        # reset buffers
        self._val_probs.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"].float()
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)

        # accumulate for epoch-end metrics
        self._test_probs.append(probs.detach().cpu())
        self._test_labels.append(y.detach().cpu())

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_test_epoch_end(self):
        if len(self._test_probs) == 0:
            return

        probs = torch.cat(self._test_probs).detach().float().cpu().numpy()
        labels = torch.cat(self._test_labels).detach().cpu().numpy().astype(int)

        # Option 1 (recommended): use threshold selected from validation during training
        thr = float(self.threshold)

        # Option 2 (if you want): sweep again on test (usually not recommended)
        # thresholds = np.linspace(0.05, 0.95, 19)
        # thr, _ = _sweep_thresholds_for_kappa(probs, labels, thresholds)

        preds = (probs >= thr).astype(int)

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            cohen_kappa_score,
        )

        self.log("test_kappa", cohen_kappa_score(labels, preds), prog_bar=True)
        self.log("test_acc", accuracy_score(labels, preds))
        self.log("test_precision", precision_score(labels, preds, zero_division=0))
        self.log("test_recall", recall_score(labels, preds, zero_division=0))
        self.log("test_f1", f1_score(labels, preds, zero_division=0))
        self.log("test_threshold", thr)

        # reset buffers
        self._test_probs.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=3, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_kappa",
                "interval": "epoch",
                "frequency": 1,
            },
        }
