"""
Tier C — Leave-One-Source-Out (LOSO) cross-region generalization for the
trachoma_resnet TF classifier.

For each data source S in turn:
  * the ENTIRE source S is held out as the test set,
  * a fresh ResNet-50 is trained on all OTHER sources,
  * a leakage-safe (source, id) group split carves a validation set out of the
    training pool,
  * the kappa-optimal threshold is selected on that in-distribution validation
    set and applied verbatim to the unseen held-out source.

This measures true cross-region generalization: how the model performs on a
source it has never seen, vs. the in-distribution per-source numbers from the
Tier A analysis. It mirrors the multiregional ablation in Yazbeck et al. 2026.

Results are written incrementally after every source, so a crash mid-run never
loses completed folds.

Run in the project venv (NOT conda — conda sklearn has a numpy ABI mismatch):

  cd /home/Trachoma/blackbox/trachoma_resnet
  /home/Trachoma/venv/bin/python -m src.loso_train \
      --csv /home/Trachoma/data/all_metadata.csv \
      --out_dir runs/loso \
      --max_epochs 25

Smoke test before committing to the overnight run (1 source, 1 epoch, few batches):

  /home/Trachoma/venv/bin/python -m src.loso_train \
      --csv /home/Trachoma/data/all_metadata.csv --out_dir runs/loso_smoke \
      --sources ICAPS --max_epochs 1 \
      --limit_train_batches 4 --limit_val_batches 4

Outputs (per held-out source) under <out_dir>/<source>/:
  checkpoints/best.ckpt, metrics.json, predictions.csv
And aggregated: <out_dir>/loso_summary.csv, <out_dir>/loso_summary.json
"""

from __future__ import annotations

import argparse
import re
import traceback
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import ImageCSVData, group_stratified_split
from .transforms import build_train_transforms, build_eval_transforms
from .model import TrachomaLitModel
from .utils import ensure_dir, save_json, seed_everything
from .tier_a_analysis import infer, compute_metrics, bootstrap_ci, sweep_threshold


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def _pos_weight(labels: np.ndarray) -> float:
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    return n_neg / max(n_pos, 1.0)


def _make_train_loader(df, tfms, batch_size, num_workers, oversample):
    ds = ImageCSVData(df, transform=tfms, return_paths=False)
    sampler = None
    if oversample:
        labels = df["label"].to_numpy().astype(int)
        class_counts = np.bincount(labels, minlength=2)
        class_w = 1.0 / np.maximum(class_counts, 1)
        sample_w = class_w[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w), replacement=True,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Metadata CSV (id,label,source,image_path).")
    p.add_argument("--out_dir", default="runs/loso")
    p.add_argument("--sources", nargs="+", default=None,
                   help="Restrict to these held-out sources (default: all).")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of the training pool held out for validation.")
    # Hyperparameters — defaults are the earlier grid-search optimum.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_backbone_epochs", type=int, default=1)
    p.add_argument("--oversample", action="store_true",
                   help="Add WeightedRandomSampler on top of BCE pos_weight.")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    # Smoke-test knobs (passed straight to the Lightning Trainer).
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)

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

    # In-distribution per-source Tier A numbers, for the generalization-gap column.
    indist = None
    indist_path = Path("eval_outputs/resnet_01/tier_a/per_source_image_metrics.csv")
    if indist_path.exists():
        indist = pd.read_csv(indist_path).set_index("source")

    train_tfms = build_train_transforms(args.img_size)
    eval_tfms = build_eval_transforms(args.img_size)

    print(f"[loso] {len(sources)} held-out source(s): {sources}")
    results = []

    for s in sources:
        tag = _sanitize(s)
        src_dir = ensure_dir(out_dir / tag)
        print("\n" + "=" * 70)
        print(f"[loso] HELD-OUT SOURCE: {s}")
        print("=" * 70)
        try:
            test_df = df[df["source"] == s].reset_index(drop=True)
            pool = df[df["source"] != s].reset_index(drop=True)

            # Leakage-safe (source, id) group split of the training pool.
            train_df, _empty, val_df = group_stratified_split(
                pool, group_cols=("source", "id"), seed=args.seed,
                test_size=args.val_frac, val_size=0.0,
            )
            print(f"[loso] train={len(train_df)}  val={len(val_df)}  "
                  f"test(held-out)={len(test_df)}  "
                  f"test prevalence={test_df['label'].mean():.4f}")

            pos_weight = _pos_weight(train_df["label"].to_numpy().astype(int))

            train_dl = _make_train_loader(
                train_df, train_tfms, args.batch_size, args.num_workers, args.oversample)
            val_dl = DataLoader(
                ImageCSVData(val_df, transform=eval_tfms), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(args.num_workers > 0))

            model = TrachomaLitModel(
                lr=args.lr, weight_decay=args.weight_decay, pos_weight=pos_weight,
                freeze_backbone_epochs=args.freeze_backbone_epochs)

            ckpt_cb = ModelCheckpoint(
                dirpath=str(src_dir / "checkpoints"), filename="best",
                monitor="val_kappa", mode="max", save_top_k=1, save_last=False)
            trainer = pl.Trainer(
                default_root_dir=str(src_dir),
                max_epochs=args.max_epochs,
                accelerator="auto", devices="auto",
                precision="16-mixed" if torch.cuda.is_available() else "32-true",
                log_every_n_steps=20,
                limit_train_batches=args.limit_train_batches,
                limit_val_batches=args.limit_val_batches,
                callbacks=[
                    ckpt_cb,
                    EarlyStopping(monitor="val_kappa", mode="max", patience=args.patience),
                    LearningRateMonitor(logging_interval="epoch"),
                ],
            )
            trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

            best_path = ckpt_cb.best_model_path or str(src_dir / "checkpoints" / "best.ckpt")
            print(f"[loso] best checkpoint: {best_path}")

            # Evaluate: threshold selected on in-distribution val, applied to held-out test.
            model = TrachomaLitModel.load_from_checkpoint(best_path)
            val_probs, val_y, _ = infer(
                model, val_df, args.img_size, args.batch_size, args.num_workers)
            thr = sweep_threshold(val_probs, val_y, "kappa")
            test_probs, test_y, test_paths = infer(
                model, test_df, args.img_size, args.batch_size, args.num_workers)

            m = compute_metrics(test_y, test_probs, thr)
            ci = bootstrap_ci(test_y, test_probs, thr, args.n_boot, args.seed)

            pd.DataFrame({"image_path": test_paths, "label": test_y,
                          "prob": test_probs}).to_csv(
                src_dir / "predictions.csv", index=False)

            row = {
                "held_out_source": s,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "test_prevalence": float(test_df["label"].mean()),
                "threshold": float(thr),
                "kappa": m["kappa"], "kappa_lo": ci["kappa"][0], "kappa_hi": ci["kappa"][1],
                "auroc": m["auroc"], "auroc_lo": ci["auroc"][0], "auroc_hi": ci["auroc"][1],
                "auprc": m["auprc"],
                "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "precision": m["precision"], "f1": m["f1"], "brier": m["brier"],
                "best_ckpt": best_path,
            }
            if indist is not None and s in indist.index:
                row["indist_kappa"] = float(indist.loc[s, "kappa"])
                row["indist_auroc"] = float(indist.loc[s, "auroc"])
                row["delta_kappa_loso_minus_indist"] = row["kappa"] - row["indist_kappa"]
                row["delta_auroc_loso_minus_indist"] = row["auroc"] - row["indist_auroc"]

            save_json(src_dir / "metrics.json", {"metrics": m, "ci95": ci, "summary": row})
            results.append(row)
            print(f"[loso] {s}: held-out kappa={m['kappa']:.3f} "
                  f"AUROC={m['auroc']:.3f} sens={m['sensitivity']:.3f} "
                  f"spec={m['specificity']:.3f}")

        except Exception as exc:  # noqa: BLE001 — keep the overnight run alive
            print(f"[loso] ERROR on source {s}: {exc}")
            traceback.print_exc()
            results.append({"held_out_source": s, "error": str(exc)})

        # Incremental save after every source.
        summ = pd.DataFrame(results)
        summ.to_csv(out_dir / "loso_summary.csv", index=False)
        save_json(out_dir / "loso_summary.json", {"results": results})

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOSO CROSS-REGION GENERALIZATION — SUMMARY")
    print("=" * 70)
    ok = [r for r in results if "error" not in r]
    for r in ok:
        line = (f"  {str(r['held_out_source'])[:34]:34s} "
                f"n_test={r['n_test']:5d} prev={r['test_prevalence']:.3f}  "
                f"kappa={r['kappa']:.3f} AUROC={r['auroc']:.3f}")
        if "indist_kappa" in r:
            line += (f"   (in-dist kappa={r['indist_kappa']:.3f}, "
                     f"Dkappa={r['delta_kappa_loso_minus_indist']:+.3f})")
        print(line)
    if ok:
        ks = np.array([r["kappa"] for r in ok], dtype=float)
        ars = np.array([r["auroc"] for r in ok], dtype=float)
        print(f"\n  mean held-out kappa  = {np.nanmean(ks):.3f}")
        print(f"  mean held-out AUROC  = {np.nanmean(ars):.3f}")
    print(f"\nSummary: {out_dir/'loso_summary.csv'}")


if __name__ == "__main__":
    main()
