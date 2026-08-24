"""
Re-score the LOSO folds at any validation-selected threshold objective
(kappa | youden | sens95) without re-training.

For each completed LOSO fold under <out_dir>/<source>/:
  1. Reconstruct that fold's training-pool + val split via group_stratified_split
     with the same seed used in src/loso_train.py.
  2. Load best.ckpt and run inference on the val split.
  3. Pick a threshold via sweep_threshold(val_probs, val_y, objective).
  4. Apply that threshold to the cached test predictions in predictions.csv.
  5. Compute metrics + bootstrap CIs.
  6. Emit loso_summary_<objective>.csv/json mirroring the kappa version.

Run from blackbox/trachoma_resnet/ in the project venv:

  /home/Trachoma/venv/bin/python -m src.rescore_loso \
      --csv /home/Trachoma/data/all_metadata.csv \
      --out_dir runs/loso \
      --objective youden
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import group_stratified_split
from .model import TrachomaLitModel
from .tier_a_analysis import infer, compute_metrics, bootstrap_ci, sweep_threshold
from .utils import ensure_dir, save_json, seed_everything


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="runs/loso")
    ap.add_argument("--objective", choices=["kappa", "youden", "sens95"],
                    default="youden")
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="Must match what loso_train.py used.")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    df = pd.read_csv(args.csv)
    for col in ("id", "label", "source", "image_path"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df["label"] = df["label"].astype(int)

    indist = None
    indist_path = Path("eval_outputs/resnet_01/tier_a_youden/per_source_image_metrics.csv") \
        if args.objective == "youden" else \
        Path("eval_outputs/resnet_01/tier_a/per_source_image_metrics.csv")
    if indist_path.exists():
        indist = pd.read_csv(indist_path).set_index("source")
        print(f"[rescore] in-dist reference: {indist_path}")

    all_sources = sorted(df["source"].unique())
    print(f"[rescore] {len(all_sources)} sources; objective={args.objective}")

    rows = []
    for s in all_sources:
        tag = _sanitize(s)
        src_dir = out_dir / tag
        ckpt = src_dir / "checkpoints" / "best.ckpt"
        preds_csv = src_dir / "predictions.csv"
        if not ckpt.exists() or not preds_csv.exists():
            print(f"[rescore] SKIP {s} — missing checkpoint or predictions")
            continue

        print(f"\n[rescore] === {s} ===")
        # Reconstruct val split (same seed, same split call as loso_train.py).
        pool = df[df["source"] != s].reset_index(drop=True)
        _train_df, _empty, val_df = group_stratified_split(
            pool, group_cols=("source", "id"), seed=args.seed,
            test_size=args.val_frac, val_size=0.0,
        )
        print(f"[rescore] val n={len(val_df)}")

        model = TrachomaLitModel.load_from_checkpoint(str(ckpt))
        val_probs, val_y, _ = infer(
            model, val_df, args.img_size, args.batch_size, args.num_workers)
        thr = sweep_threshold(val_probs, val_y, args.objective)
        print(f"[rescore] val-{args.objective} threshold = {thr:.2f}")

        test = pd.read_csv(preds_csv)
        test_y = test["label"].to_numpy().astype(int)
        test_probs = test["prob"].to_numpy().astype(float)

        m = compute_metrics(test_y, test_probs, thr)
        ci = bootstrap_ci(test_y, test_probs, thr, args.n_boot, args.seed)

        pred_prev = float((test_probs >= thr).mean())
        row = {
            "held_out_source": s,
            "n_test": int(len(test_y)),
            "test_prevalence": float(test_y.mean()),
            "pred_prevalence": pred_prev,
            "prevalence_error": pred_prev - float(test_y.mean()),
            "threshold": float(thr),
            "kappa": m["kappa"],
            "kappa_lo": ci["kappa"][0],
            "kappa_hi": ci["kappa"][1],
            "auroc": m["auroc"],
            "auroc_lo": ci["auroc"][0],
            "auroc_hi": ci["auroc"][1],
            "auprc": m["auprc"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "f1": m["f1"],
            "brier": m["brier"],
        }
        if indist is not None and s in indist.index:
            row["indist_kappa"] = float(indist.loc[s, "kappa"])
            row["indist_auroc"] = float(indist.loc[s, "auroc"])
            row["delta_kappa_loso_minus_indist"] = row["kappa"] - row["indist_kappa"]
            row["delta_auroc_loso_minus_indist"] = row["auroc"] - row["indist_auroc"]
        rows.append(row)

        print(f"[rescore] {s}: thr={thr:.2f}  kappa={m['kappa']:.3f}  "
              f"auroc={m['auroc']:.3f}  sens={m['sensitivity']:.3f}  "
              f"spec={m['specificity']:.3f}")

    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / f"loso_summary_{args.objective}.csv", index=False)
    save_json(out_dir / f"loso_summary_{args.objective}.json", {"results": rows})
    print(f"\n[rescore] wrote loso_summary_{args.objective}.csv/json to {out_dir}")


if __name__ == "__main__":
    main()
