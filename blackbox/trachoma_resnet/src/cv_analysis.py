"""
Tier A analog for the 5-fold full-data cross-validation experiment.

Loads each fold's best.ckpt and the cached val predictions, then produces every
"Tier A" analysis from the paper's Results section in both PER-FOLD (mean +/- std
across the 5 folds) and ENSEMBLE (arithmetic mean of fold probabilities) form:

  1. Overall metrics + AUROC + AUPRC, each with bootstrap 95% CIs (n=10000)
  2. Per-source metrics table (image level)
  3. Subject-level aggregation (mean probability per source+id) + per-source
  4. Predicted vs. true TF prevalence, per source (image and subject level)
  5. Calibration: Brier, ECE, reliability diagram (ensemble)
  6. Operating-point analysis on the ensemble

Threshold rules (all val-selected, applied verbatim to test):
  * Per-fold:  threshold swept on that fold's val_predictions.csv
  * Ensemble:  threshold swept on the OOF stack (union of all 5 fold val
               predictions, each image scored only by its own fold's model)

The OOF stack works as the leakage-free ensemble-threshold proxy because the
5 fold val splits are disjoint by construction (asserted in cv_split.py).

Usage (run from blackbox/trachoma_resnet/ in the project venv):

  /home/Trachoma/venv/bin/python -m src.cv_analysis \
      --csv /home/Trachoma/data/all_metadata.csv \
      --splits_dir runs/resnet_cv5/splits \
      --runs_dir   runs/resnet_cv5 \
      --n_folds    5 \
      --objective  kappa
      # produces eval_outputs/resnet_cv5/tier_a/

  /home/Trachoma/venv/bin/python -m src.cv_analysis \
      ... --objective youden --out_dir eval_outputs/resnet_cv5/tier_a_youden
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from .data import ImageCSVData
from .transforms import build_eval_transforms
from .model import TrachomaLitModel
from .utils import ensure_dir, save_json
from .tier_a_analysis import (
    infer,
    compute_metrics,
    bootstrap_ci,
    sweep_threshold,
    expected_calibration_error,
)


def _load_test(splits_dir: Path, meta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / "test.csv")
    for col in ("source", "id", "image_path"):
        if col not in df.columns:
            raise ValueError(f"test.csv missing required column: {col}")
    if "label" not in df.columns:
        meta = pd.read_csv(meta_csv)[["image_path", "label"]]
        df = df.merge(meta, on="image_path", how="left")
    if df["label"].isna().any():
        raise RuntimeError("test.csv has rows with missing labels")
    df["label"] = df["label"].astype(int)
    return df


def _per_fold_inference(
    runs_dir: Path, n_folds: int, test_df: pd.DataFrame,
    img_size: int, batch_size: int, num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Returns probs of shape (n_folds, n_test), labels, paths (aligned to test_df rows)."""
    probs_all = []
    canonical_paths = None
    canonical_labels = None
    for f in range(n_folds):
        ckpt = runs_dir / f"fold_{f}" / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        print(f"[cv_analysis] inferring fold_{f} on test ({len(test_df)} images) ...")
        model = TrachomaLitModel.load_from_checkpoint(str(ckpt))
        probs, labels, paths = infer(model, test_df, img_size, batch_size, num_workers)
        if canonical_paths is None:
            canonical_paths = paths
            canonical_labels = labels
        else:
            if paths != canonical_paths:
                raise RuntimeError(f"fold_{f}: inference returned a different test ordering")
            if not np.array_equal(labels, canonical_labels):
                raise RuntimeError(f"fold_{f}: label vector mismatch vs fold_0")
        probs_all.append(probs)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.vstack(probs_all), canonical_labels, canonical_paths


def _per_fold_val_predictions(runs_dir: Path, n_folds: int) -> pd.DataFrame:
    """Concatenate every fold's val_predictions.csv. Disjoint by construction (cv_split)."""
    frames = []
    for f in range(n_folds):
        p = runs_dir / f"fold_{f}" / "val_predictions.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing val predictions: {p}")
        d = pd.read_csv(p)
        d["fold"] = f
        frames.append(d)
    oof = pd.concat(frames, ignore_index=True)
    if oof["image_path"].duplicated().any():
        n_dup = int(oof["image_path"].duplicated().sum())
        raise RuntimeError(f"OOF stack has {n_dup} duplicate image_path entries — folds are not disjoint")
    return oof


def _per_fold_metrics(test_probs_by_fold: np.ndarray, test_y: np.ndarray,
                      fold_thresholds: List[float]) -> pd.DataFrame:
    """Per-fold metrics on the locked test set, each fold using its own val threshold."""
    rows = []
    for f, thr in enumerate(fold_thresholds):
        m = compute_metrics(test_y, test_probs_by_fold[f], thr)
        m["fold"] = f
        m["threshold"] = float(thr)
        rows.append(m)
    return pd.DataFrame(rows)


def _per_fold_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    cols = ["kappa", "auroc", "auprc", "sensitivity", "specificity",
            "precision", "f1", "accuracy", "brier"]
    rows = []
    for k in cols:
        vals = per_fold[k].to_numpy(dtype=float)
        rows.append({
            "metric": k,
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min":  float(np.nanmin(vals)),
            "max":  float(np.nanmax(vals)),
        })
    rows.append({
        "metric": "threshold",
        "mean": float(per_fold["threshold"].mean()),
        "std":  float(per_fold["threshold"].std(ddof=1)) if len(per_fold) > 1 else 0.0,
        "min":  float(per_fold["threshold"].min()),
        "max":  float(per_fold["threshold"].max()),
    })
    return pd.DataFrame(rows).set_index("metric")


def _per_source_table_image(y: np.ndarray, probs: np.ndarray,
                            src: np.ndarray, thr: float) -> pd.DataFrame:
    rows = []
    for s in sorted(set(src)):
        m = (src == s)
        d = compute_metrics(y[m], probs[m], thr)
        d["source"] = s
        rows.append(d)
    return pd.DataFrame(rows).set_index("source")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Full metadata CSV (for label/source/id merge fallback).")
    ap.add_argument("--splits_dir", required=True, help="From cv_split.py (contains test.csv).")
    ap.add_argument("--runs_dir", required=True, help="Where fold_<i>/ live.")
    ap.add_argument("--out_dir", default=None,
                    help="Defaults to eval_outputs/<runs_dir name>/tier_a (or tier_a_<objective> if not kappa).")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--objective", choices=["kappa", "youden", "sens95"], default="kappa",
                    help="Validation-selection objective for thresholds.")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--allow_overwrite", action="store_true",
                    help="By default, errors if --out_dir already has results; pass this to overwrite.")
    return ap.parse_args()


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    splits_dir = Path(args.splits_dir)

    if args.out_dir is None:
        tag = "tier_a" if args.objective == "kappa" else f"tier_a_{args.objective}"
        args.out_dir = f"eval_outputs/{runs_dir.name}/{tag}"
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise SystemExit(
            f"[cv_analysis] {out_dir} already exists and is non-empty. "
            f"Pass --allow_overwrite to write into it anyway."
        )
    out_dir = ensure_dir(out_dir)
    print(f"[cv_analysis] out_dir = {out_dir}")
    print(f"[cv_analysis] objective = {args.objective}  n_boot = {args.n_boot}")

    # ----- Load test set -----
    test_df = _load_test(splits_dir, args.csv)
    print(f"[cv_analysis] test n = {len(test_df)}  test prev = {test_df['label'].mean():.4f}")

    # ----- Per-fold inference on test -----
    test_probs_by_fold, test_y, test_paths = _per_fold_inference(
        runs_dir, args.n_folds, test_df, args.img_size, args.batch_size, args.num_workers
    )
    np.save(out_dir / "test_probs_by_fold.npy", test_probs_by_fold)
    pd.DataFrame({"image_path": test_paths, "label": test_y}).to_csv(
        out_dir / "test_index.csv", index=False
    )

    # Source/id aligned to inference order (image_path is the join key)
    src_by_path = test_df.set_index("image_path")["source"]
    id_by_path = test_df.set_index("image_path")["id"]
    test_src = np.array([src_by_path[p] for p in test_paths])
    test_id = np.array([id_by_path[p] for p in test_paths])

    # ----- Per-fold thresholds: sweep on each fold's val_predictions -----
    fold_val_thresholds = []
    for f in range(args.n_folds):
        vp = pd.read_csv(runs_dir / f"fold_{f}" / "val_predictions.csv")
        thr = sweep_threshold(vp["prob"].to_numpy(dtype=float),
                              vp["label"].to_numpy(dtype=int),
                              args.objective)
        fold_val_thresholds.append(float(thr))
    print(f"[cv_analysis] per-fold val-{args.objective} thresholds: "
          + ", ".join(f"{t:.2f}" for t in fold_val_thresholds))

    # ----- Ensemble threshold: sweep on OOF stack (disjoint by construction) -----
    oof = _per_fold_val_predictions(runs_dir, args.n_folds)
    thr_ens = sweep_threshold(
        oof["prob"].to_numpy(dtype=float),
        oof["label"].to_numpy(dtype=int),
        args.objective,
    )
    thr_ens_mean = float(np.mean(fold_val_thresholds))
    print(f"[cv_analysis] ensemble val-{args.objective} threshold (OOF sweep) = {thr_ens:.2f}")
    print(f"[cv_analysis] reference: mean of per-fold thresholds = {thr_ens_mean:.2f}")

    # ----- Per-fold metrics on test (each fold @ own val threshold) -----
    per_fold_overall = _per_fold_metrics(test_probs_by_fold, test_y, fold_val_thresholds)
    per_fold_overall.to_csv(out_dir / "per_fold_overall_image.csv", index=False)
    per_fold_summary = _per_fold_summary(per_fold_overall)
    per_fold_summary.to_csv(out_dir / "per_fold_overall_summary.csv")

    # ----- Ensemble metrics on test -----
    ensemble_probs = test_probs_by_fold.mean(axis=0)
    np.save(out_dir / "test_probs_ensemble.npy", ensemble_probs)

    overall_ens = compute_metrics(test_y, ensemble_probs, thr_ens)
    overall_ens_ci = bootstrap_ci(test_y, ensemble_probs, thr_ens, args.n_boot, args.seed)

    # ----- Per-source breakdown (ensemble + per-fold) -----
    per_source_ens = _per_source_table_image(test_y, ensemble_probs, test_src, thr_ens)
    per_source_ens.to_csv(out_dir / "per_source_image_metrics_ensemble.csv")

    per_source_per_fold_rows = []
    for f in range(args.n_folds):
        thr_f = fold_val_thresholds[f]
        tab_f = _per_source_table_image(test_y, test_probs_by_fold[f], test_src, thr_f)
        tab_f = tab_f.reset_index()
        tab_f["fold"] = f
        tab_f["threshold"] = thr_f
        per_source_per_fold_rows.append(tab_f)
    per_source_per_fold = pd.concat(per_source_per_fold_rows, ignore_index=True)
    per_source_per_fold.to_csv(out_dir / "per_source_image_metrics_per_fold.csv", index=False)

    # Per-source mean/std across folds (image level)
    cols_metric = ["kappa", "auroc", "auprc", "sensitivity", "specificity",
                   "precision", "f1", "accuracy", "brier"]
    psf = per_source_per_fold.copy()
    summary_rows = []
    for s in sorted(psf["source"].unique()):
        sub = psf[psf["source"] == s]
        row = {"source": s, "n": int(sub["n"].iloc[0])}
        for k in cols_metric:
            row[f"{k}_mean"] = float(sub[k].mean())
            row[f"{k}_std"] = float(sub[k].std(ddof=1)) if len(sub) > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).set_index("source").to_csv(
        out_dir / "per_source_image_metrics_per_fold_summary.csv"
    )

    # ----- Subject-level aggregation (ensemble) -----
    subj = (
        pd.DataFrame({"source": test_src, "id": test_id,
                      "prob": ensemble_probs, "label": test_y})
        .groupby(["source", "id"], as_index=False)
        .agg(prob=("prob", "mean"),
             label=("label", "max"),
             n_images=("prob", "size"))
    )
    subj_y = subj["label"].to_numpy().astype(int)
    subj_probs = subj["prob"].to_numpy().astype(float)
    subj_src = subj["source"].to_numpy()

    subject_overall_ens = compute_metrics(subj_y, subj_probs, thr_ens)
    subject_overall_ens_ci = bootstrap_ci(subj_y, subj_probs, thr_ens, args.n_boot, args.seed)
    per_source_subj_ens = _per_source_table_image(subj_y, subj_probs, subj_src, thr_ens)
    per_source_subj_ens.to_csv(out_dir / "per_source_subject_metrics_ensemble.csv")

    # ----- Predicted vs true prevalence (ensemble) -----
    prev_rows = []
    for s in sorted(set(test_src)):
        m_img = (test_src == s)
        m_subj = (subj_src == s)
        prev_rows.append({
            "source": s,
            "n_images": int(m_img.sum()),
            "true_prev_image": float(test_y[m_img].mean()),
            "pred_prev_image": float((ensemble_probs[m_img] >= thr_ens).mean()),
            "n_subjects": int(m_subj.sum()),
            "true_prev_subject": float(subj_y[m_subj].mean()),
            "pred_prev_subject": float((subj_probs[m_subj] >= thr_ens).mean()),
        })
    prev_df = pd.DataFrame(prev_rows).set_index("source")
    prev_df.to_csv(out_dir / "prevalence_by_source_ensemble.csv")

    # ----- Calibration (ensemble) -----
    cal = expected_calibration_error(test_y, ensemble_probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    bx = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in cal["bins"]]
    by = [b["accuracy"] for b in cal["bins"]]
    ax.plot(bx, by, "o-", label="ensemble")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed TF frequency")
    ax.set_title(f"Reliability diagram ({runs_dir.name}, ensemble)\n"
                 f"Brier={overall_ens['brier']:.4f}  ECE={cal['ece']:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_diagram_ensemble.png", dpi=200)
    plt.close(fig)

    # ----- Operating-point analysis (ensemble) -----
    op_rows = []
    for t in np.round(np.linspace(0.05, 0.95, 19), 2):
        d = compute_metrics(test_y, ensemble_probs, float(t))
        d["threshold"] = float(t)
        op_rows.append(d)
    operating = pd.DataFrame(op_rows).set_index("threshold")
    operating.to_csv(out_dir / "operating_points_ensemble.csv")

    # ----- Recommended operating points (val-selected on OOF) -----
    recommended = {}
    for name, obj in [("kappa_optimal", "kappa"),
                      ("youden_optimal", "youden"),
                      ("screening_sens95", "sens95")]:
        thr_obj = sweep_threshold(
            oof["prob"].to_numpy(dtype=float),
            oof["label"].to_numpy(dtype=int),
            obj,
        )
        recommended[name] = {
            "threshold": float(thr_obj),
            "test_ensemble": compute_metrics(test_y, ensemble_probs, float(thr_obj)),
        }

    # ----- Per-source kappa bar chart (ensemble) -----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = per_source_ens["kappa"].sort_values()
    ax.barh(ks.index.astype(str), ks.values)
    ax.axvline(0.7, color="red", ls="--", label="WHO certification (kappa=0.70)")
    ax.set_xlabel("Cohen's kappa (image level, ensemble)")
    ax.set_title(f"Per-source TF classification kappa — {runs_dir.name} ensemble")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_source_kappa_ensemble.png", dpi=200)
    plt.close(fig)

    # ----- Predicted vs true prevalence scatter (ensemble) -----
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(prev_df["true_prev_image"].max(),
              prev_df["pred_prev_image"].max()) * 1.15 + 0.01
    ax.plot([0, lim], [0, lim], "--", color="gray")
    ax.scatter(prev_df["true_prev_image"], prev_df["pred_prev_image"])
    for s, r in prev_df.iterrows():
        ax.annotate(str(s), (r["true_prev_image"], r["pred_prev_image"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("True TF prevalence")
    ax.set_ylabel("Predicted TF prevalence (ensemble)")
    ax.set_title(f"Per-source prevalence estimation — {runs_dir.name} ensemble")
    fig.tight_layout()
    fig.savefig(out_dir / "prevalence_scatter_ensemble.png", dpi=200)
    plt.close(fig)

    # ----- Summary JSON -----
    summary = {
        "runs_dir": str(runs_dir),
        "splits_dir": str(splits_dir),
        "n_folds": int(args.n_folds),
        "n_test": int(len(test_y)),
        "n_test_subjects": int(len(subj)),
        "objective": args.objective,
        "n_boot": int(args.n_boot),
        "per_fold_val_thresholds": fold_val_thresholds,
        "ensemble_threshold_oof": float(thr_ens),
        "ensemble_threshold_mean_of_folds": thr_ens_mean,
        "per_fold_image_summary": per_fold_summary.reset_index().to_dict(orient="records"),
        "ensemble_image_level": {"metrics": overall_ens, "ci95": overall_ens_ci},
        "ensemble_subject_level": {"metrics": subject_overall_ens, "ci95": subject_overall_ens_ci},
        "ensemble_calibration": {"brier": overall_ens["brier"], "ece": cal["ece"]},
        "recommended_operating_points_ensemble": recommended,
    }
    save_json(out_dir / "cv_summary.json", summary)

    # ----- Console report -----
    print("\n" + "=" * 64)
    print(f"CV ENSEMBLE TIER A — {runs_dir.name}  (objective = {args.objective})")
    print("=" * 64)
    print(f"Ensemble threshold (OOF-swept) = {thr_ens:.2f}")
    print(f"\n-- Image level (n={len(test_y)}) [ENSEMBLE @ {thr_ens:.2f}] --")
    for k in ["kappa", "auroc", "auprc", "sensitivity", "specificity",
              "precision", "f1", "accuracy", "brier"]:
        lo, hi = overall_ens_ci.get(k, [float("nan"), float("nan")])
        print(f"  {k:12s} {overall_ens[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  ECE          {cal['ece']:.4f}")

    print(f"\n-- Per-fold image-level (test set, each fold @ own val threshold) --")
    print(per_fold_summary.to_string())

    print(f"\n-- Subject level (n={len(subj)}) [ENSEMBLE] --")
    for k in ["kappa", "auroc", "auprc", "sensitivity", "specificity"]:
        lo, hi = subject_overall_ens_ci.get(k, [float("nan"), float("nan")])
        print(f"  {k:12s} {subject_overall_ens[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    print(f"\n-- Per-source kappa (image level, ensemble) --")
    for s, r in per_source_ens.iterrows():
        print(f"  {str(s):40s} n={int(r['n']):5d}  prev={r['prevalence']:.3f}  "
              f"kappa={r['kappa']:.3f}  auroc={r['auroc']:.3f}")

    print(f"\n-- Recommended operating points (val-OOF selected, applied to ensemble) --")
    for name, d in recommended.items():
        t = d["test_ensemble"]
        print(f"  {name:18s} thr={d['threshold']:.2f}  "
              f"sens={t['sensitivity']:.3f}  spec={t['specificity']:.3f}  "
              f"kappa={t['kappa']:.3f}")

    print(f"\nArtifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
