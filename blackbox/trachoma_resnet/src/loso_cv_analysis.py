"""
Aggregate the 40 LOSO-CV models (8 sources x 5 folds) into per-source
ensembles and produce the cross-region generalization tables.

For each held-out source S:
  * Load held_out_test.csv (the unseen test for S).
  * For each fold i in 0..4: load best.ckpt, infer on held-out -> fold_probs[i].
  * For each fold i: load val_predictions.csv, sweep threshold via --objective
    on that fold's val OOF -> fold_thr[i].
  * Per-fold held-out metrics: each fold uses its own val threshold.
  * Ensemble:
        ensemble_prob = mean(fold_probs[i])
        ensemble_thr  = sweep_threshold(OOF stack of all 5 val predictions)
  * Bootstrap 95% CIs on ensemble metrics (n=10000 by default).

Outputs (under --out_dir, default eval_outputs/loso_cv5/):
  loso_cv_summary_<objective>.csv           # per-source ensemble (the §15 table)
  loso_cv_summary_per_fold_<objective>.csv  # per-source per-fold (appendix)
  loso_cv_summary_per_source_std_<objective>.csv  # per-source mean +/- std across folds
  <source>/ensemble_predictions.csv         # cached ensemble probs on held-out
  <source>/per_fold_held_out_metrics.csv    # per-fold image-level metrics
  loso_cv_summary_<objective>.json          # everything combined

Run from blackbox/trachoma_resnet/ in the project venv:

  /home/Trachoma/venv/bin/python -m src.loso_cv_analysis \
      --csv /home/Trachoma/data/all_metadata.csv \
      --runs_dir runs/loso_cv5 \
      --objective kappa
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import torch

from .model import TrachomaLitModel
from .tier_a_analysis import (
    infer,
    compute_metrics,
    bootstrap_ci,
    sweep_threshold,
)
from .utils import ensure_dir, save_json, seed_everything


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Metadata CSV (used only to confirm label/source/id alignment).")
    ap.add_argument("--runs_dir", default="runs/loso_cv5",
                    help="Where the LOSO-CV training output lives.")
    ap.add_argument("--out_dir", default=None,
                    help="Defaults to eval_outputs/<runs_dir name>/.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--objective", choices=["kappa", "youden", "sens95"], default="kappa")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="Restrict to these held-out sources.")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--indist_csv", type=str, default=None,
                    help="Optional: path to per-source ensemble metrics from cv_analysis.py "
                         "for the LOSO-vs-in-dist delta column. "
                         "Defaults to eval_outputs/resnet_cv5/tier_a/per_source_image_metrics_ensemble.csv "
                         "(or _youden equivalent).")
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def _per_source_dirs(runs_dir: Path) -> List[Path]:
    out = []
    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and (p / "held_out_test.csv").exists():
            out.append(p)
    return out


def _infer_fold_on_test(ckpt: Path, held: pd.DataFrame, img_size: int,
                        batch_size: int, num_workers: int):
    model = TrachomaLitModel.load_from_checkpoint(str(ckpt))
    probs, labels, paths = infer(model, held, img_size, batch_size, num_workers)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, labels, paths


def main():
    args = parse_args()
    seed_everything(args.seed)

    runs_dir = Path(args.runs_dir)
    if args.out_dir is None:
        args.out_dir = f"eval_outputs/{runs_dir.name}"
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        # only block if there's already a summary for this objective
        if (out_dir / f"loso_cv_summary_{args.objective}.csv").exists():
            raise SystemExit(
                f"[loso_cv_analysis] {out_dir}/loso_cv_summary_{args.objective}.csv "
                f"already exists. Pass --allow_overwrite to regenerate."
            )
    out_dir = ensure_dir(out_dir)
    print(f"[loso_cv_analysis] runs_dir={runs_dir}  out_dir={out_dir}  "
          f"objective={args.objective}  n_boot={args.n_boot}")

    # In-distribution reference (CV ensemble per-source) for the delta column
    indist = None
    if args.indist_csv:
        indist_path = Path(args.indist_csv)
    else:
        if args.objective == "kappa":
            indist_path = Path("eval_outputs/resnet_cv5/tier_a/per_source_image_metrics_ensemble.csv")
        else:
            indist_path = Path(
                f"eval_outputs/resnet_cv5/tier_a_{args.objective}/per_source_image_metrics_ensemble.csv"
            )
    if indist_path.exists():
        indist = pd.read_csv(indist_path).set_index("source")
        print(f"[loso_cv_analysis] in-dist reference: {indist_path}")
    else:
        print(f"[loso_cv_analysis] no in-dist reference at {indist_path} -- delta columns will be empty")

    source_dirs = _per_source_dirs(runs_dir)
    if args.sources:
        keep = set(args.sources)
        wanted_dirs = [d for d in source_dirs if any(_sanitize(s) == d.name for s in keep)]
        if not wanted_dirs:
            wanted_dirs = [d for d in source_dirs if d.name in {_sanitize(s) for s in keep}]
        source_dirs = wanted_dirs
    print(f"[loso_cv_analysis] {len(source_dirs)} source directory(ies) to process")

    summary_rows = []
    per_fold_rows = []
    per_source_std_rows = []

    metric_keys = ["kappa", "auroc", "auprc", "sensitivity", "specificity",
                   "precision", "f1", "accuracy", "brier"]

    for src_dir in source_dirs:
        s = src_dir.name
        print(f"\n[loso_cv_analysis] {'=' * 60}\n[loso_cv_analysis] SOURCE: {s}\n[loso_cv_analysis] {'=' * 60}")

        held = pd.read_csv(src_dir / "held_out_test.csv")
        if "label" not in held.columns:
            meta = pd.read_csv(args.csv)[["image_path", "label"]]
            held = held.merge(meta, on="image_path", how="left")
        held["label"] = held["label"].astype(int)
        source_label = str(held["source"].iloc[0]) if len(held) else s
        print(f"[loso_cv_analysis] held-out n={len(held)}  prev={held['label'].mean():.4f}")

        # ----- Per-fold inference on held-out -----
        fold_probs = []
        fold_thresholds = []
        canonical_paths = canonical_labels = None
        n_folds_found = 0
        for fi in range(args.n_folds):
            fold_dir = src_dir / f"fold_{fi}"
            ckpt = fold_dir / "checkpoints" / "best.ckpt"
            vp = fold_dir / "val_predictions.csv"
            if not ckpt.exists() or not vp.exists():
                print(f"[loso_cv_analysis] {s} fold_{fi}: missing ckpt or val_predictions, skipping")
                continue
            probs, labels, paths = _infer_fold_on_test(
                ckpt, held, args.img_size, args.batch_size, args.num_workers
            )
            if canonical_paths is None:
                canonical_paths, canonical_labels = paths, labels
            else:
                if paths != canonical_paths:
                    raise RuntimeError(f"{s} fold_{fi}: inference order mismatch")
            fold_probs.append(probs)
            vpr = pd.read_csv(vp)
            thr = sweep_threshold(
                vpr["prob"].to_numpy(dtype=float),
                vpr["label"].to_numpy(dtype=int),
                args.objective,
            )
            fold_thresholds.append(float(thr))
            n_folds_found += 1

        if n_folds_found == 0:
            print(f"[loso_cv_analysis] {s}: no folds usable, skipping source")
            continue
        fold_probs = np.vstack(fold_probs)  # (n_folds_found, n_held)

        # Per-fold metrics on held-out (each fold @ own threshold)
        per_fold_image_rows = []
        for fi in range(n_folds_found):
            m = compute_metrics(canonical_labels, fold_probs[fi], fold_thresholds[fi])
            m["fold"] = fi
            m["threshold"] = fold_thresholds[fi]
            per_fold_image_rows.append(m)
            row = {"source": source_label, "fold": fi, "threshold": fold_thresholds[fi]}
            row.update({k: m[k] for k in metric_keys})
            per_fold_rows.append(row)
        pd.DataFrame(per_fold_image_rows).to_csv(
            src_dir / f"per_fold_held_out_metrics_{args.objective}.csv", index=False
        )

        # Ensemble probabilities + ensemble threshold (OOF stack of all folds' val preds)
        ensemble_probs = fold_probs.mean(axis=0)
        oof_frames = []
        for fi in range(args.n_folds):
            p = src_dir / f"fold_{fi}" / "val_predictions.csv"
            if p.exists():
                oof_frames.append(pd.read_csv(p))
        oof = pd.concat(oof_frames, ignore_index=True)
        if oof["image_path"].duplicated().any():
            raise RuntimeError(f"{s}: OOF stack has duplicates across folds")
        thr_ens = sweep_threshold(
            oof["prob"].to_numpy(dtype=float),
            oof["label"].to_numpy(dtype=int),
            args.objective,
        )

        m_ens = compute_metrics(canonical_labels, ensemble_probs, thr_ens)
        ci_ens = bootstrap_ci(canonical_labels, ensemble_probs, thr_ens,
                              args.n_boot, args.seed)

        # Cache ensemble predictions
        pd.DataFrame({
            "image_path": canonical_paths,
            "label": canonical_labels,
            "prob_ensemble": ensemble_probs,
            **{f"prob_fold_{i}": fold_probs[i] for i in range(n_folds_found)},
        }).to_csv(src_dir / f"ensemble_predictions_{args.objective}.csv", index=False)

        pred_prev_ens = float((ensemble_probs >= thr_ens).mean())
        true_prev = float(np.asarray(canonical_labels).mean())

        row = {
            "held_out_source": source_label,
            "n_test": int(len(canonical_labels)),
            "n_folds_used": int(n_folds_found),
            "test_prevalence": true_prev,
            "pred_prevalence_ensemble": pred_prev_ens,
            "prevalence_error_ensemble": pred_prev_ens - true_prev,
            "ensemble_threshold": float(thr_ens),
            "per_fold_thresholds": fold_thresholds,
            "kappa": m_ens["kappa"], "kappa_lo": ci_ens["kappa"][0], "kappa_hi": ci_ens["kappa"][1],
            "auroc": m_ens["auroc"], "auroc_lo": ci_ens["auroc"][0], "auroc_hi": ci_ens["auroc"][1],
            "auprc": m_ens["auprc"],
            "sensitivity": m_ens["sensitivity"],
            "specificity": m_ens["specificity"],
            "precision": m_ens["precision"],
            "f1": m_ens["f1"],
            "accuracy": m_ens["accuracy"],
            "brier": m_ens["brier"],
        }
        if indist is not None and source_label in indist.index:
            row["indist_kappa"] = float(indist.loc[source_label, "kappa"])
            row["indist_auroc"] = float(indist.loc[source_label, "auroc"])
            row["delta_kappa_loso_minus_indist"] = row["kappa"] - row["indist_kappa"]
            row["delta_auroc_loso_minus_indist"] = row["auroc"] - row["indist_auroc"]
        summary_rows.append(row)

        # Per-source mean +/- std across folds (for the appendix)
        pf = pd.DataFrame(per_fold_image_rows)
        std_row = {"source": source_label, "n": int(len(canonical_labels))}
        for k in metric_keys:
            std_row[f"{k}_mean"] = float(pf[k].mean())
            std_row[f"{k}_std"] = float(pf[k].std(ddof=1)) if len(pf) > 1 else 0.0
        per_source_std_rows.append(std_row)

        print(f"[loso_cv_analysis] {s}: ens_thr={thr_ens:.2f}  "
              f"kappa={m_ens['kappa']:.3f}  AUROC={m_ens['auroc']:.3f}  "
              f"sens={m_ens['sensitivity']:.3f}  spec={m_ens['specificity']:.3f}")

        # Incremental save
        pd.DataFrame(summary_rows).to_csv(
            out_dir / f"loso_cv_summary_{args.objective}.csv", index=False
        )

    # ----- Final summary -----
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / f"loso_cv_summary_{args.objective}.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv(
        out_dir / f"loso_cv_summary_per_fold_{args.objective}.csv", index=False
    )
    pd.DataFrame(per_source_std_rows).set_index("source").to_csv(
        out_dir / f"loso_cv_summary_per_source_std_{args.objective}.csv"
    )
    save_json(out_dir / f"loso_cv_summary_{args.objective}.json", {
        "runs_dir": str(runs_dir),
        "n_folds": int(args.n_folds),
        "objective": args.objective,
        "n_boot": int(args.n_boot),
        "indist_reference": str(indist_path) if indist is not None else None,
        "results": summary_rows,
    })

    print("\n" + "=" * 70)
    print(f"LOSO-CV {args.objective.upper()} — ENSEMBLE SUMMARY")
    print("=" * 70)
    for r in summary_rows:
        line = (f"  {r['held_out_source'][:36]:36s} n={r['n_test']:5d} "
                f"prev={r['test_prevalence']:.3f}  "
                f"kappa={r['kappa']:.3f} AUROC={r['auroc']:.3f}")
        if "indist_kappa" in r:
            line += (f"  (in-dist kappa={r['indist_kappa']:.3f}, "
                     f"Dkappa={r['delta_kappa_loso_minus_indist']:+.3f})")
        print(line)
    if summary_rows:
        ks = np.array([r["kappa"] for r in summary_rows], dtype=float)
        ars = np.array([r["auroc"] for r in summary_rows], dtype=float)
        print(f"\n  mean ensemble kappa  = {np.nanmean(ks):.3f}")
        print(f"  mean ensemble AUROC  = {np.nanmean(ars):.3f}")
    print(f"\nArtifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
