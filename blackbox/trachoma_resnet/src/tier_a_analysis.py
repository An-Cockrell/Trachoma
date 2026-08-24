"""
Tier A analysis for the trachoma_resnet TF classifier.

Runs inference once on the saved validation and test splits, then produces every
"Tier A" analysis for the paper's Results section — no retraining required:

  1. Overall metrics + AUROC + AUPRC, each with bootstrap 95% CIs
  2. Per-source metrics table (image level)
  3. Subject-level aggregation (mean probability per source+id) + per-source
  4. Predicted vs. true TF prevalence, per source (image and subject level)
  5. Calibration: Brier score, Expected Calibration Error, reliability diagram
  6. Operating-point analysis (descriptive grid + thresholds selected on validation)

Thresholds are always *selected on the validation split* and applied verbatim to
the test split, so nothing is tuned on test.

All metrics are implemented in numpy — this script deliberately does NOT import
sklearn (the installed sklearn has a numpy ABI mismatch in this environment).

Usage (run as a module from blackbox/trachoma_resnet/):

  # raw-image model
  python -m src.tier_a_analysis \
      --ckpt runs/resnet_01/checkpoints/best.ckpt \
      --csv /home/Trachoma/data/all_metadata.csv \
      --split_dir runs/resnet_01/splits

  # eyelid-masked model
  python -m src.tier_a_analysis \
      --ckpt runs/resnet_masked_eyelid/checkpoints/best.ckpt \
      --csv /home/Trachoma/data/all_metadata_masked.csv \
      --split_dir runs/resnet_masked_eyelid/splits

Outputs are written to eval_outputs/<run_name>/tier_a/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


# ── Metric primitives (pure numpy) ────────────────────────────────────────────

def _counts(y: np.ndarray, pred: np.ndarray):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp, fp, tn, fn


def _kappa(y: np.ndarray, pred: np.ndarray) -> float:
    """Cohen's kappa for binary labels."""
    tp, fp, tn, fn = _counts(y, pred)
    n = tp + fp + tn + fn
    if n == 0:
        return float("nan")
    po = (tp + tn) / n
    p_pred_pos = (tp + fp) / n
    p_true_pos = (tp + fn) / n
    pe = p_pred_pos * p_true_pos + (1 - p_pred_pos) * (1 - p_true_pos)
    return float((po - pe) / (1 - pe)) if (1 - pe) != 0 else float("nan")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def _avg_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based average ranks (handles ties), without scipy."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    n = len(s)
    tmp = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        tmp[i:j] = (i + j - 1) / 2.0 + 1.0
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = tmp
    return ranks


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (tie-corrected)."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(scores)
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """AUPRC, step-wise definition matching sklearn.average_precision_score."""
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def _brier(y: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


# ── Composite metrics ─────────────────────────────────────────────────────────

def compute_metrics(y: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    """All metrics at a fixed threshold. Rank metrics need both classes present."""
    pred = (probs >= thr).astype(int)
    tp, fp, tn, fn = _counts(y, pred)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "prevalence": float(np.mean(y)) if len(y) else float("nan"),
        "kappa": _kappa(y, pred),
        "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "auroc": _roc_auc(y, probs) if both else float("nan"),
        "auprc": _average_precision(y, probs) if both else float("nan"),
        "brier": _brier(y, probs),
    }


def bootstrap_ci(
    y: np.ndarray, probs: np.ndarray, thr: float, n_boot: int = 2000, seed: int = 1234
) -> dict:
    """Percentile 95% CIs by resampling cases with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ["kappa", "accuracy", "sensitivity", "specificity",
            "precision", "f1", "auroc", "auprc", "brier"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], probs[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = compute_metrics(yb, pb, thr)
        for k in keys:
            draws[k].append(m[k])
    ci = {}
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr):
            ci[k] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        else:
            ci[k] = [float("nan"), float("nan")]
    return ci


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    """ECE plus per-bin stats for a reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "confidence": float("nan"), "accuracy": float("nan")})
            continue
        conf = float(probs[mask].mean())
        acc = float(y[mask].mean())
        ece += (n_bin / len(y)) * abs(acc - conf)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": n_bin,
                     "confidence": conf, "accuracy": acc})
    return {"ece": float(ece), "bins": rows}


def sweep_threshold(probs: np.ndarray, y: np.ndarray, objective: str) -> float:
    """Pick a threshold on validation. objective in {kappa, youden, sens95}."""
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_score = 0.5, -np.inf
    for t in grid:
        pred = (probs >= t).astype(int)
        tp, fp, tn, fn = _counts(y, pred)
        sens = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        if objective == "kappa":
            score = _kappa(y, pred)
        elif objective == "youden":
            score = (sens + spec - 1.0) if not (np.isnan(sens) or np.isnan(spec)) else -np.inf
        elif objective == "sens95":
            score = spec if (not np.isnan(sens) and sens >= 0.95) else -np.inf
        else:
            raise ValueError(objective)
        if score is not None and not np.isnan(score) and score > best_score:
            best_score, best_t = score, float(t)
    return best_t


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(model, df: pd.DataFrame, img_size: int, batch_size: int, num_workers: int):
    """Return probs, labels, paths aligned to df rows."""
    tfms = build_eval_transforms(img_size)
    ds = ImageCSVData(df, transform=tfms, return_paths=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    probs, labels, paths = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device)
            p = torch.sigmoid(model(x)).detach().cpu().numpy()
            probs.append(p)
            labels.append(batch["label"].cpu().numpy())
            paths.extend(batch["image_path"])
    return (
        np.concatenate(probs).astype(float),
        np.concatenate(labels).astype(int),
        list(paths),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True, help="Metadata CSV with id,label,source,image_path.")
    ap.add_argument("--split_dir", required=True, help="runs/<run>/splits/ with val.csv,test.csv.")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Main threshold. If omitted, swept for max kappa on validation.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    run_name = split_dir.parent.name
    out_dir = ensure_dir(args.out_dir or f"eval_outputs/{run_name}/tier_a")
    print(f"[tier_a] run={run_name}  out_dir={out_dir}")

    # ── Load metadata + splits, attach labels ─────────────────────────────────
    meta = pd.read_csv(args.csv)
    if "label" not in meta.columns:
        raise ValueError("Metadata CSV must contain a 'label' column.")
    labels_by_path = meta.set_index("image_path")["label"]

    def load_split(name: str) -> pd.DataFrame:
        s = pd.read_csv(split_dir / name)
        for col in ("source", "id", "image_path"):
            if col not in s.columns:
                raise ValueError(f"{name} missing column '{col}'.")
        s = s.copy()
        s["label"] = s["image_path"].map(labels_by_path)
        missing = int(s["label"].isna().sum())
        if missing:
            raise RuntimeError(f"{name}: {missing} image_path values absent from --csv.")
        s["label"] = s["label"].astype(int)
        return s

    val_df = load_split("val.csv")
    test_df = load_split("test.csv")
    print(f"[tier_a] val n={len(val_df)}  test n={len(test_df)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    model = TrachomaLitModel.load_from_checkpoint(args.ckpt)
    print("[tier_a] inferring on validation split ...")
    val_probs, val_y, _ = infer(model, val_df, args.img_size, args.batch_size, args.num_workers)
    print("[tier_a] inferring on test split ...")
    test_probs, test_y, test_paths = infer(model, test_df, args.img_size,
                                           args.batch_size, args.num_workers)

    # Align source/id to inference order via image_path.
    src_by_path = test_df.set_index("image_path")["source"]
    id_by_path = test_df.set_index("image_path")["id"]
    test_src = np.array([src_by_path[p] for p in test_paths])
    test_id = np.array([id_by_path[p] for p in test_paths])

    # ── Thresholds chosen on validation ───────────────────────────────────────
    thr_kappa = args.threshold if args.threshold is not None \
        else sweep_threshold(val_probs, val_y, "kappa")
    thr_youden = sweep_threshold(val_probs, val_y, "youden")
    thr_sens95 = sweep_threshold(val_probs, val_y, "sens95")
    print(f"[tier_a] thresholds (val-selected): kappa={thr_kappa:.2f} "
          f"youden={thr_youden:.2f} sens95={thr_sens95:.2f}")

    # ── 1. Overall metrics + bootstrap CIs ────────────────────────────────────
    overall = compute_metrics(test_y, test_probs, thr_kappa)
    overall_ci = bootstrap_ci(test_y, test_probs, thr_kappa, args.n_boot, args.seed)

    # ── 2. Per-source metrics (image level) ───────────────────────────────────
    def per_source_table(y, probs, src) -> pd.DataFrame:
        rows = []
        for s in sorted(set(src)):
            m = (src == s)
            d = compute_metrics(y[m], probs[m], thr_kappa)
            d["source"] = s
            rows.append(d)
        return pd.DataFrame(rows).set_index("source")

    per_source_img = per_source_table(test_y, test_probs, test_src)
    per_source_img.to_csv(out_dir / "per_source_image_metrics.csv")

    # ── 3. Subject-level aggregation (mean prob per source+id) ────────────────
    subj = (
        pd.DataFrame({"source": test_src, "id": test_id,
                      "prob": test_probs, "label": test_y})
        .groupby(["source", "id"], as_index=False)
        .agg(prob=("prob", "mean"), label=("label", "max"), n_images=("prob", "size"))
    )
    subj_y = subj["label"].to_numpy().astype(int)
    subj_probs = subj["prob"].to_numpy().astype(float)
    subj_src = subj["source"].to_numpy()

    subject_overall = compute_metrics(subj_y, subj_probs, thr_kappa)
    subject_overall_ci = bootstrap_ci(subj_y, subj_probs, thr_kappa, args.n_boot, args.seed)
    per_source_subj = per_source_table(subj_y, subj_probs, subj_src)
    per_source_subj.to_csv(out_dir / "per_source_subject_metrics.csv")

    # ── 4. Predicted vs. true prevalence per source ───────────────────────────
    prev_rows = []
    for s in sorted(set(test_src)):
        m_img = (test_src == s)
        m_subj = (subj_src == s)
        prev_rows.append({
            "source": s,
            "n_images": int(m_img.sum()),
            "true_prev_image": float(test_y[m_img].mean()),
            "pred_prev_image": float((test_probs[m_img] >= thr_kappa).mean()),
            "n_subjects": int(m_subj.sum()),
            "true_prev_subject": float(subj_y[m_subj].mean()),
            "pred_prev_subject": float((subj_probs[m_subj] >= thr_kappa).mean()),
        })
    prevalence = pd.DataFrame(prev_rows).set_index("source")
    prevalence.to_csv(out_dir / "prevalence_by_source.csv")

    # ── 5. Calibration ────────────────────────────────────────────────────────
    cal = expected_calibration_error(test_y, test_probs, n_bins=10)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    bx = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in cal["bins"]]
    by = [b["accuracy"] for b in cal["bins"]]
    ax.plot(bx, by, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed TF frequency")
    ax.set_title(f"Reliability diagram ({run_name})\n"
                 f"Brier={overall['brier']:.4f}  ECE={cal['ece']:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_diagram.png", dpi=200)
    plt.close(fig)

    # ── 6. Operating-point analysis ───────────────────────────────────────────
    op_rows = []
    for t in np.round(np.linspace(0.05, 0.95, 19), 2):
        d = compute_metrics(test_y, test_probs, float(t))
        d["threshold"] = float(t)
        op_rows.append(d)
    operating = pd.DataFrame(op_rows).set_index("threshold")
    operating.to_csv(out_dir / "operating_points.csv")

    recommended = {}
    for name, t in [("kappa_optimal", thr_kappa),
                    ("youden_optimal", thr_youden),
                    ("screening_sens95", thr_sens95)]:
        recommended[name] = {"threshold": float(t),
                             "test": compute_metrics(test_y, test_probs, float(t))}

    # ── Per-source kappa bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = per_source_img["kappa"].sort_values()
    ax.barh(ks.index.astype(str), ks.values)
    ax.axvline(0.7, color="red", ls="--", label="WHO certification (kappa=0.70)")
    ax.set_xlabel("Cohen's kappa (image level)")
    ax.set_title(f"Per-source TF classification kappa ({run_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_source_kappa.png", dpi=200)
    plt.close(fig)

    # ── Predicted vs. true prevalence scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(prevalence["true_prev_image"].max(),
              prevalence["pred_prev_image"].max()) * 1.15 + 0.01
    ax.plot([0, lim], [0, lim], "--", color="gray")
    ax.scatter(prevalence["true_prev_image"], prevalence["pred_prev_image"])
    for s, r in prevalence.iterrows():
        ax.annotate(str(s), (r["true_prev_image"], r["pred_prev_image"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("True TF prevalence")
    ax.set_ylabel("Predicted TF prevalence")
    ax.set_title(f"Per-source prevalence estimation ({run_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "prevalence_scatter.png", dpi=200)
    plt.close(fig)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "run_name": run_name,
        "ckpt": args.ckpt,
        "csv": args.csv,
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_test_subjects": int(len(subj)),
        "thresholds": {"kappa": thr_kappa, "youden": thr_youden, "sens95": thr_sens95},
        "image_level": {"metrics": overall, "ci95": overall_ci},
        "subject_level": {"metrics": subject_overall, "ci95": subject_overall_ci},
        "calibration": {"brier": overall["brier"], "ece": cal["ece"]},
        "recommended_operating_points": recommended,
        "n_boot": args.n_boot,
    }
    save_json(out_dir / "tier_a_summary.json", summary)

    # ── Console report ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"TIER A RESULTS — {run_name}")
    print("=" * 64)
    print(f"Main threshold (val kappa-optimal): {thr_kappa:.2f}")
    print("\n-- Image level (n={}) --".format(len(test_y)))
    for k in ["kappa", "auroc", "auprc", "sensitivity", "specificity",
              "precision", "f1", "accuracy", "brier"]:
        lo, hi = overall_ci.get(k, [float("nan"), float("nan")])
        print(f"  {k:12s} {overall[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  ECE          {cal['ece']:.4f}")
    print("\n-- Subject level (n={}) --".format(len(subj)))
    for k in ["kappa", "auroc", "auprc", "sensitivity", "specificity"]:
        lo, hi = subject_overall_ci.get(k, [float("nan"), float("nan")])
        print(f"  {k:12s} {subject_overall[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print("\n-- Per-source kappa (image level) --")
    for s, r in per_source_img.iterrows():
        print(f"  {str(s):40s} n={int(r['n']):5d}  prev={r['prevalence']:.3f}  "
              f"kappa={r['kappa']:.3f}  auroc={r['auroc']:.3f}")
    print("\n-- Recommended operating points (test) --")
    for name, d in recommended.items():
        t = d["test"]
        print(f"  {name:18s} thr={d['threshold']:.2f}  "
              f"sens={t['sensitivity']:.3f}  spec={t['specificity']:.3f}  "
              f"kappa={t['kappa']:.3f}")
    print("\nArtifacts written to:", out_dir)
    print("  tier_a_summary.json, per_source_image_metrics.csv,")
    print("  per_source_subject_metrics.csv, prevalence_by_source.csv,")
    print("  operating_points.csv, reliability_diagram.png,")
    print("  per_source_kappa.png, prevalence_scatter.png")


if __name__ == "__main__":
    main()
