"""Generate publication-quality figures for the trachoma ResNet paper.

Reads the analysis artifacts under eval_outputs/resnet_cv5/{tier_a,tier_a_youden}/
and (optionally, when present) eval_outputs/loso_cv5/, and produces a set of
self-contained PNG figures suitable for embedding in a Word document.

Usage:
    python generate_figs.py                          # default paths, all figures
    python generate_figs.py --out_dir figs_paper/    # custom output directory
    python generate_figs.py --only fig04 fig06       # only specific figures
    python generate_figs.py --dpi 600                # higher resolution

Figures produced (when source data is available):
    fig01_dataset_composition.png   Per-source image counts + TF prevalence
    fig02_performance_curves.png    ROC + PR + calibration (3 panels)
    fig03_confusion_and_thresholds.png  Confusion @ kappa-opt + threshold sweep
    fig04_per_source_kappa.png      Per-source kappa with WHO reference line
    fig05_prevalence_scatter.png    Predicted vs true prevalence per source
    fig06_literature_comparison.png Our kappa vs published prior-work kappa
    fig07_per_fold_stability.png    Box plot of key metrics across 5 folds
    fig08_loso_kappa.png            (LOSO) per-source held-out kappa
    fig09_loso_vs_indist.png        (LOSO) generalization gap vs in-distribution

Tables produced:
    table01a_main_results_kappa.png    5-fold CV metrics @ kappa-optimal thr
    table01b_main_results_youden.png   5-fold CV metrics @ Youden-optimal thr
    table02_per_source_cv.png          Per-source 5-fold CV image-level
    table03_prevalence_significance.png  Per-source McNemar test (kappa-opt)
    table04_literature_comparison.png    Our work vs published prior literature

LOSO figures are skipped if their inputs are missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

try:
    from adjustText import adjust_text as _adjust_text
except ImportError:
    _adjust_text = None


def _adjust_text_safe(texts, ax, x, y):
    """Iteratively reposition scatter-plot labels to avoid overlap.

    Uses jacobgil's adjustText if installed; falls back to a no-op with a
    printed warning so the figure still renders (with the pre-fix overlap).
    """
    if _adjust_text is None:
        print("[generate_figs] adjustText not installed — labels may overlap. "
              "Install with: pip install adjustText")
        return
    for t in texts:
        t.set_clip_on(False)
    _adjust_text(
        texts, ax=ax, x=x, y=y,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6, alpha=0.9,
                        shrinkA=6, shrinkB=6),
        expand=(2.4, 2.4),
        force_static=1.5,
        force_text=1.2,
        force_pull=0.15,
        max_move=200,
        iter_lim=1500,
    )


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style():
    """Publication style: serif fonts, modest sizes, tight rcParams."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# Colorblind-safe palette (Wong 2011 / Okabe-Ito)
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "skyblue": "#56B4E9",
    "black":  "#000000",
    "gray":   "#808080",
}

# Short labels for sources (the full names are long)
SOURCE_SHORT = {
    "2022 Australia Trachoma Images": "Australia (2022)",
    "CC_EA2017": "EA2017",
    "Gambia PRET 18m": "Gambia PRET",
    "ICAPS": "ICAPS",
    "Kim et al": "Kim et al",
    "SOCIT": "SOCIT",
    "Solomon Islands research study 2015": "Solomon Is.",
    "TANA II study,  Ethiopia, Goncha Siso Enesie woreda, Amhara Region, Nov 2011": "TANA II (Ethiopia)",
}

WHO_KAPPA = 0.70


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def build_paths(args):
    root = Path(args.eval_root)
    paths = {
        "root": root,
        "tier_a": root / "tier_a",
        "tier_a_youden": root / "tier_a_youden",
        "loso_root": Path(args.loso_eval_root),
        "out": Path(args.out_dir),
    }
    paths["out"].mkdir(parents=True, exist_ok=True)
    return paths


def load_cv_summary(paths):
    """Load the cv_summary.json files (kappa + youden) if available."""
    out = {}
    for k, p in [("kappa", paths["tier_a"] / "cv_summary.json"),
                 ("youden", paths["tier_a_youden"] / "cv_summary.json")]:
        if p.exists():
            with p.open() as f:
                out[k] = json.load(f)
    return out


def load_test_predictions(paths):
    """Load test_probs_by_fold + ensemble + index labels."""
    p = paths["tier_a"]
    probs_by_fold = np.load(p / "test_probs_by_fold.npy")  # (n_folds, n_test)
    probs_ens = np.load(p / "test_probs_ensemble.npy")     # (n_test,)
    idx = pd.read_csv(p / "test_index.csv")                # image_path, label
    return probs_by_fold, probs_ens, idx


def load_per_source(paths):
    p = paths["tier_a"]
    return {
        "img_ens": pd.read_csv(p / "per_source_image_metrics_ensemble.csv"),
        "img_per_fold_summary": pd.read_csv(p / "per_source_image_metrics_per_fold_summary.csv"),
        "subj_ens": pd.read_csv(p / "per_source_subject_metrics_ensemble.csv"),
        "prev": pd.read_csv(p / "prevalence_by_source_ensemble.csv"),
        "ops": pd.read_csv(p / "operating_points_ensemble.csv"),
        "per_fold_overall_image": pd.read_csv(p / "per_fold_overall_image.csv"),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig01_dataset_composition(paths, src):
    """Per-source image counts + TF prevalence."""
    df = src["img_ens"].copy()
    df["short"] = df["source"].map(SOURCE_SHORT).fillna(df["source"])
    df = df.sort_values("n", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: image counts (log scale because of large dynamic range)
    ax1.barh(df["short"], df["n"], color=PALETTE["blue"], alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of test images (log scale)")
    ax1.set_title("(a) Held-out test set size by source")
    for y, (n, p) in enumerate(zip(df["n"], df["n_pos"])):
        ax1.text(n * 1.08, y, f"{n:,} ({p}+)", va="center", fontsize=8)

    # Panel B: TF prevalence
    ax2.barh(df["short"], df["prevalence"] * 100,
             color=PALETTE["orange"], alpha=0.85)
    overall_prev = (df["n_pos"].sum() / df["n"].sum()) * 100
    ax2.axvline(overall_prev, color=PALETTE["red"], linestyle="--",
                linewidth=1.2, label=f"Overall ({overall_prev:.1f}%)")
    ax2.set_xlabel("TF prevalence (%)")
    ax2.set_title("(b) TF prevalence by source")
    ax2.legend(loc="upper right")
    for y, p in enumerate(df["prevalence"]):
        ax2.text(p * 100 + 0.7, y, f"{p*100:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    out = paths["out"] / "fig01_dataset_composition.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig02_performance_curves(paths, probs_by_fold, probs_ens, idx, summary):
    """ROC + PR + calibration (3-panel)."""
    y = idx["label"].to_numpy().astype(int)
    n_folds = probs_by_fold.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))

    # ----- (a) ROC -----
    ax = axes[0]
    # per-fold (light)
    for f in range(n_folds):
        fpr, tpr, _ = roc_curve(y, probs_by_fold[f])
        ax.plot(fpr, tpr, color=PALETTE["gray"], alpha=0.45, linewidth=0.9,
                label="Per-fold" if f == 0 else None)
    # ensemble (bold)
    fpr_e, tpr_e, _ = roc_curve(y, probs_ens)
    auroc_e = auc(fpr_e, tpr_e)
    ax.plot(fpr_e, tpr_e, color=PALETTE["blue"], linewidth=2.2,
            label=f"Ensemble (AUROC = {auroc_e:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["black"], linestyle=":", linewidth=0.8)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("(a) ROC curve")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.legend(loc="lower right")

    # ----- (b) PR -----
    ax = axes[1]
    for f in range(n_folds):
        pr, rc, _ = precision_recall_curve(y, probs_by_fold[f])
        ax.plot(rc, pr, color=PALETTE["gray"], alpha=0.45, linewidth=0.9,
                label="Per-fold" if f == 0 else None)
    pr_e, rc_e, _ = precision_recall_curve(y, probs_ens)
    auprc_e = average_precision_score(y, probs_ens)
    ax.plot(rc_e, pr_e, color=PALETTE["orange"], linewidth=2.2,
            label=f"Ensemble (AUPRC = {auprc_e:.3f})")
    prev = y.mean()
    ax.axhline(prev, color=PALETTE["black"], linestyle=":", linewidth=0.8,
               label=f"No-skill ({prev:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision–recall curve")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.legend(loc="lower left")


    fig.tight_layout()
    out = paths["out"] / "fig02_performance_curves.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig03_confusion_and_thresholds(paths, probs_ens, idx, src, summary):
    """Confusion matrix at kappa-optimal + threshold sweep."""
    y = idx["label"].to_numpy().astype(int)
    thr_k = summary["kappa"]["recommended_operating_points_ensemble"]["kappa_optimal"]["threshold"]
    thr_y = summary["kappa"]["recommended_operating_points_ensemble"]["youden_optimal"]["threshold"]

    fig = plt.figure(figsize=(11.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])

    # ----- (a) Confusion matrix at kappa-optimal threshold -----
    ax = fig.add_subplot(gs[0, 0])
    pred = (probs_ens >= thr_k).astype(int)
    cm = confusion_matrix(y, pred)
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    n = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / n * 100
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. TF−", "Pred. TF+"])
    ax.set_yticklabels(["True TF−", "True TF+"])
    ax.set_title(f"(a) Confusion matrix @ κ-optimal threshold ({thr_k:.2f})")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ----- (b) Threshold sweep on pooled OOF validation predictions -----
    # Pool the 5 folds' held-out val predictions (each sample appears exactly
    # once, always predicted by a model that did not see it in training).
    # Sweeping this stack is exactly how the OOF thresholds were selected,
    # so the dashed lines will sit at (or immediately next to) the peaks.
    ax = fig.add_subplot(gs[0, 1])
    val_frames = []
    for k in range(5):
        vp = pd.read_csv(Path("runs/resnet_cv5") / f"fold_{k}" / "val_predictions.csv")
        val_frames.append(vp[["label", "prob"]])
    oof = pd.concat(val_frames, ignore_index=True)
    y_val = oof["label"].to_numpy().astype(int)
    p_val = oof["prob"].to_numpy()

    ths = np.arange(0.05, 1.0, 0.05)
    kappa_v = np.empty_like(ths); sens_v = np.empty_like(ths)
    spec_v = np.empty_like(ths); f1_v = np.empty_like(ths)
    for i, t in enumerate(ths):
        pred = (p_val >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        sens_v[i] = tp / (tp + fn) if (tp + fn) else 0.0
        spec_v[i] = tn / (tn + fp) if (tn + fp) else 0.0
        kappa_v[i] = cohen_kappa_score(y_val, pred)
        f1_v[i] = f1_score(y_val, pred, zero_division=0)

    ax.plot(ths, kappa_v, color=PALETTE["blue"],
            linewidth=2, label="Cohen's κ", marker="o", markersize=3)
    ax.plot(ths, sens_v, color=PALETTE["green"],
            linewidth=1.5, label="Sensitivity")
    ax.plot(ths, spec_v, color=PALETTE["red"],
            linewidth=1.5, label="Specificity")
    ax.plot(ths, f1_v, color=PALETTE["purple"],
            linewidth=1.5, label="F1")
    ax.axhline(WHO_KAPPA, color=PALETTE["black"], linestyle=":",
               linewidth=0.9, alpha=0.6)
    ax.text(0.97, WHO_KAPPA + 0.012, f"WHO κ ≥ {WHO_KAPPA}", fontsize=8,
            ha="right", va="bottom", color=PALETTE["black"])
    # operating points (selected on this same OOF stack)
    for thr, name, color in [
        (thr_y, "Youden", PALETTE["skyblue"]),
        (thr_k, "κ-opt", PALETTE["red"]),
    ]:
        ax.axvline(thr, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(thr + 0.005, 0.02, name, rotation=90, fontsize=8,
                color=color, va="bottom")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("(b) Pooled OOF validation metrics vs threshold (image level)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", ncol=2)

    fig.tight_layout()
    out = paths["out"] / "fig03_confusion_and_thresholds.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig04_per_source_kappa(paths, src):
    """Per-source ensemble kappa with WHO reference line."""
    img = src["img_ens"].set_index("source")
    df = pd.DataFrame({
        "short": [SOURCE_SHORT.get(s, s) for s in img.index],
        "n": img["n"],
        "kappa_ens": img["kappa"],
    }).sort_values("kappa_ens", ascending=True)

    def kappa_color(k):
        if k >= 0.80:
            return PALETTE["green"]
        if k >= WHO_KAPPA:
            return PALETTE["skyblue"]
        return PALETTE["red"]

    colors = [kappa_color(k) for k in df["kappa_ens"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["kappa_ens"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.5)

    who = ax.axvline(WHO_KAPPA, color=PALETTE["black"], linestyle="--",
                     linewidth=1.2, alpha=0.8, label=f"WHO κ ≥ {WHO_KAPPA}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["short"])
    ax.set_xlabel("Cohen's κ (image level)")
    ax.set_xlim(0, 1.18)  # leave room for annotations

    # annotate n at fixed right margin
    for y, n in enumerate(df["n"]):
        ax.text(1.165, y, f"n={n:,}", va="center", ha="right", fontsize=8,
                color="dimgray")

    legend_elements = [
        Patch(facecolor=PALETTE["green"], edgecolor="black", linewidth=0.5,
              label="κ ≥ 0.80 (excellent)"),
        Patch(facecolor=PALETTE["skyblue"], edgecolor="black", linewidth=0.5,
              label="0.70 ≤ κ < 0.80 (WHO-pass)"),
        Patch(facecolor=PALETTE["red"], edgecolor="black", linewidth=0.5,
              label="κ < 0.70"),
        who,
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=4, framealpha=0.95,
              edgecolor="lightgray", fontsize=8.5).set_frame_on(True)

    fig.tight_layout()
    out = paths["out"] / "fig04_per_source_kappa.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig05_prevalence_scatter(paths, src):
    """Predicted vs true prevalence scatter at kappa-optimal threshold."""
    prev = src["prev"].copy()
    prev["short"] = prev["source"].map(SOURCE_SHORT).fillna(prev["source"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))

    for ax, level, n_col, true_col, pred_col, title in [
        (ax1, "image", "n_images", "true_prev_image", "pred_prev_image",
         "(a) Image-level prevalence"),
        (ax2, "eye", "n_subjects", "true_prev_subject", "pred_prev_subject",
         "(b) Eye-level prevalence"),
    ]:
        data_max = max(prev[true_col].max(), prev[pred_col].max())
        lim = data_max * 1.15 + 0.02
        margin = lim * 0.05  # allow labels to overshoot the axes a hair
        ax.plot([0, lim], [0, lim], color=PALETTE["black"], linestyle=":",
                linewidth=1.0, label="Perfect estimation")
        # size by log(n)
        sizes = np.log10(prev[n_col].clip(lower=1)) * 60 + 30
        ax.scatter(prev[true_col], prev[pred_col], s=sizes,
                   color=PALETTE["blue"], alpha=0.7,
                   edgecolor="black", linewidth=0.6)
        # Default is upper-right; override per (panel, source) to avoid overlaps.
        _default = ((5, 5), "left", "bottom")
        _up_left = ((-5, 5), "right", "bottom")
        _down_right = ((5, -5), "left", "top")
        label_placement = {
            ("image", "Solomon Is."): _up_left,
            ("image", "TANA II (Ethiopia)"): _up_left,
            ("image", "EA2017"): _down_right,
            ("eye", "Solomon Is."): _up_left,
            ("eye", "EA2017"): _down_right,
        }
        for _, row in prev.iterrows():
            xytext, ha, va = label_placement.get(
                (level, row["short"]), _default)
            ax.annotate(row["short"],
                        (row[true_col], row[pred_col]),
                        fontsize=8, xytext=xytext,
                        textcoords="offset points",
                        ha=ha, va=va)
        ax.set_xlabel(f"True TF prevalence ({level} level)")
        ax.set_ylabel(f"Predicted TF prevalence ({level} level)")
        ax.set_title(title)
        ax.set_xlim(-margin, lim)
        ax.set_ylim(-margin, lim)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")

    fig.tight_layout()
    out = paths["out"] / "fig05_prevalence_scatter.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig06_literature_comparison(paths, summary):
    """Our ensemble κ vs published prior-work κ.

    Prior numbers from existing_trachoma_ml_papers/lit_review_summary.md:
      Kim 2019:    TF ensemble κ = 0.44 [0.26, 0.62], n_test=100, 2 countries, image-level split
      Joye 2024:   AI alone κ = 0.634 (combined dataset, no CI)
      Joye 2024:   AI + human overread κ = 0.787 (combined dataset, no CI)
      Kulohoma 2026: κ = 0.71 [0.58, 0.84], n_val=62, 7 countries
      (Yazbeck multiregion 2026 reports F1, not κ — excluded.)

    Reading: our headline is the IMAGE-level ensemble at the κ-optimal
    threshold; we also report SUBJECT-level for parity with Joye/Kulohoma.
    """
    ours_img = summary["kappa"]["ensemble_image_level"]
    ours_subj = summary["kappa"]["ensemble_subject_level"]

    rows = [
        # name, k, lo, hi, n, geo, color, has_ci
        ("Kim 2019\n(ensemble, image)", 0.44, 0.26, 0.62,
         100, "2 countries", PALETTE["gray"], True),
        ("Joye 2024\n(AI alone, combined)", 0.634, None, None,
         None, "Single-region (ICAPS+Kim)", PALETTE["gray"], False),
        ("Joye 2024\n(AI + overread)", 0.787, None, None,
         None, "Single-region (ICAPS+Kim)", PALETTE["gray"], False),
        ("Kulohoma 2026\n(subject, single-split)", 0.71, 0.58, 0.84,
         62, "7 countries (n=62 val)", PALETTE["gray"], True),
        ("This work\n(image, ensemble)",
         ours_img["metrics"]["kappa"],
         ours_img["ci95"]["kappa"][0], ours_img["ci95"]["kappa"][1],
         4631, "8 sources", PALETTE["blue"], True),
        ("This work\n(eye, ensemble)",
         ours_subj["metrics"]["kappa"],
         ours_subj["ci95"]["kappa"][0], ours_subj["ci95"]["kappa"][1],
         3021, "8 sources", PALETTE["blue"], True),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 5))
    y_pos = np.arange(len(rows))
    # Right-side annotation column starts here
    annot_x = 1.02
    for y, (name, k, lo, hi, n, geo, color, has_ci) in enumerate(rows):
        ax.barh(y, k, color=color, alpha=0.85,
                edgecolor="black", linewidth=0.5)
        if has_ci:
            ax.errorbar(k, y, xerr=[[k - lo], [hi - k]],
                        fmt="none", color="black", capsize=4, linewidth=1.2)
        # primary label: kappa + CI
        if has_ci:
            text = f"κ = {k:.3f} [{lo:.2f}, {hi:.2f}]"
        else:
            text = f"κ = {k:.3f}"
        if n is not None:
            text += f"\nn_test = {n:,}"
        ax.text(annot_x, y, text, va="center", ha="left", fontsize=8.5)

    ax.axvline(WHO_KAPPA, color=PALETTE["red"], linestyle="--",
               linewidth=1.3, alpha=0.8, label=f"WHO κ ≥ {WHO_KAPPA}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Cohen's κ")
    ax.set_xlim(0, 1.32)  # leave room for right-side annotations
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = paths["out"] / "fig06_literature_comparison.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig07_per_fold_stability(paths, src):
    """Box / strip plot of key metrics across the 5 folds."""
    df = src["per_fold_overall_image"].copy()
    metrics = ["kappa", "auroc", "auprc", "sensitivity",
               "specificity", "f1"]
    labels = ["κ", "AUROC", "AUPRC", "Sens.", "Spec.", "F1"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = [df[m].values for m in metrics]
    bp = ax.boxplot(values, tick_labels=labels, patch_artist=True,
                    widths=0.45, showmeans=True,
                    meanprops=dict(marker="D",
                                   markerfacecolor=PALETTE["red"],
                                   markeredgecolor=PALETTE["red"],
                                   markersize=5))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["skyblue"])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)

    # overlay individual fold points
    rng = np.random.default_rng(0)
    for i, vals in enumerate(values, start=1):
        x = rng.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, color="black", s=18, alpha=0.7, zorder=3)

    ax.axhline(WHO_KAPPA, color=PALETTE["red"], linestyle=":",
               linewidth=0.9, alpha=0.5)
    ax.text(len(metrics) + 0.4, WHO_KAPPA, f"WHO κ = {WHO_KAPPA}",
            fontsize=8, va="center", color=PALETTE["red"])
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.5, 1.02)
    fig.tight_layout()
    out = paths["out"] / "fig07_per_fold_stability.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig08_loso_kappa(paths, summary):
    """Per-source held-out kappa from LOSO-CV, if available."""
    csv = paths["loso_root"] / "loso_cv_summary_kappa.csv"
    if not csv.exists():
        print(f"[generate_figs] skipping fig08: {csv} not found (LOSO not done?)")
        return None
    df = pd.read_csv(csv)
    df["short"] = df["held_out_source"].map(SOURCE_SHORT).fillna(
        df["held_out_source"])
    df = df.sort_values("kappa", ascending=True)

    def kappa_color(k):
        if k >= 0.80:
            return PALETTE["green"]
        if k >= WHO_KAPPA:
            return PALETTE["skyblue"]
        return PALETTE["red"]
    colors = [kappa_color(k) for k in df["kappa"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["kappa"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.errorbar(df["kappa"], y_pos,
                xerr=[df["kappa"] - df["kappa_lo"],
                      df["kappa_hi"] - df["kappa"]],
                fmt="none", color="black", capsize=3, linewidth=1.0)
    ax.axvline(WHO_KAPPA, color=PALETTE["black"], linestyle="--",
               linewidth=1.2, alpha=0.8, label=f"WHO κ ≥ {WHO_KAPPA}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["short"])
    ax.set_xlabel("Cohen's κ on held-out source (LOSO ensemble, κ-optimal)")
    ax.set_xlim(0, 1.12)
    # annotate n and AUROC at the right edge
    for y, (n, auc_, prev) in enumerate(zip(df["n_test"], df["auroc"],
                                            df["test_prevalence"])):
        ax.text(1.10, y, f"n={int(n):,}\nAUROC={auc_:.3f}",
                va="center", ha="right", fontsize=7, color="dimgray")

    legend_elements = [
        Patch(facecolor=PALETTE["green"], edgecolor="black", linewidth=0.5,
              label="κ ≥ 0.80"),
        Patch(facecolor=PALETTE["skyblue"], edgecolor="black", linewidth=0.5,
              label="0.70 ≤ κ < 0.80"),
        Patch(facecolor=PALETTE["red"], edgecolor="black", linewidth=0.5,
              label="κ < 0.70"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=4, framealpha=0.95,
              edgecolor="lightgray", fontsize=8.5).set_frame_on(True)

    fig.tight_layout()
    out = paths["out"] / "fig08_loso_kappa.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def fig09_loso_vs_indist(paths, src):
    """Per-source LOSO κ vs in-distribution κ (generalization gap)."""
    csv = paths["loso_root"] / "loso_cv_summary_kappa.csv"
    if not csv.exists():
        print(f"[generate_figs] skipping fig09: {csv} not found (LOSO not done?)")
        return None
    loso = pd.read_csv(csv)
    loso["short"] = loso["held_out_source"].map(SOURCE_SHORT).fillna(
        loso["held_out_source"])
    # The CSV already contains indist_kappa and delta_kappa columns
    loso = loso.sort_values("indist_kappa", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(loso))
    width = 0.4
    ax.bar(x - width/2, loso["indist_kappa"], width,
           label="In-distribution (5-fold CV ensemble)",
           color=PALETTE["blue"], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, loso["kappa"], width,
           label="LOSO (source held out)",
           color=PALETTE["orange"], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(WHO_KAPPA, color=PALETTE["black"], linestyle="--",
               linewidth=1.0, alpha=0.7, label=f"WHO κ ≥ {WHO_KAPPA}")
    ax.set_xticks(x)
    ax.set_xticklabels(loso["short"], rotation=30, ha="right")
    ax.set_ylabel("Cohen's κ (image level)")
    # annotate Δκ on top of each pair
    for i, (in_k, lo_k, d) in enumerate(zip(loso["indist_kappa"], loso["kappa"],
                                            loso["delta_kappa_loso_minus_indist"])):
        y = max(in_k, lo_k) + 0.02
        ax.text(i, y, f"Δκ={d:+.2f}", ha="center", va="bottom", fontsize=8,
                color=PALETTE["red"] if d < -0.05 else "black")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = paths["out"] / "fig09_loso_vs_indist.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def table_s1_dataset_composition(paths, summary=None, src=None):
    """Supplementary Table — per-source dataset composition.

    Numbers verified against /home/Trachoma/data/all_metadata.csv;
    text fields from the user's Methods prose and lit-review attribution.
    Em-dash = field still needs to be filled in from the source publication.
    """
    _td_consensus = ("10 expert photograders "
                     "(consensus method not specified)")
    rows = [
        # Dataset, Country/Region, Year(s), Eyes, Images, Camera, Consensus
        ["SOCIT [18]",
         "Tanzania",
         "2006",
         "6,809", "11,736",
         "Nikon Coolpix E990",
         "1 expert photo grader, confirmed by 1 "
         "international expert photo grader"],
        ["EA2017 [20, 21]",
         "East Amhara, Ethiopia",
         "2017",
         "2,476", "4,950",
         "Canon EOS 60D",
         "2 graders, with a third as the tie-breaker "
         "when needed"],
        ["ICAPS [22]",
         "Chamwino, Tanzania",
         "~2019–20",
         "2,299", "2,299",
         "Samsung Galaxy S8 + Gear VR mount",
         "1 international expert photo grader and 1 "
         "field grader (gradable images with "
         "concordant grades included)"],
        ["Kim et al. (UCSF) [10]",
         "Ethiopia (TANA) + Niger (PRET)",
         "2006–2013",
         "1,656", "1,656",
         "Nikon D40x / D60 (PRET); "
         "Nikon D70 / D80 / D90 / D100 (TANA)",
         "3 graders (consensus method not specified)"],
        ["Tropical Data — TANA II [19]",
         "Goncha Siso Enesie woreda, Amhara, Ethiopia",
         "2011",
         "410", "804",
         "iPhone 4s + CellScope adapter (half); "
         "Nikon D70 / D90 (half)",
         _td_consensus],
        ["Tropical Data — Solomon Islands [19]",
         "Solomon Islands",
         "2015",
         "1,286", "1,475",
         "Nikon D3000",
         _td_consensus],
        ["Tropical Data — Australia [19]",
         "Australia",
         "2022",
         "117", "134",
         "Not specified",
         _td_consensus],
        ["Tropical Data — Gambia PRET 18m [19]",
         "The Gambia",
         "~2012",
         "50", "50",
         "Nikon D40x / D60",
         _td_consensus],
        ["__SECTION__Total"],
        ["5 datasets / 6 countries", "—", "—",
         "15,103", "23,104", "—", "—"],
    ]
    columns = ["Dataset (source)", "Country / Region", "Year(s)",
               "Eyes (n)", "Images (n)", "Camera", "Consensus details"]
    footnotes = [
        "EA2017 images were graded at the Gondar Grading Center "
        "(Ethiopia).",
        "Tropical Data is a curation by the International Coalition for "
        "Trachoma Control; its four constituent sub-studies (TANA II "
        "Ethiopia, Solomon Islands 2015, 2022 Australia, Gambia PRET "
        "18-month follow-up) are listed individually for transparency.",
        "Overall TF prevalence across the merged dataset: 9.81% (2,267 "
        "TF-positive images of 23,104 total).",
    ]
    return render_table(
        paths["out"] / "table_s1_dataset_composition.png",
        title="Table 1. Dataset composition across the "
              "five constituent datasets (eight underlying source studies).",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[2.2, 3.2, 0.9, 0.9, 0.9, 2.1, 3.0],
        alignments=["left", "left", "center",
                    "right", "right", "left", "left"],
        dpi=args.dpi,
    )


def _sanitize_source(name):
    """Match the sanitization used by loso_cv_train.py for directory names."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def table06_loso_prevalence(paths, summary):
    """Per-source LOSO predicted-vs-true prevalence with McNemar tests."""
    csv = paths["loso_root"] / "loso_cv_summary_kappa.csv"
    if not csv.exists():
        print(f"[generate_figs] skipping table06: {csv} not found")
        return None
    summary_df = pd.read_csv(csv)

    # Path to per-source ensemble predictions
    runs_root = Path("runs/loso_cv5")

    rows = []
    for _, r in summary_df.iterrows():
        src_name = r["held_out_source"]
        sanitized = _sanitize_source(src_name)
        pred_csv = runs_root / sanitized / "ensemble_predictions_kappa.csv"
        if not pred_csv.exists():
            print(f"[generate_figs] table06: missing {pred_csv}, skipping {src_name}")
            continue
        ep = pd.read_csv(pred_csv)
        thr = float(r["ensemble_threshold"])
        probs = ep["prob_ensemble"].to_numpy()
        y = ep["label"].to_numpy().astype(int)
        tp, pp, dd, p_exact, n = _mcnemar_for_table(probs, y, thr)
        rows.append([
            SOURCE_SHORT.get(src_name, src_name),
            f"{n:,}",
            f"{tp*100:.2f}%",
            f"{pp*100:.2f}%",
            f"{dd*100:+.2f}",
            f"{thr:.2f}",
            _fmt_p(p_exact),
            _sig_code(p_exact),
        ])

    columns = ["Held-out source", "n images", "True prev.", "Predicted prev.",
               "Δ (pp)", "Thr.", "McNemar p", "Sig."]
    footnotes = [
        "McNemar's exact (two-sided binomial) test for marginal homogeneity "
        "on per-image paired predictions vs. ground-truth labels.",
        "Threshold (Thr.) is the per-source OOF-swept κ-optimal threshold "
        "(differs per source because each LOSO model is trained on a "
        "different 7-source pool).",
        "Δ = predicted − true prevalence in percentage points. Negative Δ "
        "indicates the model underestimates prevalence on the held-out "
        "source; positive indicates overestimation.",
        "Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. p ≥ 0.05.",
        "Three sources are non-significant under McNemar: ICAPS "
        "(Δ = −0.13 pp; the model genuinely matches true prevalence) "
        "and Gambia PRET 18m (Δ = −4.00 pp) and 2022 Australia "
        "(Δ = −5.97 pp), whose tests are underpowered by very small "
        "test-set sizes (n = 50 and 134 respectively). The remaining "
        "five sources show statistically detectable prevalence bias.",
    ]
    return render_table(
        paths["out"] / "table06_loso_prevalence.png",
        title="Table 7. Predicted vs true TF prevalence on LOSO held-out "
              "sources (κ-optimal threshold).",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[1.7, 0.85, 0.95, 1.05, 0.85, 0.6, 1.0, 0.55],
        dpi=args.dpi,
    )


def table05_loso_per_source(paths, summary):
    """Per-source LOSO breakdown at κ-optimal threshold."""
    csv = paths["loso_root"] / "loso_cv_summary_kappa.csv"
    pfstd_csv = paths["loso_root"] / "loso_cv_summary_per_source_std_kappa.csv"
    if not csv.exists():
        print(f"[generate_figs] skipping table05: {csv} not found (LOSO not done?)")
        return None
    df = pd.read_csv(csv)
    if pfstd_csv.exists():
        pf = pd.read_csv(pfstd_csv).rename(columns={"source": "held_out_source"})
        df = df.merge(pf[["held_out_source", "kappa_mean", "kappa_std"]],
                      on="held_out_source", how="left")
    else:
        df["kappa_mean"] = np.nan
        df["kappa_std"] = np.nan

    df = df.sort_values("kappa", ascending=False)

    rows = []
    for _, r in df.iterrows():
        src = SOURCE_SHORT.get(r["held_out_source"], r["held_out_source"])
        rows.append([
            src,
            f"{int(r['n_test']):,}",
            f"{r['test_prevalence']*100:.2f}%",
            _val_ci(r["kappa"], r["kappa_lo"], r["kappa_hi"]),
            _mean_std(r.get("kappa_mean"), r.get("kappa_std")),
            _val_ci(r["auroc"], r["auroc_lo"], r["auroc_hi"]),
            _fmt(r["sensitivity"]),
            _fmt(r["specificity"]),
            _fmt(r["indist_kappa"]),
            f"{r['delta_kappa_loso_minus_indist']:+.3f}",
        ])

    columns = ["Held-out source", "n", "TF prev.",
               "κ (LOSO ens.) [95% CI]", "κ (per-fold mean ± SD)",
               "AUROC (LOSO) [95% CI]",
               "Sens.", "Spec.",
               "κ (in-dist)", "Δκ"]
    footnotes = [
        "All metrics on the held-out source (entire source withheld from "
        "training; predictions are the arithmetic mean of 5 fold models "
        "trained on the other 7 sources). κ-optimal threshold (OOF-swept "
        "on the LOSO validation stack).",
        "κ per-fold is mean ± SD across the 5 LOSO folds, each at its own "
        "validation-selected threshold.",
        "Δκ = LOSO κ − in-distribution κ from the 5-fold CV ensemble on the "
        "same source's 20% test slice (Table 2). ",
    ]
    return render_table(
        paths["out"] / "table05_loso_per_source.png",
        title="Table 6. Per-source LOSO-CV performance (κ-optimal threshold).",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[1.7, 0.7, 0.85, 1.7, 1.8, 1.7, 0.7, 0.7, 0.9, 0.7],
        dpi=args.dpi,
    )


# ---------------------------------------------------------------------------
# Booktabs-style table renderer
# ---------------------------------------------------------------------------

def render_table(out_path, title, columns, rows, footnotes=None,
                 col_widths=None, alignments=None, dpi=300,
                 footnote_fontsize=8, header_fontsize=9.5,
                 cell_fontsize=9, row_height=0.32,
                 header_height=0.45, title_height=0.5):
    """Render a simple booktabs-style table (top/mid/bottom horizontal rules).

    Parameters
    ----------
    columns : list[str | tuple[str, list[str]]]
        Either flat column headers, OR a list of (group_header, [sub1, sub2])
        tuples for two-level headers. The two formats can be mixed; a plain
        string is treated as a single-cell group with itself as the only sub.
    rows : list[list[str]]
        Each row's cells. Cells starting with '__SECTION__' mark a section
        divider with the remaining text used as the bold heading row.
    """
    # Flatten columns to leaf headers, keep group structure for top-row
    leaves = []
    groups = []  # list of (group_header, span_start_idx, span_end_idx_exclusive)
    for c in columns:
        if isinstance(c, tuple):
            group_label, subs = c
            start = len(leaves)
            leaves.extend(subs)
            groups.append((group_label, start, len(leaves)))
        else:
            start = len(leaves)
            leaves.append(c)
            groups.append((None, start, len(leaves)))

    n_cols = len(leaves)
    if col_widths is None:
        col_widths = [1.0] * n_cols
    if alignments is None:
        alignments = ["right"] * n_cols
        alignments[0] = "left"

    total_w = sum(col_widths)
    # Figure width: scale total column widths so that the average column is
    # roughly 1.5 inches.  This keeps long names readable while not blowing up.
    fig_w = total_w * 1.5 + 0.5  # +0.5" margin
    has_groups = any(g[0] is not None for g in groups)
    n_header_rows = 2 if has_groups else 1

    # Pre-wrap each body cell to fit its column width, and compute per-row
    # heights so cells that need multiple lines don't overflow.
    import textwrap
    char_w = cell_fontsize * 0.0065  # ~inches per character at cell_fontsize
    wrapped_rows = []
    row_line_counts = []
    for row in rows:
        if row and isinstance(row[0], str) and row[0].startswith("__SECTION__"):
            wrapped_rows.append(row)
            row_line_counts.append(1)
            continue
        wrapped = []
        max_lines = 1
        for i, cell in enumerate(row):
            text = str(cell)
            max_chars = max(6, int((col_widths[i] - 0.20) / char_w))
            if len(text) <= max_chars:
                wrapped.append(text)
                continue
            parts = textwrap.wrap(text, width=max_chars,
                                  break_long_words=False,
                                  break_on_hyphens=True)
            joined = "\n".join(parts) if parts else text
            wrapped.append(joined)
            max_lines = max(max_lines, joined.count("\n") + 1)
        wrapped_rows.append(wrapped)
        row_line_counts.append(max_lines)

    n_rows = len(rows)
    body_h = sum(row_height * (0.55 + 0.45 * lc) if lc > 1 else row_height
                 for lc in row_line_counts)
    fig_h = (title_height
             + n_header_rows * header_height
             + body_h
             + (0.3 + 0.22 * (len(footnotes) if footnotes else 0)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Column x-positions (left edges) and centers
    edges = np.concatenate([[0], np.cumsum(col_widths)])

    def cell_x(col_idx, align):
        left = edges[col_idx]
        right = edges[col_idx + 1]
        pad = 0.08
        if align == "left":
            return left + pad, "left"
        if align == "right":
            return right - pad, "right"
        return (left + right) / 2, "center"

    # Vertical layout: top to bottom
    y_cursor = fig_h - 0.05
    # Title
    ax.text(total_w / 2, y_cursor, title, ha="center", va="top",
            fontsize=header_fontsize + 1.5, fontweight="bold")
    y_cursor -= title_height

    # Top rule
    ax.hlines(y_cursor, 0, total_w, color="black", linewidth=1.4)
    y_cursor -= 0.05

    # Header rows
    if has_groups:
        # Group row
        y_grp = y_cursor - header_height * 0.45
        for label, s, e in groups:
            if label is None:
                continue
            center = (edges[s] + edges[e]) / 2
            ax.text(center, y_grp, label, ha="center", va="center",
                    fontsize=header_fontsize, fontweight="bold")
        # Mini rules under spanning groups
        y_rule = y_cursor - header_height * 0.85
        for label, s, e in groups:
            if label is None or (e - s) <= 1:
                continue
            ax.hlines(y_rule, edges[s] + 0.05, edges[e] - 0.05,
                      color="black", linewidth=0.6)
        y_cursor -= header_height
        # Leaf headers
        for i, leaf in enumerate(leaves):
            x, ha = cell_x(i, alignments[i])
            ax.text(x, y_cursor - header_height * 0.45, leaf,
                    ha=ha, va="center", fontsize=header_fontsize,
                    fontweight="bold")
        y_cursor -= header_height
    else:
        # Single header row
        for i, leaf in enumerate(leaves):
            x, ha = cell_x(i, alignments[i])
            ax.text(x, y_cursor - header_height * 0.45, leaf,
                    ha=ha, va="center", fontsize=header_fontsize,
                    fontweight="bold")
        y_cursor -= header_height

    # Mid rule
    ax.hlines(y_cursor, 0, total_w, color="black", linewidth=0.8)
    y_cursor -= 0.05

    # Body rows (using pre-wrapped cells and per-row heights). Cells are
    # top-aligned so a row's cells all start on the same baseline, and
    # multi-line cells simply continue downward beneath that baseline.
    top_pad = row_height * 0.30
    for row, lc in zip(wrapped_rows, row_line_counts):
        rh = row_height * (0.55 + 0.45 * lc) if lc > 1 else row_height
        if row and isinstance(row[0], str) and row[0].startswith("__SECTION__"):
            label = row[0].replace("__SECTION__", "").strip()
            ax.text(edges[0] + 0.08, y_cursor - top_pad, label,
                    ha="left", va="top", fontsize=cell_fontsize,
                    fontweight="bold", fontstyle="italic")
            y_cursor -= rh
            continue
        for i, cell in enumerate(row):
            x, ha = cell_x(i, alignments[i])
            ax.text(x, y_cursor - top_pad, str(cell),
                    ha=ha, va="top", fontsize=cell_fontsize)
        y_cursor -= rh

    # Bottom rule
    ax.hlines(y_cursor, 0, total_w, color="black", linewidth=1.4)
    y_cursor -= 0.1

    # Footnotes
    if footnotes:
        for fn in footnotes:
            ax.text(0.05, y_cursor - footnote_fontsize / 100,
                    fn, ha="left", va="top", fontsize=footnote_fontsize,
                    color="black")
            y_cursor -= 0.22

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.1, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fmt(v, n=3):
    """Format a number for table cells, with NaN handling."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{n}f}"


def _fmt_ci(lo, hi, n=3):
    return f"[{_fmt(lo, n)}, {_fmt(hi, n)}]"


def _val_ci(v, lo, hi, n=3):
    return f"{_fmt(v, n)} {_fmt_ci(lo, hi, n)}"


def _mean_std(mean, std, n=3):
    return f"{_fmt(mean, n)} ± {_fmt(std, n)}"


def _table01_one_threshold(paths, summary_obj, pf_summary, label, thr,
                           letter, out_name):
    """Render a single-column-friendly results table for one threshold."""
    def row(metric_label, k):
        em = summary_obj["ensemble_image_level"]["metrics"][k]
        elo, ehi = summary_obj["ensemble_image_level"]["ci95"].get(
            k, (None, None))
        sm = summary_obj["ensemble_subject_level"]["metrics"][k]
        slo, shi = summary_obj["ensemble_subject_level"]["ci95"].get(
            k, (None, None))
        nd = 3 if k != "brier" else 4
        return [
            metric_label,
            _val_ci(em, elo, ehi, nd),
            _val_ci(sm, slo, shi, nd),
        ]

    rows = [
        row("Cohen's κ",        "kappa"),
        row("AUROC",            "auroc"),
        row("AUPRC",            "auprc"),
        row("Sensitivity",      "sensitivity"),
        row("Specificity",      "specificity"),
        row("Precision",        "precision"),
        row("F1 score",         "f1"),
        row("Accuracy",         "accuracy"),
        row("Brier score",      "brier"),
    ]
    columns = [
        "Metric",
        "Image level [95% CI]",
        "Eye level [95% CI]",
    ]
    footnotes = [
        f"Held-out test set: n = {summary_obj['n_test']:,} images "
        f"from {summary_obj['n_test_subjects']:,} eyes "
        f"(TF prevalence 9.83%).",
        f"Ensemble threshold (OOF-swept) = {thr:.2f}. "
        f"95% CIs: case-resampling bootstrap, "
        f"n = {summary_obj['n_boot']:,} resamples.",
        "Ensemble predictions are the arithmetic mean of the 5 fold "
        "probabilities; eye-level aggregates per-eye mean "
        "probability before thresholding (max over labels).",
    ]
    return render_table(
        paths["out"] / out_name,
        title=f"Table 2{letter}. TF classification at the image and eye "
              f"level — 5-fold CV ensemble at {label} threshold ({thr:.2f}).",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[1.3, 2.2, 2.2],
        dpi=args.dpi,
    )


def table01a_main_results_kappa(paths, summary, src):
    """5-fold CV ensemble at κ-optimal threshold."""
    sk = summary["kappa"]
    return _table01_one_threshold(
        paths, sk, sk["per_fold_image_summary"],
        label="κ-optimal", thr=sk["ensemble_threshold_oof"],
        letter="a", out_name="table01a_main_results_kappa.png")


def table01b_main_results_youden(paths, summary, src):
    """5-fold CV ensemble at Youden-optimal threshold."""
    sy = summary["youden"]
    return _table01_one_threshold(
        paths, sy, sy["per_fold_image_summary"],
        label="Youden-optimal", thr=sy["ensemble_threshold_oof"],
        letter="b", out_name="table01b_main_results_youden.png")


def table02_combined_thresholds(paths, summary, src):
    """5-fold CV ensemble at both κ-optimal and Youden-optimal thresholds,
    combined into a single hierarchical-header table."""
    sk = summary["kappa"]
    sy = summary["youden"]
    thr_k = sk["ensemble_threshold_oof"]
    thr_y = sy["ensemble_threshold_oof"]

    def cells(summary_obj, k):
        em = summary_obj["ensemble_image_level"]["metrics"][k]
        elo, ehi = summary_obj["ensemble_image_level"]["ci95"].get(k, (None, None))
        sm = summary_obj["ensemble_subject_level"]["metrics"][k]
        slo, shi = summary_obj["ensemble_subject_level"]["ci95"].get(k, (None, None))
        nd = 3 if k != "brier" else 4
        return [_val_ci(em, elo, ehi, nd), _val_ci(sm, slo, shi, nd)]

    def row(metric_label, k):
        return [metric_label] + cells(sk, k) + cells(sy, k)

    rows = [
        row("Cohen's κ",   "kappa"),
        row("AUROC",       "auroc"),
        row("AUPRC",       "auprc"),
        row("Sensitivity", "sensitivity"),
        row("Specificity", "specificity"),
        row("Precision",   "precision"),
        row("F1 score",    "f1"),
        row("Accuracy",    "accuracy"),
    ]

    columns = [
        "Metric",
        (f"κ-optimal threshold ({thr_k:.2f})",
         ["Image level [95% CI]", "Eye level [95% CI]"]),
        (f"Youden-optimal threshold ({thr_y:.2f})",
         ["Image level [95% CI]", "Eye level [95% CI]"]),
    ]
    footnotes = [
        f"Held-out test set: n = {sk['n_test']:,} images from "
        f"{sk['n_test_subjects']:,} eyes (TF prevalence 9.83%).",
        f"Ensemble thresholds selected out-of-fold on the pooled "
        f"validation stack. 95% CIs: case-resampling bootstrap, "
        f"n = {sk['n_boot']:,} resamples.",
        "Ensemble predictions are the arithmetic mean of the 5 fold "
        "probabilities; eye-level aggregates per-eye mean probability "
        "before thresholding (max over labels).",
    ]
    return render_table(
        paths["out"] / "table02_main_results_combined.png",
        title="Table 2. TF classification at the image and eye level — "
              "5-fold CV ensemble at both operating thresholds.",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[1.3, 2.1, 2.1, 2.1, 2.1],
        dpi=args.dpi,
    )


def table02_per_source_cv(paths, src):
    """Per-source 5-fold CV image-level metrics."""
    ens = src["img_ens"].set_index("source")
    pf = src["img_per_fold_summary"].set_index("source")
    order = ens.sort_values("kappa", ascending=False).index

    rows = []
    for source in order:
        e = ens.loc[source]
        p = pf.loc[source]
        rows.append([
            SOURCE_SHORT.get(source, source),
            f"{int(e['n']):,}",
            f"{e['prevalence']*100:.2f}%",
            _fmt(e["kappa"]),
            _mean_std(p["kappa_mean"], p["kappa_std"]),
            _fmt(e["auroc"]),
            _fmt(e["sensitivity"]),
            _fmt(e["specificity"]),
        ])
    # combined row from JSON
    # Use the ensemble image-level metrics for combined row
    rows.append(["__SECTION__"])
    columns = [
        "Source", "n", "TF prev.",
        "κ (ens.)", "κ (per-fold, mean ± SD)",
        "AUROC (ens.)", "Sens. (ens.)", "Spec. (ens.)",
    ]
    footnotes = [
        "All metrics evaluated on the held-out test set at the ensemble's "
        "out-of-fold–selected κ-optimal threshold (0.90).",
        "κ per-fold is mean ± SD across the 5 CV folds, each using its own "
        "validation-selected threshold on this source.",
        "Sources ordered by ensemble image-level κ.",
    ]
    # remove the empty section divider we appended
    rows = [r for r in rows if r and r[0] != "__SECTION__"]
    return render_table(
        paths["out"] / "table02_per_source_cv.png",
        title="Table 3. Per-source classification performance — "
              "5-fold CV ensemble, image level.",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[1.7, 0.7, 0.85, 0.9, 1.7, 1.0, 0.9, 0.9],
        dpi=args.dpi,
    )


def _mcnemar_for_table(probs, y, thr):
    """Return (true_prev, pred_prev, diff, p_exact) at the given threshold."""
    pred = (probs >= thr).astype(int)
    n = len(y)
    a = int(((pred == 1) & (y == 1)).sum())
    b = int(((pred == 1) & (y == 0)).sum())
    c = int(((pred == 0) & (y == 1)).sum())
    d = int(((pred == 0) & (y == 0)).sum())
    true_prev = (a + c) / n
    pred_prev = (a + b) / n
    diff = pred_prev - true_prev
    if b + c == 0:
        p_exact = 1.0
    else:
        k = min(b, c)
        p_exact = stats.binomtest(k, b + c, 0.5,
                                  alternative="two-sided").pvalue
    return true_prev, pred_prev, diff, p_exact, n


def _sig_code(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _fmt_p(p):
    if p >= 0.01:
        return f"{p:.3f}"
    return f"{p:.2g}"


def table03_prevalence_significance(paths, probs_ens, idx, summary):
    """Per-source + combined McNemar test for predicted vs true prevalence
    at the image level (one row per test-set image)."""
    thr = summary["kappa"]["recommended_operating_points_ensemble"]["kappa_optimal"]["threshold"]
    meta_path = Path("/home/Trachoma/data/all_metadata.csv")
    assert meta_path.exists(), f"Need {meta_path} for per-source grouping."
    meta = pd.read_csv(meta_path)[["image_path", "source"]]
    df = idx.merge(meta, on="image_path", how="left").reset_index(drop=True)
    assert df["source"].notna().all(), "missing source after merge"
    df["__prob"] = probs_ens

    rows = []
    for source in sorted(df["source"].unique()):
        sub = df[df["source"] == source]
        p_sub = sub["__prob"].to_numpy()
        y_sub = sub["label"].to_numpy().astype(int)
        tp, pp, dd, pp_p, n_sub = _mcnemar_for_table(p_sub, y_sub, thr)
        rows.append([
            SOURCE_SHORT.get(source, source),
            f"{n_sub:,}",
            f"{tp*100:.2f}%",
            f"{pp*100:.2f}%",
            f"{dd*100:+.2f}",
            _fmt_p(pp_p),
            _sig_code(pp_p),
        ])
    # combined (image level)
    tp, pp, dd, pp_p, n = _mcnemar_for_table(
        df["__prob"].to_numpy(),
        df["label"].to_numpy().astype(int),
        thr,
    )
    rows.append(["Combined", f"{n:,}", f"{tp*100:.2f}%",
                 f"{pp*100:.2f}%", f"{dd*100:+.2f}",
                 _fmt_p(pp_p), _sig_code(pp_p)])

    columns = ["Source", "n images", "True prev.", "Predicted prev.",
               "Δ (pp)", "McNemar p", "Sig."]
    footnotes = [
        "One row per test-set image; no per-eye aggregation.",
        f"McNemar's exact (two-sided binomial) test for marginal homogeneity "
        f"at the κ-optimal ensemble threshold (0.90).",
        "Δ = predicted − true prevalence in percentage points (pp).",
        "Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. p ≥ 0.05.",
    ]
    return render_table(
        paths["out"] / "table03_prevalence_significance.png",
        title="Table 4. Predicted vs true TF prevalence — "
              "per-source and combined (image level).",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[1.7, 0.85, 0.95, 1.05, 0.85, 1.0, 0.55],
        dpi=args.dpi,
    )


def table04_literature_comparison(paths, summary):
    """Our work vs published trachoma TF classification work.

    Numbers verified by reading each source PDF (2026-05-30 audit):
      Kim et al.  2019:  PLoS ONE 14(2) e0210463, Table 3 p.9. TF ensemble
                         κ=0.44 [0.26, 0.62], sens 0.86, spec 0.58, n_test=100
                         (50 TF / 50 normal balanced), 2 countries (Niger PRET
                         + Ethiopia TANA).
      Socia et al. 2022: Combined ICAPS+Kim test n=770, TF prev 15.2%. AI
                         alone κ=0.634; AI + overread κ=0.787, sens 0.786,
                         spec 0.976. Threshold tuned to 0.2.
      Joye et al.  2024: WUHA Ethiopia single site, n=56,725 images, test
                         n=5,622 images / 1,136 subjects, TF prev 30%.
                         AUROC=0.943 [0.931, 0.954], F1=0.923, sens 0.83,
                         spec 0.91. Threshold 0.33 (Youden). NO κ reported.
                         Predicted prevalence 32% vs true 30% (n.s.).
      Pan et al.   2024: Lietman/Kim dataset. ResNet50 + SAM ROI best.
                         Median over 10 splits, TF: sens 0.743, spec 0.866,
                         F1 0.725, AUROC 0.894, acc 0.827. NO κ reported.
                         70/15/15 image-level split, n_test ≈ 247 per run.
                         TF prev ≈32%.
      Kulohoma 2026:    Tropical Data 7-country, train+val n=572 total.
                        Val n=89 (per confusion matrix; abstract says 62
                        — inconsistency). TF+TT merged; trachoma prev
                        47/89 = 52.8%. κ 0.71 [0.58, 0.84], acc 0.854
                        [0.763, 0.920], sens 0.809, spec 0.905. NO AUROC.
      Yazbeck multi-   71,206 images, 3 countries, MobileNetV3. Complete-
        region 2026:   model on Ethiopia test: AUROC 0.96, F1 0.85, true
                       prev 28.6%, pred 32.5% (n.s.). Niger: AUROC 0.79,
                       F1 0.25, prev 3.25/1.95% (n.s.). Peru: AUROC 0.99,
                       F1 0.89, prev 26.0/31.2% (n.s.). NO κ reported.
                       Per-region test n not extracted from PDF (tooling
                       issue at audit time).
    """
    ek = summary["kappa"]["ensemble_image_level"]
    sk = summary["kappa"]["ensemble_subject_level"]

    def kciv(k, lo=None, hi=None):
        if lo is None:
            return _fmt(k)
        return f"{_fmt(k)} [{_fmt(lo,2)}, {_fmt(hi,2)}]"

    rows = [
        # Study, Year, Test set / setting, TF prev, κ [CI], AUROC, Sens, Spec
        ["Kim et al.",   "2019",
         "100 images, balanced ensemble", "50.0%",
         kciv(0.44, 0.26, 0.62), "—", "0.860", "0.580"],
        ["Socia et al.",  "2022",
         "ICAPS+Kim, AI alone (thr=0.20)", "15.2%",
         kciv(0.634), "—", "—", "—"],
        ["Socia et al.",  "2022",
         "ICAPS+Kim, AI + skilled overread", "15.2%",
         kciv(0.787), "—", "0.786", "0.976"],
        ["Joye et al.",   "2024",
         "WUHA Ethiopia (thr=0.33, Youden)", "30.0%",
         "0.732†", "0.943 [0.93, 0.95]", "0.830", "0.910"],
        ["Pan et al.",    "2024",
         "Kim dataset, ResNet50+SAM-ROI (median of 10)", "≈32%",
         "0.606†", "0.894", "0.743", "0.866"],
        ["Kulohoma & Wesonga", "2026",
         "Tropical Data, 7-country, val n=89", "52.8%*",
         kciv(0.71, 0.58, 0.84), "—", "0.809", "0.905"],
        ["Yazbeck (multiregion)", "2026",
         "Complete model, Ethiopia test", "28.6%",
         "—", "0.960", "—", "—"],
        ["Yazbeck (multiregion)", "2026",
         "Complete model, Niger test", "3.25%",
         "—", "0.790", "—", "—"],
        ["Yazbeck (multiregion)", "2026",
         "Complete model, Peru test", "26.0%",
         "—", "0.990", "—", "—"],
        ["__SECTION__This work — 5-fold CV ensemble (κ-optimal, thr=0.90)"],
        ["This work (image)", "2026",
         f"{ek['metrics']['n']:,} images, 8 sources",
         f"{ek['metrics']['prevalence']*100:.2f}%",
         kciv(ek['metrics']['kappa'],
              ek['ci95']['kappa'][0], ek['ci95']['kappa'][1]),
         f"{ek['metrics']['auroc']:.3f}",
         f"{ek['metrics']['sensitivity']:.3f}",
         f"{ek['metrics']['specificity']:.3f}"],
        ["This work (eye)", "2026",
         f"{sk['metrics']['n']:,} eyes, 8 sources",
         f"{sk['metrics']['prevalence']*100:.2f}%",
         kciv(sk['metrics']['kappa'],
              sk['ci95']['kappa'][0], sk['ci95']['kappa'][1]),
         f"{sk['metrics']['auroc']:.3f}",
         f"{sk['metrics']['sensitivity']:.3f}",
         f"{sk['metrics']['specificity']:.3f}"],
    ]

    columns = ["Study", "Year", "Test set / setting", "TF prev.",
               "Cohen's κ [95% CI]", "AUROC", "Sens.", "Spec."]
    footnotes = [
        "All values reproduced from the cited papers (PDFs audited 2026-05-30); "
        "em-dash (—) = metric not reported by the original study.",
        "† κ derived from reported sensitivity, specificity, and prevalence "
        "via the closed-form expression κ = (P_o − P_e) / (1 − P_e); the "
        "original publication did not report κ directly.",
        "* Kulohoma 2026 combines TF + TT into a single positive class; the "
        "52.8% figure is trachoma (TF+TT) prevalence in their 89-image "
        "validation set, not TF alone.",
    ]
    return render_table(
        paths["out"] / "table04_literature_comparison.png",
        title="Table 5. Comparison to published trachoma TF "
              "classification work.",
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        col_widths=[1.9, 0.55, 3.2, 0.85, 1.6, 1.1, 0.75, 0.75],
        alignments=["left", "center", "left", "right",
                    "right", "right", "right", "right"],
        dpi=args.dpi,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    set_style()
    paths = build_paths(args)
    summary = load_cv_summary(paths)
    if "kappa" not in summary:
        raise SystemExit(
            f"Missing {paths['tier_a']/'cv_summary.json'} — has CV analysis run?")
    probs_by_fold, probs_ens, idx = load_test_predictions(paths)
    src = load_per_source(paths)

    REGISTRY = {
        "fig01": lambda: fig01_dataset_composition(paths, src),
        "fig02": lambda: fig02_performance_curves(
            paths, probs_by_fold, probs_ens, idx, summary),
        "fig03": lambda: fig03_confusion_and_thresholds(
            paths, probs_ens, idx, src, summary),
        "fig04": lambda: fig04_per_source_kappa(paths, src),
        "fig05": lambda: fig05_prevalence_scatter(paths, src),
        "fig06": lambda: fig06_literature_comparison(paths, summary),
        "fig07": lambda: fig07_per_fold_stability(paths, src),
        "fig08": lambda: fig08_loso_kappa(paths, summary),
        "fig09": lambda: fig09_loso_vs_indist(paths, src),
        "table01a": lambda: table01a_main_results_kappa(paths, summary, src),
        "table01b": lambda: table01b_main_results_youden(paths, summary, src),
        "table02_combined": lambda: table02_combined_thresholds(paths, summary, src),
        "table02": lambda: table02_per_source_cv(paths, src),
        "table03": lambda: table03_prevalence_significance(paths, probs_ens, idx, summary),
        "table04": lambda: table04_literature_comparison(paths, summary),
        "table05": lambda: table05_loso_per_source(paths, summary),
        "table06": lambda: table06_loso_prevalence(paths, summary),
        "table_s1": lambda: table_s1_dataset_composition(paths),
    }
    names = args.only if args.only else list(REGISTRY.keys())
    for name in names:
        if name not in REGISTRY:
            print(f"[generate_figs] unknown figure: {name}")
            continue
        out = REGISTRY[name]()
        if out is not None:
            print(f"[generate_figs] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_root", default="eval_outputs/resnet_cv5",
                    help="Root of 5-fold CV eval outputs.")
    ap.add_argument("--loso_eval_root", default="eval_outputs/loso_cv5",
                    help="Root of LOSO-CV eval outputs (used for fig08-09).")
    ap.add_argument("--out_dir", default="eval_outputs/resnet_cv5/figures_paper",
                    help="Where to save the PNG figures.")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Resolution in DPI (300 default, 600 for camera-ready).")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only specific figures (e.g. fig04 fig06).")
    args = ap.parse_args()
    main(args)
