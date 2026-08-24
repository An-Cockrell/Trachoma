"""Prevalence-agreement analysis for the LOSO experiment, matched to the method
used by Joye 2024 and Yazbeck multiregion 2026.

For each leave-one-source-out model we treat the held-out source as one
"region" (their unit of analysis) and compare AI-predicted TF prevalence to the
reference (label) prevalence on that source, with 10,000-sample bootstrap CIs.
"No significant difference" is judged by whether the bootstrap CI of the paired
difference (pred - true) contains zero, both uncorrected (alpha=0.05) and with
Bonferroni correction across the 8 sources (Yazbeck used Bonferroni).

Significance is computed via a 4-cell (TP/FP/FN/TN) multinomial bootstrap, which
is exact for prevalence and avoids materialising per-image resamples.

Run:  /home/Trachoma/venv/bin/python -m src.prevalence_agreement
Outputs (in runs/loso/):
  prevalence_agreement_kappa.csv, prevalence_agreement_youden.csv
  bland_altman_prevalence.png
"""
import glob
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSO_DIR = "runs/loso"
B = 10_000
SEED = 12345
ALPHA = 0.05


def load_thresholds(path):
    """Map unique n_test -> (source_name, threshold)."""
    df = pd.read_csv(path)
    return {int(r.n_test): (r.held_out_source, float(r.threshold)) for r in df.itertuples()}


def fold_cells(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn


def bootstrap_prev(tp, fp, fn, tn, rng, B=B):
    n = tp + fp + fn + tn
    probs = np.array([tp, fp, fn, tn], dtype=float) / n
    draws = rng.multinomial(n, probs, size=B)  # B x 4 : tp,fp,fn,tn
    btp, bfp, bfn, _ = draws.T
    pred_prev = (btp + bfp) / n
    true_prev = (btp + bfn) / n
    diff = pred_prev - true_prev
    return pred_prev, true_prev, diff


def pct_ci(a, alpha):
    lo = np.percentile(a, 100 * alpha / 2)
    hi = np.percentile(a, 100 * (1 - alpha / 2))
    return lo, hi


def run(threshold_csv, label):
    table = load_thresholds(threshold_csv)
    rng = np.random.default_rng(SEED)
    rows = []
    n_sources = len(glob.glob(f"{LOSO_DIR}/*/predictions.csv"))
    alpha_bonf = ALPHA / n_sources
    for pcsv in sorted(glob.glob(f"{LOSO_DIR}/*/predictions.csv")):
        df = pd.read_csv(pcsv)
        y = df.label.astype(int).values
        p = df.prob.astype(float).values
        n = len(df)
        src, thr = table[n]
        pred = (p >= thr).astype(int)
        tp, fp, fn, tn = fold_cells(y, pred)
        true_prev = (tp + fn) / n
        pred_prev = (tp + fp) / n
        diff = pred_prev - true_prev
        bp, bt, bd = bootstrap_prev(tp, fp, fn, tn, rng)
        pred_lo, pred_hi = pct_ci(bp, ALPHA)
        d_lo, d_hi = pct_ci(bd, ALPHA)
        db_lo, db_hi = pct_ci(bd, alpha_bonf)
        sig95 = not (d_lo <= 0 <= d_hi)
        sigbonf = not (db_lo <= 0 <= db_hi)
        rows.append(dict(
            source=src, n=n, thr=thr,
            true_prev=true_prev, pred_prev=pred_prev,
            pred_ci_lo=pred_lo, pred_ci_hi=pred_hi,
            diff=diff, diff_ci_lo=d_lo, diff_ci_hi=d_hi,
            diff_bonf_lo=db_lo, diff_bonf_hi=db_hi,
            sig_95=sig95, sig_bonferroni=sigbonf,
        ))
    out = pd.DataFrame(rows).sort_values("diff", ascending=False)
    out.to_csv(f"{LOSO_DIR}/prevalence_agreement_{label}.csv", index=False)
    return out


def bland_altman(ax, out, title):
    mean_xy = (out["pred_prev"] + out["true_prev"]) / 2
    diff = out["diff"]
    bias = diff.mean()
    sd = diff.std(ddof=1)
    loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd
    ax.scatter(mean_xy, diff, s=40, zorder=3)
    for x, d, s in zip(mean_xy, diff, out.source):
        ax.annotate(s[:10], (x, d), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(bias, color="C1", lw=1.5, label=f"mean bias {bias:+.3f}")
    ax.axhline(loa_hi, color="C3", ls="--", lw=1, label=f"+1.96 SD {loa_hi:+.3f}")
    ax.axhline(loa_lo, color="C3", ls="--", lw=1, label=f"-1.96 SD {loa_lo:+.3f}")
    ax.axhline(0, color="grey", lw=0.8, zorder=0)
    ax.set_xlabel("mean(pred, true) prevalence")
    ax.set_ylabel("pred - true prevalence")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")


def main():
    res = {}
    for label, csv in [("kappa", f"{LOSO_DIR}/loso_summary_kappa.csv"),
                       ("youden", f"{LOSO_DIR}/loso_summary_youden.csv")]:
        out = run(csv, label)
        res[label] = out
        print(f"\n================  {label.upper()} threshold  ================")
        show = out[["source", "n", "true_prev", "pred_prev", "pred_ci_lo", "pred_ci_hi",
                    "diff", "diff_ci_lo", "diff_ci_hi", "sig_95", "sig_bonferroni"]].copy()
        for c in ["true_prev", "pred_prev", "pred_ci_lo", "pred_ci_hi", "diff", "diff_ci_lo", "diff_ci_hi"]:
            show[c] = show[c].map(lambda v: f"{v:+.3f}" if c == "diff" else f"{v:.3f}")
        print(show.to_string(index=False))
        nsig = out.sig_95.sum()
        nsigb = out.sig_bonferroni.sum()
        print(f"  significant (95% CI excludes 0): {nsig}/{len(out)} ;  Bonferroni: {nsigb}/{len(out)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bland_altman(axes[0], res["kappa"], "LOSO prevalence agreement — kappa-optimal")
    bland_altman(axes[1], res["youden"], "LOSO prevalence agreement — Youden")
    fig.tight_layout()
    fig.savefig(f"{LOSO_DIR}/bland_altman_prevalence.png", dpi=150)
    print(f"\nsaved {LOSO_DIR}/bland_altman_prevalence.png")
    print(f"saved {LOSO_DIR}/prevalence_agreement_{{kappa,youden}}.csv")


if __name__ == "__main__":
    main()
