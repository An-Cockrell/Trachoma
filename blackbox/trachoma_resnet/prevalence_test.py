"""Test whether the model's predicted TF prevalence differs significantly
from the true TF prevalence on the combined (whole) held-out test set.

Method (canonical for paired binary outcomes, matches Yazbeck 2026 and Joye 2024):

  * McNemar's test on the 2x2 paired contingency (model vs ground truth).
    The marginal-difference equivalence makes this the right test for
    "is the model's prevalence rate the same as the true rate?".
  * Report exact (binomial) and continuity-corrected forms.
  * Wilson 95% CIs on both proportions and on the paired prevalence
    difference (Newcombe's hybrid score interval for paired proportions).

We operate at the chosen ensemble threshold (default = kappa-optimal = 0.90,
from cv_summary.json). Pass --objective youden to instead use the Youden
threshold from tier_a_youden/cv_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_paired_diff_ci(b: int, c: int, n: int,
                            alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe (1998) score CI for the difference of two paired proportions.

    For paired binary outcomes (here: model vs truth on the same subject),
    the marginal difference p_model - p_truth equals (b - c) / n where:
      b = model positive, truth negative
      c = model negative, truth positive
    """
    if n == 0:
        return (np.nan, np.nan)
    # marginal proportions
    # We need both marginals; caller passes b, c, n only, but the marginal
    # difference is (b - c)/n. For Newcombe's CI we need each marginal's
    # Wilson CI plus a correlation adjustment.
    raise NotImplementedError(
        "newcombe_paired_diff_ci requires the full 2x2 — use the wrapper below."
    )


def newcombe_paired_diff_ci_full(a: int, b: int, c: int, d: int,
                                 alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe's method 10 for paired-proportion difference 95% CI.

    Contingency cells:
        a = model+, truth+
        b = model+, truth-
        c = model-, truth+
        d = model-, truth-

    Difference of interest: p1 - p2 where
        p1 = (a + b) / n  (model marginal prevalence)
        p2 = (a + c) / n  (truth marginal prevalence)
        n  = a + b + c + d

    Reference: Newcombe RG. Stat Med 1998; 17:2635-50, method 10.
    """
    n = a + b + c + d
    p1 = (a + b) / n
    p2 = (a + c) / n
    diff = p1 - p2

    lo1, hi1 = wilson_ci(a + b, n, alpha)
    lo2, hi2 = wilson_ci(a + c, n, alpha)

    # phi: empirical correlation of the two paired binary variates
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        phi = 0.0
    else:
        phi = (a * d - b * c) / denom

    lower_adj = np.sqrt((p1 - lo1) ** 2 - 2 * phi * (p1 - lo1) * (hi2 - p2)
                        + (hi2 - p2) ** 2)
    upper_adj = np.sqrt((hi1 - p1) ** 2 - 2 * phi * (hi1 - p1) * (p2 - lo2)
                        + (p2 - lo2) ** 2)
    return (diff - lower_adj, diff + upper_adj)


def mcnemar_test(b: int, c: int) -> dict:
    """Both asymptotic (chi-square with continuity correction) and exact
    (binomial) McNemar tests."""
    out = {"b_model_pos_truth_neg": int(b),
           "c_model_neg_truth_pos": int(c),
           "discordant_total": int(b + c)}
    if b + c == 0:
        out["asymptotic_chi2"] = np.nan
        out["asymptotic_pvalue"] = 1.0
        out["exact_pvalue"] = 1.0
        return out
    # continuity-corrected chi-square
    chi2_cc = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    out["asymptotic_chi2_cc"] = float(chi2_cc)
    out["asymptotic_pvalue_cc"] = float(1 - stats.chi2.cdf(chi2_cc, df=1))
    # exact binomial two-sided
    k = min(b, c)
    n_dis = b + c
    out["exact_pvalue"] = float(stats.binomtest(k, n_dis, 0.5,
                                                 alternative="two-sided").pvalue)
    return out


def run(args):
    root = Path(args.eval_root)
    tier_dir = root / ("tier_a" if args.objective == "kappa" else "tier_a_youden")
    summary = json.loads((tier_dir / "cv_summary.json").read_text())
    probs = np.load(tier_dir / "test_probs_ensemble.npy")
    idx = pd.read_csv(tier_dir / "test_index.csv")

    if args.threshold is None:
        thr = summary["ensemble_threshold_oof"]
    else:
        thr = args.threshold

    y = idx["label"].to_numpy().astype(int)
    pred = (probs >= thr).astype(int)
    n = len(y)

    # 2x2 contingency
    a = int(((pred == 1) & (y == 1)).sum())  # model+, truth+
    b = int(((pred == 1) & (y == 0)).sum())  # model+, truth-
    c = int(((pred == 0) & (y == 1)).sum())  # model-, truth+
    d = int(((pred == 0) & (y == 0)).sum())  # model-, truth-

    true_pos = a + c
    pred_pos = a + b
    true_prev = true_pos / n
    pred_prev = pred_pos / n
    diff = pred_prev - true_prev

    lo_t, hi_t = wilson_ci(true_pos, n)
    lo_p, hi_p = wilson_ci(pred_pos, n)
    lo_d, hi_d = newcombe_paired_diff_ci_full(a, b, c, d)
    mc = mcnemar_test(b, c)

    print(f"\nDataset: combined held-out test set")
    print(f"Eval source: {tier_dir}")
    print(f"Objective: {args.objective} (threshold = {thr:.2f})")
    print(f"N images: {n:,}\n")

    print("2x2 paired contingency table (model vs truth):")
    print(f"                  truth+    truth-")
    print(f"    model+      {a:>8,}  {b:>8,}")
    print(f"    model-      {c:>8,}  {d:>8,}\n")

    print(f"True TF prevalence:      {true_prev*100:6.3f}% "
          f"(95% Wilson CI: [{lo_t*100:.3f}%, {hi_t*100:.3f}%])")
    print(f"Predicted TF prevalence: {pred_prev*100:6.3f}% "
          f"(95% Wilson CI: [{lo_p*100:.3f}%, {hi_p*100:.3f}%])")
    print(f"Difference (pred - true): {diff*100:+.3f} pp "
          f"(95% Newcombe CI: [{lo_d*100:+.3f} pp, {hi_d*100:+.3f} pp])")
    if true_prev > 0:
        print(f"Relative bias:           {(diff/true_prev)*100:+.2f}% of true prevalence\n")

    print("McNemar's test for marginal homogeneity (paired proportions):")
    print(f"  discordant pairs: b = {b:,} (model+/truth-)")
    print(f"                    c = {c:,} (model-/truth+)")
    print(f"  asymptotic chi² (with continuity correction) = "
          f"{mc['asymptotic_chi2_cc']:.4f}")
    print(f"  asymptotic p-value:  {mc['asymptotic_pvalue_cc']:.4g}")
    print(f"  exact binomial p-value: {mc['exact_pvalue']:.4g}\n")

    verdict = "NOT statistically different" if mc["exact_pvalue"] > 0.05 \
        else "STATISTICALLY DIFFERENT"
    print(f"VERDICT at α=0.05 (exact test): predicted prevalence is {verdict} "
          f"from true prevalence.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_root", default="eval_outputs/resnet_cv5")
    ap.add_argument("--objective", choices=["kappa", "youden"], default="kappa",
                    help="Which threshold to evaluate at (default kappa-optimal).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the threshold (default: OOF-swept value).")
    args = ap.parse_args()
    run(args)
