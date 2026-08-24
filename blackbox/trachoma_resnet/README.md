# Trachoma TF Classification — ResNet-50

Binary classification of **TF vs. No-TF** from tarsal conjunctiva photographs, using
an ImageNet-pretrained **ResNet-50** at **512 px**.

This directory contains the code and evaluation artifacts for the two experiments
reported in the paper:

| Experiment | What it measures | Code | Results |
|---|---|---|---|
| **Tier A** — 5-fold grouped CV | Held-out performance on a pooled 20% test set | `src/cv_split.py`, `src/cv_train.py`, `src/cv_analysis.py` | `eval_outputs/resnet_cv5/` |
| **Tier C** — leave-one-source-out CV | Generalization to an unseen study site | `src/loso_cv_train.py`, `src/loso_cv_analysis.py` | `eval_outputs/loso_cv5/` |

The reported numbers are not restated here. Every metric, threshold, and interval
in the paper is committed under `eval_outputs/` and can be recomputed directly —
see [Verifying the reported results](#verifying-the-reported-results). Rendered
versions of the paper tables and figures are in
`eval_outputs/resnet_cv5/figures_paper/`.

---

## Data availability

**The image data is not in this repository and cannot be redistributed.** The
photographs are governed by data-use agreements with the contributing studies
(SOCIT, TANA II, Gambia PRET, ICAPS, Solomon Islands, CC_EA2017, 2022 Australia
Trachoma Images, and Kim et al.). Accordingly, this repo excludes:

- all source photographs and the `data/` tree
- `all_metadata.csv` (the master label file)
- model checkpoints (~16 GB) and training logs under `runs/`
- Grad-CAM exemplar images, which are patient photographs

Requests for data access should be directed to the corresponding author and
routed through the originating studies.

**What this means for reproduction:** retraining from scratch requires the images.
Everything downstream of training — every metric, threshold, and confidence
interval reported in the paper — can be recomputed from files committed here, with
no image access. See [Verifying the reported results](#verifying-the-reported-results).

---

## Verifying the reported results

Saved out-of-sample predictions are committed, so any reported number can be
independently recomputed in seconds:

```python
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

probs = np.load("eval_outputs/resnet_cv5/tier_a/test_probs_ensemble.npy")   # (4631,)
index = pd.read_csv("eval_outputs/resnet_cv5/tier_a/test_index.csv")        # image_path, label
y = index["label"].values

ops = pd.read_csv("eval_outputs/resnet_cv5/tier_a/operating_points_ensemble.csv")
best = ops.loc[ops["kappa"].idxmax()]
t = float(best["threshold"])

print(cohen_kappa_score(y, (probs >= t).astype(int)))  # 0.741885
print(roc_auc_score(y, probs))                         # 0.977415
```

Relevant artifacts, per analysis directory:

| File | Contents |
|---|---|
| `test_probs_ensemble.npy` | Mean predicted probability across the 5 folds, test set |
| `test_probs_by_fold.npy` | Per-fold probabilities, shape `(5, n_test)` |
| `test_index.csv` | Row-aligned `image_path` and ground-truth `label` |
| `operating_points_ensemble.csv` | Full threshold sweep: kappa, sens, spec, PPV, F1, AUROC, AUPRC, Brier |
| `per_source_image_metrics_per_fold.csv` | Metrics broken out by contributing study |

Each experiment is analyzed under **two** threshold-selection rules, in parallel
directories: `tier_a/` selects the threshold maximizing Cohen's kappa, and
`tier_a_youden/` maximizes Youden's J. The LOSO equivalents are
`loso_cv_summary_per_fold_kappa.csv` and `..._youden.csv`.

---

## Reproducing the full pipeline

Requires the image data (see above), a CUDA GPU, and several days of compute.

### Install

```bash
python -m venv venv && source venv/bin/activate

# GPU (as used for the reported results — Python 3.12, CUDA 12.6):
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions of the environment the reported
results were produced in. On a CPU-only machine, skip the first `pip install`
and run `pip install -r requirements.txt` alone — enough to rerun the
verification snippet above, but not to retrain.

### Input format

A single metadata CSV with four required columns:

| Column | Meaning |
|---|---|
| `image_path` | Absolute or relative path to the image |
| `label` | `0` = No-TF, `1` = TF |
| `source` | Contributing study — the held-out unit for LOSO-CV |
| `id` | Subject identifier; combined with `source` to form the grouping key |

Splits are grouped on `(source, id)` so that all images from one subject stay on
the same side of every split. This is what prevents subject-level leakage, and
`cv_split.py` enforces it.

### Run everything

```bash
bash run_all_overnight.sh
```

Validate the wiring first — this runs all five stages end-to-end in 5–10 minutes
at 1 epoch and writes to separate `*_smoke` directories:

```bash
bash run_all_overnight.sh --smoke
```

Override defaults by environment variable:

```bash
CSV=/path/to/all_metadata.csv \
PY=/path/to/venv/bin/python \
LOG=/path/to/run.log \
    bash run_all_overnight.sh
```

The driver is **resumable** — every training step passes `--skip_if_done`, so
re-invoking after a crash picks up where it stopped. Analysis steps need
`--allow_overwrite` to replace an existing output directory.

### Stages

| Step | Script | Cost (1 GPU @ 512 px) |
|---|---|---|
| 1 | `src.cv_split` — 80/20 hold-out + 5 folds, grouped and stratified | seconds |
| 2 | `src.cv_train` × 5 folds | ~25–35 h |
| 3 | `src.cv_analysis` — Tier A ensemble, kappa and Youden | ~1 h |
| 4 | `src.loso_cv_train` — 8 sources × 5 folds = 40 trainings | ~3–4 days |
| 5 | `src.loso_cv_analysis` — Tier C, kappa and Youden | ~1 h |

---

## Configuration

Pipeline-level defaults are set at the top of `run_all_overnight.sh`; model-level
defaults are argparse defaults in `src/cv_train.py`. The values used for the
published runs are the defaults in both files:

| Parameter | Value | Defined in |
|---|---|---|
| Backbone | ResNet-50, ImageNet weights | `src/model.py` |
| Image size | 512 px | `run_all_overnight.sh` |
| Batch size | 8, `accumulate_grad_batches=3` (effective 24) | `run_all_overnight.sh` |
| Seed | 1234 | `run_all_overnight.sh` |
| Folds | 5 | `run_all_overnight.sh` |
| Bootstrap resamples | 10,000 | `run_all_overnight.sh` |
| Learning rate | 1e-4 | `src/cv_train.py` |
| Weight decay | 1e-4 | `src/cv_train.py` |
| Max epochs | 30 | `src/cv_train.py` |
| Frozen backbone epochs | 1 | `src/cv_train.py` |
| Class weighting | `pos_weight`, computed from training-fold prevalence | `src/model.py` |

Per-run resolved hyperparameters, chosen thresholds, and split statistics are
committed as JSON/YAML under `runs/` even though the checkpoints themselves are
not — see `runs/resnet_cv5/splits/split_info.json` for the dataset composition
(23,104 images; 18,473 pool / 4,631 test; 9.83% test prevalence) and per-fold
sizes.

### Augmentation

Training (`src/transforms.py`), chosen to mimic field-acquisition variability:
random resized crop (scale 0.75–1.0), horizontal flip, color jitter (p=0.8),
Gaussian blur (p=0.25), sharpness adjust (p=0.2), autocontrast (p=0.2),
perspective (p=0.15), rotation ±10° (p=0.3). Evaluation is deterministic:
resize to 1.15×, center crop, ImageNet normalization.

---

## Other scripts

| Script | Purpose |
|---|---|
| `generate_figs.py` | Builds all paper figures and tables into `eval_outputs/resnet_cv5/figures_paper/` |
| `generate_gradcam_fig.py` | Grad-CAM panel (outputs excluded from this repo — patient images) |
| `prevalence_test.py` | Tests whether predicted TF prevalence differs significantly from ground truth |
| `src/prevalence_agreement.py` | Prevalence agreement across the 8 sources, Bonferroni-corrected |
| `preprocess_eyelid_mask.py` | Builds the masked-eyelid input variant (`eval_outputs/resnet_masked_eyelid/`) |
| `find_conflicted_subjects.py` | Flags subjects with inconsistent labels across images |
| `src/grid_search.py` | Hyperparameter sweep used during development |
| `src/rescore_loso.py` | Re-scores existing LOSO checkpoints without retraining |
| `src/train.py`, `src/eval.py` | Single-split train/eval, superseded by the CV pipeline (`eval_outputs/resnet_01/`) |

## Layout

```
trachoma_resnet/
├── src/                  training, splitting, and analysis modules
├── eval_outputs/         committed metrics, predictions, and figures
│   ├── resnet_cv5/       Tier A — 5-fold CV (+ figures_paper/)
│   ├── loso_cv5/         Tier C — leave-one-source-out
│   ├── resnet_01/        single-split baseline
│   └── resnet_masked_eyelid/   masked-eyelid ablation
├── runs/                 run metadata (JSON/YAML only; checkpoints excluded)
└── run_all_overnight.sh  master driver
```
