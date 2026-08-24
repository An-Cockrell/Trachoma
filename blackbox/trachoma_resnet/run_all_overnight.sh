#!/usr/bin/env bash
# Master driver for the 5-fold CV + LOSO-CV experiment.
#
# Order of operations:
#   1. cv_split.py            (one-time: makes the 80/20 hold-out + 5 fold splits)
#   2. cv_train.py x 5 folds  (full-data CV, ~25-35h on one GPU @ 512px)
#   3. cv_analysis.py         (Tier A ensemble + per-fold, kappa AND youden)
#   4. loso_cv_train.py       (40 trainings, ~3-4 days on one GPU @ 512px)
#   5. loso_cv_analysis.py    (Tier C ensemble + per-fold, kappa AND youden)
#
# Resumable: every train step uses --skip_if_done, so re-invoking the script
# after a crash picks up where it left off. Analysis steps require
# --allow_overwrite to rewrite an existing output dir.
#
# Smoke test (5-10 minutes, validates the whole pipeline):
#   bash run_all_overnight.sh --smoke
#
# Real overnight + multi-day run:
#   bash run_all_overnight.sh
#
# Custom CSV / venv / log path:
#   CSV=/path/all_metadata.csv PY=/path/venv/bin/python LOG=/path/run.log \
#       bash run_all_overnight.sh

set -euo pipefail

# ----- Configurable via environment variables -----
PY=${PY:-/home/Trachoma/venv/bin/python}
CSV=${CSV:-/home/Trachoma/data/all_metadata.csv}
CV_OUT=${CV_OUT:-runs/resnet_cv5}
LOSO_OUT=${LOSO_OUT:-runs/loso_cv5}
EVAL_CV=${EVAL_CV:-eval_outputs/resnet_cv5}
EVAL_LOSO=${EVAL_LOSO:-eval_outputs/loso_cv5}
LOG=${LOG:-runs/cv_overnight.log}
SEED=${SEED:-1234}
N_FOLDS=${N_FOLDS:-5}
IMG_SIZE=${IMG_SIZE:-512}
BS=${BS:-8}
ACCUM=${ACCUM:-3}
NUM_WORKERS=${NUM_WORKERS:-8}
N_BOOT=${N_BOOT:-10000}

# ----- Smoke-test mode -----
SMOKE_ARGS_TRAIN=""
SMOKE_ARGS_LOSO=""
if [[ "${1:-}" == "--smoke" ]]; then
    echo "[run_all] SMOKE MODE — fast sanity run, NOT a real experiment"
    CV_OUT=runs/resnet_cv5_smoke
    LOSO_OUT=runs/loso_cv5_smoke
    EVAL_CV=eval_outputs/resnet_cv5_smoke
    EVAL_LOSO=eval_outputs/loso_cv5_smoke
    LOG=runs/cv_overnight_smoke.log
    SMOKE_ARGS_TRAIN="--max_epochs 1 --limit_train_batches 4 --limit_val_batches 4"
    SMOKE_ARGS_LOSO="--max_epochs 1 --limit_train_batches 4 --limit_val_batches 4 --sources ICAPS --folds 0 1"
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "================================================================"
echo " CV + LOSO-CV pipeline started at $(date)"
echo " Python:        $PY"
echo " CSV:           $CSV"
echo " CV out:        $CV_OUT"
echo " LOSO out:      $LOSO_OUT"
echo " Eval CV:       $EVAL_CV"
echo " Eval LOSO:     $EVAL_LOSO"
echo " Image size:    $IMG_SIZE px"
echo " Batch size:    $BS (accumulate $ACCUM -> effective $((BS * ACCUM)))"
echo " Bootstrap n:   $N_BOOT"
echo " Log:           $LOG"
[[ -n "$SMOKE_ARGS_TRAIN" ]] && echo " MODE:          SMOKE"
echo "================================================================"

# Sanity checks before kicking off long runs
if [[ ! -x "$PY" ]]; then
    echo "[run_all] ERROR: python not executable at $PY"; exit 1
fi
if [[ ! -f "$CSV" ]]; then
    echo "[run_all] ERROR: CSV not found at $CSV"; exit 1
fi

# ====================================================================
# STEP 1 — Build the 80/20 + 5-fold splits (idempotent)
# ====================================================================
echo
echo "----------------------------------------------------------------"
echo "[run_all] STEP 1: cv_split.py"
echo "----------------------------------------------------------------"
SPLIT_INFO="$CV_OUT/splits/split_info.json"
if [[ -f "$SPLIT_INFO" ]]; then
    echo "[run_all] splits already exist at $CV_OUT/splits — skipping"
else
    $PY -m src.cv_split \
        --csv "$CSV" \
        --out_dir "$CV_OUT" \
        --n_folds "$N_FOLDS" \
        --seed "$SEED"
fi

# ====================================================================
# STEP 2 — Train each of the 5 full-data folds
# ====================================================================
echo
echo "----------------------------------------------------------------"
echo "[run_all] STEP 2: cv_train.py x $N_FOLDS folds"
echo "----------------------------------------------------------------"
for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo
    echo "[run_all] >>> cv_train fold $fold"
    $PY -m src.cv_train \
        --csv "$CSV" \
        --splits_dir "$CV_OUT/splits" \
        --out_dir "$CV_OUT" \
        --fold "$fold" \
        --img_size "$IMG_SIZE" \
        --batch_size "$BS" \
        --accumulate_grad_batches "$ACCUM" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED" \
        --skip_if_done $SMOKE_ARGS_TRAIN
done

# ====================================================================
# STEP 3 — Analyze the full-data CV (kappa + youden)
# ====================================================================
echo
echo "----------------------------------------------------------------"
echo "[run_all] STEP 3: cv_analysis.py (kappa, then youden)"
echo "----------------------------------------------------------------"
$PY -m src.cv_analysis \
    --csv "$CSV" \
    --splits_dir "$CV_OUT/splits" \
    --runs_dir "$CV_OUT" \
    --n_folds "$N_FOLDS" \
    --objective kappa \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --n_boot "$N_BOOT" \
    --seed "$SEED" \
    --out_dir "$EVAL_CV/tier_a" \
    --allow_overwrite

$PY -m src.cv_analysis \
    --csv "$CSV" \
    --splits_dir "$CV_OUT/splits" \
    --runs_dir "$CV_OUT" \
    --n_folds "$N_FOLDS" \
    --objective youden \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --n_boot "$N_BOOT" \
    --seed "$SEED" \
    --out_dir "$EVAL_CV/tier_a_youden" \
    --allow_overwrite

# ====================================================================
# STEP 4 — Train all 40 LOSO-CV folds
# ====================================================================
echo
echo "----------------------------------------------------------------"
echo "[run_all] STEP 4: loso_cv_train.py (8 sources x $N_FOLDS folds)"
echo "----------------------------------------------------------------"
$PY -m src.loso_cv_train \
    --csv "$CSV" \
    --out_dir "$LOSO_OUT" \
    --n_folds "$N_FOLDS" \
    --img_size "$IMG_SIZE" \
    --batch_size "$BS" \
    --accumulate_grad_batches "$ACCUM" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --skip_if_done $SMOKE_ARGS_LOSO

# ====================================================================
# STEP 5 — Analyze the LOSO-CV (kappa + youden)
# ====================================================================
echo
echo "----------------------------------------------------------------"
echo "[run_all] STEP 5: loso_cv_analysis.py (kappa, then youden)"
echo "----------------------------------------------------------------"
$PY -m src.loso_cv_analysis \
    --csv "$CSV" \
    --runs_dir "$LOSO_OUT" \
    --out_dir "$EVAL_LOSO" \
    --n_folds "$N_FOLDS" \
    --objective kappa \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --n_boot "$N_BOOT" \
    --seed "$SEED" \
    --allow_overwrite

$PY -m src.loso_cv_analysis \
    --csv "$CSV" \
    --runs_dir "$LOSO_OUT" \
    --out_dir "$EVAL_LOSO" \
    --n_folds "$N_FOLDS" \
    --objective youden \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --n_boot "$N_BOOT" \
    --seed "$SEED" \
    --allow_overwrite

echo
echo "================================================================"
echo " ALL DONE at $(date)"
echo " Full-data CV:    $EVAL_CV/tier_a/ , $EVAL_CV/tier_a_youden/"
echo " LOSO-CV:         $EVAL_LOSO/loso_cv_summary_kappa.csv , _youden.csv"
echo "================================================================"
