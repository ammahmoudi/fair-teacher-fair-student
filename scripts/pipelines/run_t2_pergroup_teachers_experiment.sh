#!/usr/bin/env bash
set -euo pipefail

# T2 experiment — per-group teachers
# 1) Build male-only and female-only training/testing CSVs from all_patients data
# 2) Train a male-only BERT teacher and a female-only BERT teacher
# 3) Multi-teacher distill into one BERT-tiny student (each sample routed to its
#    group's specialized teacher via batch_groups)
# 4) Run per-patient inference + refresh the fairness comparison table
#
# CAVEAT: the female teacher trains on only 5 patients (559,567,575,588,591) vs
# 7 male patients (540,544,552,563,570,584,596). The female teacher is therefore
# data-thin; T2's result must be read with that limitation in mind.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$ROOT_DIR/venv/bin/activate" ]]; then
  source "$ROOT_DIR/venv/bin/activate"
fi

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PIPELINE_DIR="distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17"
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
PATIENTS="540,544,552,559,563,567,570,575,584,588,591,596"
SEED=42
EPOCHS=10

ALL_DIR="data/ohiot1dm/all_patients_combined"
MALE_DIR="data/ohiot1dm/male_patients_combined"
FEMALE_DIR="data/ohiot1dm/female_patients_combined"

MALE_TEACHER_DIR="$PHASE1_DIR/bert_male_patients_${EPOCHS}epochs"
FEMALE_TEACHER_DIR="$PHASE1_DIR/bert_female_patients_${EPOCHS}epochs"

RUN_NAME="bert_to_bert-tiny_all_patients_t2pergroup_gender_fair_teacher"
RUN_DIR="$PHASE3_DIR/$RUN_NAME"
RUN_INFERENCE_DIR="$RUN_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
RUN_RESULTS="$RUN_INFERENCE_DIR/experiment_results.csv"

# ── Step 1: build group-filtered CSVs (Female=0: 559,567,575,588,591; Male=1: rest) ──
echo "[T2] Building male-only / female-only train+test CSVs"
"$PYTHON_BIN" - <<PYEOF
import os, pandas as pd
male = {540,544,552,563,570,584,596}
female = {559,567,575,588,591}
os.makedirs("$MALE_DIR", exist_ok=True)
os.makedirs("$FEMALE_DIR", exist_ok=True)
for split in ["training", "testing"]:
    src = f"$ALL_DIR/all_patients_{split}.csv"
    df = pd.read_csv(src)
    df[df["item_id"].isin(male)].to_csv(f"$MALE_DIR/all_patients_{split}.csv", index=False)
    df[df["item_id"].isin(female)].to_csv(f"$FEMALE_DIR/all_patients_{split}.csv", index=False)
    print(f"  {split}: male={df['item_id'].isin(male).sum()} rows, female={df['item_id'].isin(female).sum()} rows")
PYEOF

train_teacher() {
  local group_dir="$1"; local out_dir="$2"; local label="$3"
  local ckpt
  ckpt=$(find "$out_dir" -name "checkpoint.pth" 2>/dev/null | sort | tail -1 || true)
  if [[ -n "$ckpt" ]]; then
    echo "⏭️  $label teacher already trained: $ckpt"
    return
  fi
  echo "[T2] Training $label teacher (data: $group_dir)"
  "$PYTHON_BIN" distillation/scripts/train_teachers.py \
    --model bert \
    --all-patients \
    --dataset ohiot1dm \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --lr 0.001 \
    --batch-size 32 \
    --output-dir "$out_dir" \
    --config-dir "$out_dir" \
    --train-data-override "$group_dir/all_patients_training.csv" \
    --test-data-override "$group_dir/all_patients_testing.csv"
}

# ── Step 2: train the two per-group teachers ──
train_teacher "$MALE_DIR"   "$MALE_TEACHER_DIR"   "MALE"
train_teacher "$FEMALE_DIR" "$FEMALE_TEACHER_DIR" "FEMALE"

MALE_CKPT=$(find "$MALE_TEACHER_DIR" -name "checkpoint.pth" 2>/dev/null | sort | tail -1 || true)
FEMALE_CKPT=$(find "$FEMALE_TEACHER_DIR" -name "checkpoint.pth" 2>/dev/null | sort | tail -1 || true)
if [[ -z "$MALE_CKPT" || -z "$FEMALE_CKPT" ]]; then
  echo "Missing a per-group teacher checkpoint. Male=$MALE_CKPT Female=$FEMALE_CKPT"
  exit 1
fi
echo "[T2] Male teacher:   $MALE_CKPT"
echo "[T2] Female teacher: $FEMALE_CKPT"

# ── Step 3: multi-teacher distillation (primary = male/group1, second = female/group0) ──
run_per_patient_inference() {
  local checkpoint="$1"; local inference_dir="$2"
  mkdir -p "$inference_dir"
  "$PYTHON_BIN" scripts/time_llm/config_generator.py \
    --mode per_patient_inference \
    --checkpoint-path "$checkpoint" \
    --llm_models BERT-tiny \
    --patients "$PATIENTS" \
    --seeds "$SEED" \
    --dataset ohiot1dm \
    --data_scenario standardized \
    --pred-lengths 9 \
    --torch-dtype float32 \
    --output_dir "$inference_dir"
  "$PYTHON_BIN" scripts/time_llm/run_experiments.py \
    --experiments_dir "$(dirname "$inference_dir")"
}

echo "========================================================================"
echo "▶  T2: Multi-teacher distillation (male + female teachers → one student)"
echo "   Run dir: $RUN_DIR"
echo "========================================================================"

RUN_CHECKPOINT=$(find "$RUN_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1 || true)
if [[ -n "$RUN_CHECKPOINT" ]]; then
  echo "⏭️  Skipping distillation — checkpoint already exists: $RUN_CHECKPOINT"
else
  "$PYTHON_BIN" distillation/scripts/distill_students.py \
    --teacher bert \
    --student prajjwal1/bert-tiny \
    --all-patients \
    --dataset ohiot1dm \
    --seed "$SEED" \
    --lr 0.001 \
    --batch-size 32 \
    --alpha 0.5 \
    --beta 0.5 \
    --distill-epochs "$EPOCHS" \
    --teacher-checkpoint-path "$MALE_CKPT" \
    --multi-teacher \
    --multi-teacher-feature gender \
    --second-teacher-checkpoint-path "$FEMALE_CKPT" \
    --multi-teacher-group0 0 \
    --output-dir "$PHASE3_DIR" \
    --config-output-dir "$PHASE3_DIR" \
    --pipeline-dir "$PIPELINE_DIR" \
    --dir-suffix _fair_teacher

  RUN_CHECKPOINT=$(find "$RUN_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1 || true)
fi

if [[ -z "$RUN_CHECKPOINT" ]]; then
  echo "T2 student checkpoint not found under: $RUN_DIR"
  exit 1
fi
echo "Student checkpoint: $RUN_CHECKPOINT"

if [[ -f "$RUN_RESULTS" ]] && [[ $(wc -l < "$RUN_RESULTS") -ge 13 ]]; then
  echo "⏭️  Skipping inference — experiment_results.csv already complete"
else
  run_per_patient_inference "$RUN_CHECKPOINT" "$RUN_INFERENCE_DIR"
fi

echo "[T2] Update fairness comparison with leakage-free patient holdout calibration"
"$PYTHON_BIN" scripts/fairness/compute_fairness_comparison.py \
  --pipeline-dir "$PIPELINE_DIR" \
  --calibration-mode patient-holdout \
  --calibration-folds 2

echo "[T2] Done. Outputs:"
echo "  - Run directory: $RUN_DIR"
echo "  - Comparison CSV: $PIPELINE_DIR/fairness_comparison_results.csv"
echo "  - Comparison TXT: $PIPELINE_DIR/fairness_comparison_results.txt"
