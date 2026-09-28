#!/usr/bin/env bash
set -euo pipefail

# T1 + K4 experiment
# 1) Distill student from the fair teacher checkpoint with selective KD replay
#    (upweight the teacher-matching loss on minority-group [female] hypo windows)
# 2) Run per-patient inference
# 3) Refresh the leakage-free fairness comparison table
#
# K4 pushes the student harder to match the teacher specifically on rare
# female-hypoglycemia windows, without distorting majority-group learning.
# Tuning: KD_REPLAY_FACTOR is the main knob (how much extra weight those windows
# get). Group 0 = Female (the minority) by project convention.

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

KD_REPLAY_FACTOR="${KD_REPLAY_FACTOR:-4.0}"

FAIR_TEACHER_CHECKPOINT="$PHASE1_DIR/bert_all_patients_10epochs_fair_gender/bert_all_patients_10epochs/logs/logs_2026-06-13_11-13-20/checkpoints/checkpoint.pth"
RUN_NAME="bert_to_bert-tiny_all_patients_k4replay_gender_fair_teacher"
RUN_DIR="$PHASE3_DIR/$RUN_NAME"
RUN_CHECKPOINT=""
RUN_INFERENCE_DIR="$RUN_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
RUN_RESULTS="$RUN_INFERENCE_DIR/experiment_results.csv"

if [[ ! -f "$FAIR_TEACHER_CHECKPOINT" ]]; then
  echo "Missing fair teacher checkpoint: $FAIR_TEACHER_CHECKPOINT"
  exit 1
fi

mkdir -p "$PHASE3_DIR"

run_per_patient_inference() {
  local checkpoint="$1"
  local inference_dir="$2"
  local run_label="$3"

  echo "──────────────────────────────────────────────────"
  echo "📊 Per-patient inference: $run_label"
  echo "   Checkpoint: $checkpoint"
  echo "──────────────────────────────────────────────────"

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
echo "▶  T1 + K4: Distill from fair teacher with selective KD replay"
echo "   Pipeline:    $PIPELINE_DIR"
echo "   Run dir:     $RUN_DIR"
echo "   Replay factor: $KD_REPLAY_FACTOR  (minority group 0 = Female)"
echo "========================================================================"

RUN_CHECKPOINT=$(find "$RUN_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1 || true)
if [[ -n "$RUN_CHECKPOINT" ]]; then
  echo "⏭️  Skipping training — checkpoint already exists:"
  echo "   $RUN_CHECKPOINT"
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
    --distill-epochs 10 \
    --teacher-checkpoint-path "$FAIR_TEACHER_CHECKPOINT" \
    --output-dir "$PHASE3_DIR" \
    --config-output-dir "$PHASE3_DIR" \
    --pipeline-dir "$PIPELINE_DIR" \
    --kd-replay \
    --kd-replay-feature gender \
    --kd-replay-minority-group 0 \
    --kd-replay-factor "$KD_REPLAY_FACTOR" \
    --dir-suffix _fair_teacher

  RUN_CHECKPOINT=$(find "$RUN_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1 || true)
fi

if [[ -z "$RUN_CHECKPOINT" ]]; then
  echo "T1+K4 checkpoint not found under: $RUN_DIR"
  exit 1
fi

echo "Student checkpoint: $RUN_CHECKPOINT"

if [[ -f "$RUN_RESULTS" ]] && [[ $(wc -l < "$RUN_RESULTS") -ge 13 ]]; then
  echo "⏭️  Skipping inference — experiment_results.csv already complete"
else
  run_per_patient_inference "$RUN_CHECKPOINT" "$RUN_INFERENCE_DIR" "T1+K4"
fi

echo "[T1+K4] Update fairness comparison with leakage-free patient holdout calibration"
"$PYTHON_BIN" scripts/fairness/compute_fairness_comparison.py \
  --pipeline-dir "$PIPELINE_DIR" \
  --calibration-mode patient-holdout \
  --calibration-folds 2

echo "[T1+K4] Done. Outputs:"
echo "  - Run directory: $RUN_DIR"
echo "  - Comparison CSV: $PIPELINE_DIR/fairness_comparison_results.csv"
echo "  - Comparison TXT: $PIPELINE_DIR/fairness_comparison_results.txt"
