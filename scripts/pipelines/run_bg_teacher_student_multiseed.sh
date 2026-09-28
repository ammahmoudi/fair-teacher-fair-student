#!/usr/bin/env bash
# Run five-seed BG reference baselines: original Teacher and no-KD Student.

set -euo pipefail

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

DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-10}"
PIPELINE_DIR="${PIPELINE_DIR:-distillation_experiments/all_patients_pipeline/teacher_student_multiseed}"
PATIENTS="540,544,552,559,563,567,570,575,584,588,591,596"
INFERENCE_SUBPATH="per_patient_inference/time_llm_per_patient_inference_ohiot1dm"

if [[ -z "${SEEDS:-}" ]]; then
  SEEDS=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'scripts/utilities'); from seeds import fixed_seeds; print(','.join(map(str, fixed_seeds)))")
fi
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1; received: $DRY_RUN" >&2
  exit 2
fi
if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer; received: $EPOCHS" >&2
  exit 2
fi
for seed in "${SEED_ARR[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 2
  fi
done

run_command() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

find_checkpoint() {
  local run_dir="$1"
  find "$run_dir" -path '*/checkpoints/checkpoint.pth' -type f 2>/dev/null | sort | tail -1 || true
}

inference_complete() {
  local inference_dir="$1"
  local completed
  completed=$(find "$inference_dir" -name inference_results_reformatted.csv -type f 2>/dev/null | wc -l)
  [[ "$completed" -ge 12 ]]
}

run_inference() {
  local label="$1"
  local model="$2"
  local checkpoint="$3"
  local seed="$4"
  local phase_dir="$5"
  local inference_dir="$phase_dir/$INFERENCE_SUBPATH"

  if [[ "$DRY_RUN" != "1" ]] && inference_complete "$inference_dir"; then
    echo "Skipping $label seed $seed inference: $inference_dir"
    return
  fi

  echo "Running $label seed $seed per-patient inference"
  run_command \
    "$PYTHON_BIN" scripts/time_llm/config_generator.py \
    --mode per_patient_inference \
    --checkpoint-path "$checkpoint" \
    --llm_models "$model" \
    --patients "$PATIENTS" \
    --seeds "$seed" \
    --dataset ohiot1dm \
    --data_scenario standardized \
    --pred-lengths 9 \
    --torch-dtype float32 \
    --output_dir "$inference_dir"
  run_command \
    "$PYTHON_BIN" scripts/time_llm/run_experiments.py \
    --experiments_dir "$(dirname "$inference_dir")"

  if [[ "$DRY_RUN" != "1" ]] && ! inference_complete "$inference_dir"; then
    echo "Incomplete $label seed $seed inference: $inference_dir" >&2
    exit 1
  fi
}

run_teacher() {
  local seed="$1"
  local seed_dir="$PIPELINE_DIR/seed_$seed"
  local phase_dir="$seed_dir/phase_1_teacher"
  local checkpoint
  checkpoint=$(find_checkpoint "$phase_dir")

  if [[ -n "$checkpoint" ]]; then
    echo "Skipping original Teacher seed $seed training: $checkpoint"
  else
    echo "Running original Teacher seed $seed"
    run_command \
      "$PYTHON_BIN" distillation/scripts/train_teachers.py \
      --model bert \
      --all-patients \
      --dataset ohiot1dm \
      --seed "$seed" \
      --lr 0.001 \
      --batch-size 32 \
      --epochs "$EPOCHS" \
      --output-dir "$phase_dir" \
      --config-dir "$phase_dir"
    if [[ "$DRY_RUN" == "1" ]]; then
      checkpoint="$phase_dir/bert_all_patients_${EPOCHS}epochs/logs/logs_dry_run/checkpoints/checkpoint.pth"
    else
      checkpoint=$(find_checkpoint "$phase_dir")
      [[ -n "$checkpoint" ]] || { echo "No checkpoint produced for original Teacher seed $seed" >&2; exit 1; }
    fi
  fi

  run_inference "original Teacher" "BERT" "$checkpoint" "$seed" "$phase_dir"
}

run_student() {
  local seed="$1"
  local seed_dir="$PIPELINE_DIR/seed_$seed"
  local phase_dir="$seed_dir/phase_2_student"
  local checkpoint
  checkpoint=$(find_checkpoint "$phase_dir")

  if [[ -n "$checkpoint" ]]; then
    echo "Skipping no-KD Student seed $seed training: $checkpoint"
  else
    echo "Running no-KD Student seed $seed"
    run_command \
      "$PYTHON_BIN" distillation/scripts/train_students.py \
      --model prajjwal1/bert-tiny \
      --all-patients \
      --dataset ohiot1dm \
      --seed "$seed" \
      --lr 0.001 \
      --batch-size 32 \
      --epochs "$EPOCHS" \
      --output-dir "$phase_dir" \
      --config-dir "$phase_dir"
    if [[ "$DRY_RUN" == "1" ]]; then
      checkpoint="$phase_dir/bert-tiny_all_patients_${EPOCHS}epochs/logs/logs_dry_run/checkpoints/checkpoint.pth"
    else
      checkpoint=$(find_checkpoint "$phase_dir")
      [[ -n "$checkpoint" ]] || { echo "No checkpoint produced for no-KD Student seed $seed" >&2; exit 1; }
    fi
  fi

  run_inference "no-KD Student" "BERT-tiny" "$checkpoint" "$seed" "$phase_dir"
}

echo "========================================================================"
echo "BG five-seed reference baselines"
echo "Methods:       original BERT Teacher; independently trained BERT-tiny Student"
echo "Pipeline:      $PIPELINE_DIR"
echo "Seeds:         $SEEDS"
echo "Epochs:        $EPOCHS"
echo "Dry run:       $DRY_RUN"
echo "========================================================================"

for seed in "${SEED_ARR[@]}"; do
  run_teacher "$seed"
  run_student "$seed"
done

run_command \
  "$PYTHON_BIN" scripts/fairness/aggregate_bg_teacher_student_multiseed.py \
  --suite-dir "$PIPELINE_DIR" \
  --seeds "$SEEDS" \
  --calibration-mode patient-holdout \
  --calibration-folds 2

echo "Done. Results: $PIPELINE_DIR/teacher_student_multiseed_results.csv"