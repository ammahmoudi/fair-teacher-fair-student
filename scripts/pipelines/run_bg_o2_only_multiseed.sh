#!/usr/bin/env bash
# Run the standalone O2-only five-seed OhioT1DM BG ablation.
#
# O2-only distills from the canonical baseline BERT teacher while jointly
# learning the gender-conditioned student output calibration head. It excludes
# every T1, T1+O2, baseline, and other fairness-method condition.

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
SOURCE_PIPELINE_DIR="${SOURCE_PIPELINE_DIR:-distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17}"
PIPELINE_DIR="${PIPELINE_DIR:-distillation_experiments/all_patients_pipeline/o2_only_multiseed}"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
PATIENTS="540,544,552,559,563,567,570,575,584,588,591,596"
INFERENCE_SUBPATH="per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
BASE_TEACHER="$SOURCE_PIPELINE_DIR/phase_1_teacher/bert_all_patients_10epochs/logs/logs_2025-10-28_14-20-20/checkpoints/checkpoint.pth"

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
if [[ "$DRY_RUN" == "0" && ! -f "$BASE_TEACHER" ]]; then
  echo "Missing canonical baseline teacher checkpoint: $BASE_TEACHER" >&2
  exit 1
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

find_distilled_checkpoint() {
  local directory="$1"
  find "$directory" -name student_distilled.pth -type f 2>/dev/null | sort | tail -1 || true
}

inference_complete() {
  local inference_dir="$1"
  local expected_patients="${2:-12}"
  local completed
  completed=$(find "$inference_dir" -name inference_results_reformatted.csv -type f 2>/dev/null | wc -l)
  [[ "$completed" -ge "$expected_patients" ]]
}

run_inference() {
  local checkpoint="$1"
  local seed="$2"
  local run_dir="$3"
  local inference_dir="$run_dir/$INFERENCE_SUBPATH"

  if [[ "$DRY_RUN" != "1" ]] && inference_complete "$inference_dir"; then
    echo "Skipping O2-only seed $seed inference: $inference_dir"
    return
  fi

  echo "Running O2-only seed $seed per-patient inference"
  run_command \
    "$PYTHON_BIN" scripts/time_llm/config_generator.py \
    --mode per_patient_inference \
    --checkpoint-path "$checkpoint" \
    --llm_models BERT-tiny \
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
    echo "Incomplete per-patient inference for O2-only seed $seed: $inference_dir" >&2
    exit 1
  fi
}

echo "========================================================================"
echo "Standalone O2-only BG five-seed ablation"
echo "Source teacher: $BASE_TEACHER"
echo "Pipeline:       $PIPELINE_DIR"
echo "Seeds:          $SEEDS"
echo "Epochs:         $EPOCHS"
echo "Dry run:        $DRY_RUN"
echo "========================================================================"

for seed in "${SEED_ARR[@]}"; do
  run_name="bert_to_bert-tiny_all_patients_o2_gender_seed${seed}"
  run_dir="$PHASE3_DIR/$run_name"
  checkpoint=$(find_distilled_checkpoint "$run_dir")

  if [[ -n "$checkpoint" ]]; then
    echo "Skipping O2-only seed $seed training: $checkpoint"
  else
    echo "Running O2-only seed $seed"
    run_command \
      "$PYTHON_BIN" distillation/scripts/distill_students.py \
      --teacher bert \
      --student prajjwal1/bert-tiny \
      --all-patients \
      --dataset ohiot1dm \
      --seed "$seed" \
      --lr 0.001 \
      --batch-size 32 \
      --alpha 0.5 \
      --beta 0.5 \
      --distill-epochs "$EPOCHS" \
      --teacher-checkpoint-path "$BASE_TEACHER" \
      --output-dir "$PHASE3_DIR" \
      --config-output-dir "$PHASE3_DIR" \
      --pipeline-dir "$PIPELINE_DIR" \
      --student-calibration-head \
      --student-calibration-feature gender \
      --fairness-feature gender \
      --dir-suffix "_seed${seed}"
    if [[ "$DRY_RUN" == "1" ]]; then
      checkpoint="$run_dir/student_distilled.pth"
    else
      checkpoint=$(find_distilled_checkpoint "$run_dir")
      [[ -n "$checkpoint" ]] || { echo "No checkpoint produced for O2-only seed $seed" >&2; exit 1; }
    fi
  fi

  run_inference "$checkpoint" "$seed" "$run_dir"
done

run_command \
  "$PYTHON_BIN" scripts/fairness/aggregate_bg_o2_only_multiseed.py \
  --suite-dir "$PIPELINE_DIR" \
  --seeds "$SEEDS" \
  --calibration-mode patient-holdout \
  --calibration-folds 2

echo "Done. Results: $PIPELINE_DIR/o2_only_multiseed_results.csv"