#!/usr/bin/env bash
set -euo pipefail

# Multi-seed robustness check.
# Re-runs the three methods that matter for the paper's headline —
#   (1) Baseline KD (no fairness),
#   (2) Distilled from Fair Teacher (T1),
#   (3) T1 + O2 Calibration Head  —
# once per seed in scripts/utilities/seeds.py (fixed_seeds), then aggregates
# RMSE / EO_raw / EO_cal to mean ± std across seeds. Purpose: confirm O2's gap
# (EO_raw ~0.119 vs ~0.230 baseline) is robust to seed and not a single-seed fluke.
#
# Each (method, seed) writes its own dir: <run_name>_seed<seed>.

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
EPOCHS=10

BASE_TEACHER="$PHASE1_DIR/bert_all_patients_10epochs/logs/logs_2025-10-28_14-20-20/checkpoints/checkpoint.pth"
FAIR_TEACHER="$PHASE1_DIR/bert_all_patients_10epochs_fair_gender/bert_all_patients_10epochs/logs/logs_2026-06-13_11-13-20/checkpoints/checkpoint.pth"

for ck in "$BASE_TEACHER" "$FAIR_TEACHER"; do
  [[ -f "$ck" ]] || { echo "Missing teacher checkpoint: $ck"; exit 1; }
done

# Canonical seeds — single source of truth in scripts/utilities/seeds.py
SEEDS=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0,'scripts/utilities'); from seeds import fixed_seeds; print(','.join(map(str,fixed_seeds)))")
echo "Seeds (from scripts/utilities/seeds.py): $SEEDS"

run_inference() {
  local checkpoint="$1"; local inference_dir="$2"; local seed="$3"
  mkdir -p "$inference_dir"
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
  "$PYTHON_BIN" scripts/time_llm/run_experiments.py \
    --experiments_dir "$(dirname "$inference_dir")"
}

# distill_one <method> <teacher_ckpt> <seed> <run_name> [extra distill flags...]
distill_one() {
  local method="$1"; local teacher="$2"; local seed="$3"; local run_name="$4"; shift 4
  local run_dir="$PHASE3_DIR/$run_name"
  local inf_dir="$run_dir/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
  local results="$inf_dir/experiment_results.csv"

  echo "────────── $method | seed $seed | $run_name ──────────"
  local ckpt
  ckpt=$(find "$run_dir" -name "student_distilled.pth" 2>/dev/null | sort | tail -1 || true)
  if [[ -z "$ckpt" ]]; then
    "$PYTHON_BIN" distillation/scripts/distill_students.py \
      --teacher bert --student prajjwal1/bert-tiny --all-patients --dataset ohiot1dm \
      --seed "$seed" --lr 0.001 --batch-size 32 --alpha 0.5 --beta 0.5 --distill-epochs "$EPOCHS" \
      --teacher-checkpoint-path "$teacher" \
      --output-dir "$PHASE3_DIR" --config-output-dir "$PHASE3_DIR" --pipeline-dir "$PIPELINE_DIR" \
      "$@"
    ckpt=$(find "$run_dir" -name "student_distilled.pth" 2>/dev/null | sort | tail -1 || true)
  else
    echo "⏭️  training exists: $ckpt"
  fi
  [[ -n "$ckpt" ]] || { echo "No checkpoint for $run_name"; return 1; }

  if [[ -f "$results" ]] && [[ $(wc -l < "$results") -ge 13 ]]; then
    echo "⏭️  inference complete for $run_name"
  else
    run_inference "$ckpt" "$inf_dir" "$seed"
  fi
}

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for seed in "${SEED_ARR[@]}"; do
  echo "================= SEED $seed ================="
  # 1) Baseline KD (standard teacher, no fairness flags)
  distill_one "Baseline KD" "$BASE_TEACHER" "$seed" \
    "bert_to_bert-tiny_all_patients_seed${seed}" \
    --dir-suffix "_seed${seed}"
  # 2) T1: distilled from fair teacher
  distill_one "T1 fair-teacher" "$FAIR_TEACHER" "$seed" \
    "bert_to_bert-tiny_all_patients_fair_teacher_seed${seed}" \
    --dir-suffix "_fair_teacher_seed${seed}"
  # 3) T1 + O2 calibration head
  distill_one "T1+O2" "$FAIR_TEACHER" "$seed" \
    "bert_to_bert-tiny_all_patients_o2_gender_fair_teacher_seed${seed}" \
    --student-calibration-head --student-calibration-feature gender \
    --fairness-feature gender \
    --dir-suffix "_fair_teacher_seed${seed}"
done

echo "================= AGGREGATING ================="
"$PYTHON_BIN" scripts/fairness/aggregate_multiseed_results.py \
  --pipeline-dir "$PIPELINE_DIR" \
  --calibration-mode patient-holdout \
  --calibration-folds 2

echo "[multiseed] Done. Summary: $PIPELINE_DIR/multiseed_robustness_results.txt"
