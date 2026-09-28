#!/usr/bin/env bash
set -euo pipefail

# Regenerate missing per-seed per-patient inference for the multi-seed runs.
#
# Why: the multi-seed run trained all 15 students (baseline/T1/O2 × 5 seeds), but
# the per-patient window-level data (inference_results_reformatted.csv, which the
# fairness aggregator needs) is missing for some methods:
#   - O2:        inference never ran on the source box → regenerated here.
#   - baseline:  window data lost to Windows path-length truncation in transfer.
#   - T1:        window data lost to Windows path-length truncation in transfer.
# Only the shallow experiment_results.csv summaries survived the transfer for
# baseline/T1, which is not enough for the aggregator.
#
# All 15 student checkpoints ARE present, so this re-runs per-patient inference
# (NO training). For O2, per-patient inference auto-applies the calibration head
# from student_calibration_head.json next to the checkpoint (llms/time_llm.py),
# reproducing the calibrated O2 result. Baseline/T1 have no sidecar (correct).
#
# Inference only — ~4 min/seed. Idempotent: skips a seed whose 12 reformatted
# CSVs already exist.

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
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
PATIENTS="540,544,552,559,563,567,570,575,584,588,591,596"

SEEDS=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0,'scripts/utilities'); from seeds import fixed_seeds; print(','.join(map(str,fixed_seeds)))")
echo "Seeds: $SEEDS"

# Method dir-name stems (the {seed} placeholder is filled per seed).
METHOD_STEMS=(
  "bert_to_bert-tiny_all_patients_seed{seed}"                       # baseline KD
  "bert_to_bert-tiny_all_patients_fair_teacher_seed{seed}"          # T1
  "bert_to_bert-tiny_all_patients_o2_gender_fair_teacher_seed{seed}" # T1+O2 (auto-calibrated)
)

run_inference_for_dir() {
  local run_dir="$1"; local seed="$2"; local label="$3"
  local inf_dir="$run_dir/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"

  if [[ ! -d "$run_dir" ]]; then echo "  ❌ missing: $run_dir"; return; fi
  local ckpt
  ckpt=$(find "$run_dir" -name student_distilled.pth 2>/dev/null | sort | tail -1 || true)
  if [[ -z "$ckpt" ]]; then echo "  ❌ no checkpoint in $run_dir"; return; fi

  # Already complete? (12 patients' reformatted CSVs)
  local have
  have=$(find "$inf_dir" -name inference_results_reformatted.csv 2>/dev/null | wc -l)
  if [[ "$have" -ge 12 ]]; then echo "  ⏭️  $label seed$seed already complete ($have/12)"; return; fi

  # O2 sanity: warn if calibration sidecar is absent (would give uncalibrated output)
  if [[ "$label" == "O2" ]]; then
    if [[ ! -f "$(dirname "$ckpt")/student_calibration_head.json" ]]; then
      echo "  ⚠️  O2 seed$seed: no calibration sidecar next to checkpoint — skipping to avoid uncalibrated CSV"; return
    fi
  fi

  echo "  ▶ $label seed$seed — inference ($ckpt)"
  mkdir -p "$inf_dir"
  "$PYTHON_BIN" scripts/time_llm/config_generator.py \
    --mode per_patient_inference \
    --checkpoint-path "$ckpt" \
    --llm_models BERT-tiny \
    --patients "$PATIENTS" \
    --seeds "$seed" \
    --dataset ohiot1dm \
    --data_scenario standardized \
    --pred-lengths 9 \
    --torch-dtype float32 \
    --output_dir "$inf_dir"
  "$PYTHON_BIN" scripts/time_llm/run_experiments.py \
    --experiments_dir "$(dirname "$inf_dir")"
}

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for seed in "${SEED_ARR[@]}"; do
  echo "================= SEED $seed ================="
  run_inference_for_dir "$PHASE3_DIR/bert_to_bert-tiny_all_patients_seed${seed}" "$seed" "baseline"
  run_inference_for_dir "$PHASE3_DIR/bert_to_bert-tiny_all_patients_fair_teacher_seed${seed}" "$seed" "T1"
  run_inference_for_dir "$PHASE3_DIR/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher_seed${seed}" "$seed" "O2"
done

echo
echo "================= RE-AGGREGATING MULTI-SEED ================="
"$PYTHON_BIN" scripts/fairness/aggregate_multiseed_results.py \
  --pipeline-dir "$PIPELINE_DIR" \
  --calibration-mode patient-holdout \
  --calibration-folds 2

echo
echo "[regen] Done. Check the summary has all 3 methods × 5 seeds:"
echo "  $PIPELINE_DIR/multiseed_robustness_results.txt"
