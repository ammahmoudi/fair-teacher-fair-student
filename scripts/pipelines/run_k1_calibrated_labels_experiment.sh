#!/usr/bin/env bash
# run_k1_calibrated_labels_experiment.sh
# Run K1: calibrated teacher soft labels during distillation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$VIRTUAL_ENV" == "" ]]; then
    source venv/bin/activate
fi

PIPELINE_DIR="distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17"
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
PATIENTS="540,544,552,559,563,567,570,575,584,588,591,596"
SEED=42

COMMON_DISTILL_ARGS="--teacher bert \
    --student prajjwal1/bert-tiny \
    --all-patients \
    --dataset ohiot1dm \
    --seed $SEED \
    --lr 0.001 \
    --batch-size 32 \
    --alpha 0.5 \
    --beta 0.5 \
    --distill-epochs 10 \
    --teacher-checkpoint-dir $PHASE1_DIR \
    --student-config-dir $PHASE2_DIR \
    --output-dir $PHASE3_DIR \
    --config-output-dir $PHASE3_DIR \
    --pipeline-dir $PIPELINE_DIR"

LOG_FILE="$PIPELINE_DIR/k1_calibrated_labels_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "========================================================================"
echo "K1 CALIBRATED SOFT-LABEL EXPERIMENT"
echo "  Distill with group-conditional teacher output offsets (gender)."
echo "  Log: $LOG_FILE"
echo "  Started: $(date)"
echo "========================================================================"

# ── Step 1: Compute K1 offsets from teacher inference ─────────────────────────
K1_OFFSET_JSON="$PHASE1_DIR/k1_teacher_offsets_gender.json"

echo ""
echo ">> Step 1/4: Compute K1 offsets from teacher inference"
python scripts/fairness/compute_k1_teacher_offsets.py \
    --inference-dir "$PHASE1_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm" \
    --feature gender \
    --group0 Female \
    --group1 Male \
    --base-threshold 70.0 \
    --output "$K1_OFFSET_JSON" \
    2>&1 | tee -a "$LOG_FILE"

# ── Step 2: Distill with K1 calibrated labels ─────────────────────────────────
RUN_K1_DIR="$PHASE3_DIR/bert_to_bert-tiny_all_patients_k1cal_gender"

echo ""
echo ">> Step 2/4: Distill with K1 calibrated teacher labels"
echo "   Output: $RUN_K1_DIR"

if find "$RUN_K1_DIR" -name "student_distilled.pth" 2>/dev/null | grep -q .; then
    echo "   [SKIP] K1 distillation already complete"
else
    python distillation/scripts/distill_students.py \
        $COMMON_DISTILL_ARGS \
        --teacher-calibration \
        --teacher-calibration-json "$K1_OFFSET_JSON" \
        2>&1 | tee -a "$LOG_FILE"
    echo ">> K1 distillation complete: $(date)"
fi

# ── Step 3: Per-patient inference ─────────────────────────────────────────────
echo ""
echo ">> Step 3/4: Per-patient inference for K1 distilled model"
RUN_K1_CHECKPOINT=$(find "$RUN_K1_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1)
RUN_K1_INFERENCE="$RUN_K1_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
mkdir -p "$RUN_K1_INFERENCE"

if [ -f "$RUN_K1_INFERENCE/experiment_results.csv" ]; then
    echo "   [SKIP] K1 inference already done"
else
    python scripts/time_llm/config_generator.py \
        --mode per_patient_inference \
        --checkpoint-path "$RUN_K1_CHECKPOINT" \
        --llm_models BERT-tiny \
        --patients $PATIENTS \
        --seeds $SEED \
        --dataset ohiot1dm \
        --data_scenario standardized \
        --pred-lengths 9 \
        --torch-dtype float32 \
        --output_dir "$RUN_K1_INFERENCE" \
        2>&1 | tee -a "$LOG_FILE"

    python scripts/time_llm/run_experiments.py \
        --experiments_dir "$RUN_K1_DIR/per_patient_inference" \
        2>&1 | tee -a "$LOG_FILE"

    echo ">> K1 per-patient inference complete: $(date)"
fi

# ── Step 4: Regenerate comparison table ───────────────────────────────────────
echo ""
echo ">> Step 4/4: Regenerating fairness comparison table"
python scripts/fairness/compute_fairness_comparison.py \
    --pipeline-dir "$PIPELINE_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================================================"
echo "K1 CALIBRATED SOFT-LABEL EXPERIMENT COMPLETE"
echo "Finished: $(date)"
echo "Results: $PIPELINE_DIR/fairness_comparison_results.txt"
echo "========================================================================"
