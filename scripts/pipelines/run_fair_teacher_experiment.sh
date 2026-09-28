#!/usr/bin/env bash
# run_fair_teacher_experiment.sh
# Run T1: Retrain teacher with fair hypo-oversampling, then distill into BERT-tiny.
# Produces comparable run for the paper (same hyperparams as Runs 1–4).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$VIRTUAL_ENV" == "" ]]; then
    source venv/bin/activate
fi

# ── Settings ─────────────────────────────────────────────────────────────────
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

LOG_FILE="$PIPELINE_DIR/fair_teacher_experiment_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "========================================================================"
echo "T1: FAIR TEACHER EXPERIMENT"
echo "  Train a teacher with hypo-oversampling, then distill into BERT-tiny."
echo "  Log: $LOG_FILE"
echo "  Started: $(date)"
echo "========================================================================"

# ── Step 1: Train fair teacher ────────────────────────────────────────────────
FAIR_TEACHER_DIR="$PHASE1_DIR/bert_all_patients_10epochs_fair_gender"
FAIR_TEACHER_CONFIG="$PHASE1_DIR/config_teacher_fair_gender.gin"

echo ""
echo ">> Step 1/3: Train fair teacher (with hypo oversampling by gender)"

if find "$FAIR_TEACHER_DIR" -name "best_model.pth" 2>/dev/null | grep -q .; then
    echo "   [SKIP] Fair teacher already trained: $FAIR_TEACHER_DIR"
else
    echo "   Training fair teacher (BERT, all_patients, 10 epochs, fair_teacher_sampling=gender)..."
    python distillation/scripts/train_teachers.py \
        --model bert \
        --all-patients \
        --dataset ohiot1dm \
        --seed $SEED \
        --lr 0.001 \
        --batch-size 32 \
        --epochs 10 \
        --output-dir "$FAIR_TEACHER_DIR" \
        --config-dir "$PHASE1_DIR" \
        --fair-teacher \
        --fair-teacher-feature gender \
        2>&1 | tee -a "$LOG_FILE"

    echo ">> Fair teacher training complete: $(date)"
fi

# ── Step 2: Find fair teacher checkpoint ─────────────────────────────────────
echo ""
echo ">> Step 2/3: Locating fair teacher checkpoint"
FAIR_TEACHER_CHECKPOINT=$(find "$FAIR_TEACHER_DIR" -name "checkpoint.pth" 2>/dev/null | sort | tail -1)
if [ -z "$FAIR_TEACHER_CHECKPOINT" ]; then
    echo "ERROR: Could not find fair teacher checkpoint (checkpoint.pth) in $FAIR_TEACHER_DIR"
    exit 1
fi
echo "   Found: $FAIR_TEACHER_CHECKPOINT"

# ── Step 3: Distill from fair teacher (no fairness loss, clean baseline) ─────
RUN5_DIR="$PHASE3_DIR/bert_to_bert-tiny_all_patients_fair_teacher"
echo ""
echo ">> Step 3/3: Distill from fair teacher into BERT-tiny"
echo "   Output: $RUN5_DIR"

if find "$RUN5_DIR" -name "student_distilled.pth" 2>/dev/null | grep -q .; then
    echo "   [SKIP] Distillation already complete"
else
    python distillation/scripts/distill_students.py \
        $COMMON_DISTILL_ARGS \
        --teacher-checkpoint-dir "$FAIR_TEACHER_DIR" \
        --dir-suffix "_fair_teacher" \
        2>&1 | tee -a "$LOG_FILE"
    echo ">> Distillation complete: $(date)"
fi

# ── Step 4: Per-patient inference ─────────────────────────────────────────────
echo ""
echo ">> Step 4: Per-patient inference for fair-teacher distilled model"
RUN5_CHECKPOINT=$(find "$RUN5_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1)
RUN5_INFERENCE="$RUN5_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
mkdir -p "$RUN5_INFERENCE"

if [ -f "$RUN5_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv" ]; then
    echo "   [SKIP] Inference already done"
else
    python scripts/time_llm/config_generator.py \
        --mode per_patient_inference \
        --checkpoint-path "$RUN5_CHECKPOINT" \
        --llm_models BERT-tiny \
        --patients $PATIENTS \
        --seeds $SEED \
        --dataset ohiot1dm \
        --data_scenario standardized \
        --pred-lengths 9 \
        --torch-dtype float32 \
        --output_dir "$RUN5_INFERENCE" \
        2>&1 | tee -a "$LOG_FILE"

    python scripts/time_llm/run_experiments.py \
        --experiments_dir "$RUN5_DIR/per_patient_inference" \
        2>&1 | tee -a "$LOG_FILE"

    echo ">> Per-patient inference complete: $(date)"
fi

# ── Step 5: Regenerate comparison table ───────────────────────────────────────
echo ""
echo ">> Step 5: Regenerating fairness comparison table"
python scripts/fairness/compute_fairness_comparison.py \
    --pipeline-dir "$PIPELINE_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================================================"
echo "T1 FAIR TEACHER EXPERIMENT COMPLETE"
echo "Finished: $(date)"
echo "Results: $PIPELINE_DIR/fairness_comparison_results.txt"
echo "========================================================================"
