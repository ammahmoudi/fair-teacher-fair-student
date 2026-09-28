#!/bin/bash
# ============================================================================
# Fairness Distillation Experiments Pipeline
# ============================================================================
# Runs all fairness distillation variants and computes EO Gap comparison table.
#
# Experiments:
#   Run 3: Oversampling only          (--hypo-oversample, no fairness loss)
#   Run 4: Oversampling + TPR loss    (--hypo-oversample + --fairness-weight)
#
# Already done (not re-run):
#   Run 1: Distilled no fairness      (bert_to_bert-tiny_all_patients/)
#   Run 2: HypoglycemiaTPR loss only  (bert_to_bert-tiny_all_patients_fairness_gender/)
#
# After each training run: generates per-patient inference configs, runs
# inference on all 12 patients, saves experiment_results.csv.
#
# At the end: runs compute_fairness_comparison.py to produce the full table.
#
# Usage:
#   cd /path/to/fair-teacher-fair-student
#   bash scripts/pipelines/run_fairness_distillation_experiments.sh
#
# Estimated time: ~12 hours (2 training runs × ~5h + inference ~1h each)
# ============================================================================

set -euo pipefail

# ── Activate venv ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$VIRTUAL_ENV" == "" ]]; then
    source venv/bin/activate
fi
echo "✅ venv: $VIRTUAL_ENV"
echo "📁 Project: $PROJECT_ROOT"
echo ""

# ── Shared settings ──────────────────────────────────────────────────────────
PIPELINE_DIR="distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17"
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
PATIENTS="540,544,552,559,563,567,570,575,584,588,591,596"
SEED=42
LOG_FILE="$PIPELINE_DIR/fairness_experiments_$(date +%Y%m%d_%H%M%S).log"

COMMON_DISTILL_ARGS="--teacher bert
    --student prajjwal1/bert-tiny
    --all-patients --dataset ohiot1dm
    --seed $SEED --lr 0.001 --batch-size 32
    --alpha 0.5 --beta 0.5 --distill-epochs 10
    --teacher-checkpoint-dir $PHASE1_DIR
    --student-config-dir $PHASE2_DIR
    --output-dir $PHASE3_DIR
    --config-output-dir $PHASE3_DIR
    --pipeline-dir $PIPELINE_DIR
    --fairness-feature gender
    --target-threshold 70.0"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "🔬 Fairness Distillation Experiments Pipeline"
echo "   Started: $(date)"
echo "   Log: $LOG_FILE"
echo "========================================================================"
echo ""

# ── Helper: run per-patient inference for a given checkpoint ─────────────────
run_per_patient_inference() {
    local CHECKPOINT="$1"
    local INFERENCE_DIR="$2"
    local RUN_LABEL="$3"

    echo "──────────────────────────────────────────────────"
    echo "📊 Per-patient inference: $RUN_LABEL"
    echo "   Checkpoint: $CHECKPOINT"
    echo "──────────────────────────────────────────────────"

    mkdir -p "$INFERENCE_DIR"

    python scripts/time_llm/config_generator.py \
        --mode per_patient_inference \
        --checkpoint-path "$CHECKPOINT" \
        --llm_models BERT-tiny \
        --patients "$PATIENTS" \
        --seeds "$SEED" \
        --dataset ohiot1dm \
        --data_scenario standardized \
        --pred-lengths 9 \
        --torch-dtype float32 \
        --output_dir "$INFERENCE_DIR"

    python scripts/time_llm/run_experiments.py \
        --experiments_dir "$(dirname "$INFERENCE_DIR")"

    echo "✅ Inference done: $RUN_LABEL"
    echo ""
}

# ── RUN 3: Oversampling only (no fairness loss) ───────────────────────────────
echo "========================================================================"
echo "▶  RUN 3/4: Oversampling only (--hypo-oversample, no fairness loss)"
echo "   Expected output: bert_to_bert-tiny_all_patients_oversample_gender/"
echo "   Started: $(date)"
echo "========================================================================"

RUN3_DIR="$PHASE3_DIR/bert_to_bert-tiny_all_patients_oversample_gender"
RUN3_CHECKPOINT=$(find "$RUN3_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1)

if [ -n "$RUN3_CHECKPOINT" ]; then
    echo "⏭️  Skipping Run 3 training — checkpoint already exists:"
    echo "   $RUN3_CHECKPOINT"
else
    python distillation/scripts/distill_students.py \
        $COMMON_DISTILL_ARGS \
        --hypo-oversample
    RUN3_CHECKPOINT=$(find "$RUN3_DIR" -name "student_distilled.pth" | sort | tail -1)
fi
echo "Run 3 checkpoint: $RUN3_CHECKPOINT"

RUN3_INFERENCE="$PHASE3_DIR/bert_to_bert-tiny_all_patients_oversample_gender/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
RUN3_RESULTS="$RUN3_INFERENCE/experiment_results.csv"

if [ -f "$RUN3_RESULTS" ] && [ $(wc -l < "$RUN3_RESULTS") -ge 13 ]; then
    echo "⏭️  Skipping Run 3 inference — experiment_results.csv already complete"
else
    run_per_patient_inference "$RUN3_CHECKPOINT" "$RUN3_INFERENCE" "Run3 (oversample only)"
fi

echo "✅ RUN 3 COMPLETE — $(date)"
echo ""

# ── RUN 4: Oversampling + HypoglycemiaTPRLoss ────────────────────────────────
echo "========================================================================"
echo "▶  RUN 4/4: Oversampling + HypoglycemiaTPRLoss (combined)"
echo "   Expected output: bert_to_bert-tiny_all_patients_fairness_gender_oversample/"
echo "   Started: $(date)"
echo "========================================================================"

RUN4_DIR="$PHASE3_DIR/bert_to_bert-tiny_all_patients_fairness_gender_oversample"
RUN4_CHECKPOINT=$(find "$RUN4_DIR" -name "student_distilled.pth" 2>/dev/null | sort | tail -1)

if [ -n "$RUN4_CHECKPOINT" ]; then
    echo "⏭️  Skipping Run 4 training — checkpoint already exists:"
    echo "   $RUN4_CHECKPOINT"
else
    python distillation/scripts/distill_students.py \
        $COMMON_DISTILL_ARGS \
        --fairness-weight 0.3 \
        --hypo-oversample
    RUN4_CHECKPOINT=$(find "$RUN4_DIR" -name "student_distilled.pth" | sort | tail -1)
fi
echo "Run 4 checkpoint: $RUN4_CHECKPOINT"

RUN4_INFERENCE="$PHASE3_DIR/bert_to_bert-tiny_all_patients_fairness_gender_oversample/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
RUN4_RESULTS="$RUN4_INFERENCE/experiment_results.csv"

if [ -f "$RUN4_RESULTS" ] && [ $(wc -l < "$RUN4_RESULTS") -ge 13 ]; then
    echo "⏭️  Skipping Run 4 inference — experiment_results.csv already complete"
else
    run_per_patient_inference "$RUN4_CHECKPOINT" "$RUN4_INFERENCE" "Run4 (oversample + TPR loss)"
fi

echo "✅ RUN 4 COMPLETE — $(date)"
echo ""

# ── Compute full comparison table ─────────────────────────────────────────────
echo "========================================================================"
echo "📊 Computing full fairness comparison table..."
echo "========================================================================"

python scripts/fairness/compute_fairness_comparison.py \
    --pipeline-dir "$PIPELINE_DIR"

echo ""
echo "========================================================================"
echo "🎉 ALL EXPERIMENTS COMPLETE — $(date)"
echo "   Results table: $PIPELINE_DIR/fairness_comparison_results.csv"
echo "   Log: $LOG_FILE"
echo "========================================================================"
