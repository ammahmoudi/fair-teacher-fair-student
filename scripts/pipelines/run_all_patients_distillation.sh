#!/bin/bash
# Run distillation pipeline on ALL PATIENTS COMBINED
# This script uses the existing distillation pipeline with the --all-patients flag

# Auto-activate virtual environment if not already active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        echo "🔧 Activating virtual environment..."
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found! Please create one or activate manually."
        exit 1
    fi
fi

echo "🌍 Running Distillation Pipeline on ALL PATIENTS COMBINED"
echo "==========================================================="
echo ""
echo "This will run in 2 main stages:"
echo "  STAGE 1: ALL TRAININGS"
echo "    1. Train teacher model on combined data from all 12 patients (134K samples)"
echo "    2. Train student baseline on combined data from all 12 patients"
echo "    3. Perform knowledge distillation from teacher to student"
echo "  STAGE 2: ALL INFERENCES"
echo "    4. Per-patient inference on teacher"
echo "    5. Per-patient inference on student baseline"
echo "    6. Per-patient inference on distilled student"
echo ""

# Parse command line arguments or use defaults
TEACHER="${TEACHER:-bert}"
STUDENT="${STUDENT:-prajjwal1/bert-tiny}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-5}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-5}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-10}"
SEED="${SEED:-42}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ALPHA="${ALPHA:-0.5}"
BETA="${BETA:-0.5}"
FAIRNESS_WEIGHT="${FAIRNESS_WEIGHT:-0.0}"
FAIRNESS_FEATURE="${FAIRNESS_FEATURE:-}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-70.0}"
PRED_THRESHOLD="${PRED_THRESHOLD:-70.0}"
RUN_TRAINING=true
RUN_INFERENCE=true
EXISTING_PIPELINE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --teacher)
            TEACHER="$2"
            shift 2
            ;;
        --student)
            STUDENT="$2"
            shift 2
            ;;
        --teacher-epochs)
            TEACHER_EPOCHS="$2"
            shift 2
            ;;
        --student-epochs)
            STUDENT_EPOCHS="$2"
            shift 2
            ;;
        --distill-epochs)
            DISTILL_EPOCHS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --fairness-weight)
            FAIRNESS_WEIGHT="$2"
            shift 2
            ;;
        --fairness-feature)
            FAIRNESS_FEATURE="$2"
            shift 2
            ;;
        --target-threshold)
            TARGET_THRESHOLD="$2"
            shift 2
            ;;
        --pred-threshold)
            PRED_THRESHOLD="$2"
            shift 2
            ;;
        --training-only)
            RUN_TRAINING=true
            RUN_INFERENCE=false
            shift
            ;;
        --inference-only)
            RUN_TRAINING=false
            RUN_INFERENCE=true
            shift
            ;;
        --pipeline-dir)
            EXISTING_PIPELINE_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --teacher <model>        Teacher model (default: bert)"
            echo "  --student <model>        Student model (default: prajjwal1/bert-tiny)"
            echo "  --teacher-epochs <n>     Teacher training epochs (default: 5)"
            echo "  --student-epochs <n>     Student training epochs (default: 5)"
            echo "  --distill-epochs <n>     Distillation epochs (default: 10)"
            echo "  --seed <n>               Random seed (default: 42)"
            echo "  --lr <rate>              Learning rate (default: 0.001)"
            echo "  --batch-size <size>      Batch size (default: 32)"
            echo "  --alpha <weight>         Ground truth loss weight (default: 0.5)"
            echo "  --beta <weight>          Teacher loss weight (default: 0.5)"
            echo ""
            echo "Stage Control:"
            echo "  --training-only          Run only training stages (no inference)"
            echo "  --inference-only         Run only inference stages (requires --pipeline-dir)"
            echo "  --pipeline-dir <path>    Use existing pipeline directory for inference"
            echo ""
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run full pipeline (training + inference)"
            echo "  $0 --teacher bert --student prajjwal1/bert-tiny --teacher-epochs 10"
            echo ""
            echo "  # Run only training"
            echo "  $0 --teacher bert --student prajjwal1/bert-tiny --training-only"
            echo ""
            echo "  # Run only inference on existing results"
            echo "  $0 --inference-only --pipeline-dir distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_10-30-45"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ "$RUN_INFERENCE" = true ] && [ "$RUN_TRAINING" = false ]; then
    if [ -z "$EXISTING_PIPELINE_DIR" ]; then
        echo "❌ Error: --inference-only requires --pipeline-dir to specify existing training results"
        echo "Usage: $0 --inference-only --pipeline-dir <path_to_pipeline>"
        exit 1
    fi
    if [ ! -d "$EXISTING_PIPELINE_DIR" ]; then
        echo "❌ Error: Pipeline directory not found: $EXISTING_PIPELINE_DIR"
        exit 1
    fi
fi

echo "Configuration:"
echo "  Teacher: $TEACHER ($TEACHER_EPOCHS epochs)"
echo "  Student: $STUDENT ($STUDENT_EPOCHS epochs)"
echo "  Distillation: $DISTILL_EPOCHS epochs"
echo "  Seed: $SEED"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Alpha (ground truth weight): $ALPHA"
echo "  Beta (teacher weight): $BETA"
if [ -n "$FAIRNESS_FEATURE" ]; then
    echo "  Fairness Weight: $FAIRNESS_WEIGHT"
    echo "  Fairness Feature: $FAIRNESS_FEATURE"
    echo "  Target Threshold: $TARGET_THRESHOLD mg/dL"
    echo "  Pred Threshold: $PRED_THRESHOLD mg/dL"
fi
echo ""
echo "Execution Mode:"
if [ "$RUN_TRAINING" = true ] && [ "$RUN_INFERENCE" = true ]; then
    echo "  ✅ Full Pipeline (Training + Inference)"
elif [ "$RUN_TRAINING" = true ]; then
    echo "  🏋️  Training Only"
elif [ "$RUN_INFERENCE" = true ]; then
    echo "  🔍 Inference Only (using: $EXISTING_PIPELINE_DIR)"
fi
echo ""

# Create or use existing pipeline directory
if [ -n "$EXISTING_PIPELINE_DIR" ]; then
    PIPELINE_DIR="$EXISTING_PIPELINE_DIR"
    echo "Using existing pipeline directory: $PIPELINE_DIR"
else
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    PIPELINE_DIR="distillation_experiments/all_patients_pipeline/pipeline_${TIMESTAMP}"
    mkdir -p "$PIPELINE_DIR"
    echo "Created new pipeline directory: $PIPELINE_DIR"
fi
echo ""

# Record start time
PIPELINE_START_TIME=$(date +%s)

# ========================================
# STAGE 1: TRAINING
# ========================================
if [ "$RUN_TRAINING" = true ]; then
echo "========================================="
echo "🏋️  STAGE 1: ALL TRAININGS"
echo "========================================="
echo ""

# ========================================
# PHASE 1: Train Teacher
# ========================================
echo "=================================================="
echo "🎓 Phase 1/3: Training Teacher Model ($TEACHER)"
echo "=================================================="
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
mkdir -p "$PHASE1_DIR"

python distillation/scripts/train_teachers.py \
    --model $TEACHER \
    --all-patients \
    --dataset ohiot1dm \
    --seed $SEED \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --epochs $TEACHER_EPOCHS \
    --output-dir "$PHASE1_DIR" \
    --config-dir "$PHASE1_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Teacher training failed!"
    exit 1
fi
echo "✅ Teacher training completed"
echo ""

# ========================================
# PHASE 2: Train Student Baseline
# ========================================
echo "=================================================="
echo "👨‍🎓 Phase 2/3: Training Student Baseline ($STUDENT)"
echo "=================================================="
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
mkdir -p "$PHASE2_DIR"

python distillation/scripts/train_students.py \
    --model $STUDENT \
    --all-patients \
    --dataset ohiot1dm \
    --seed $SEED \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --epochs $STUDENT_EPOCHS \
    --output-dir "$PHASE2_DIR" \
    --config-dir "$PHASE2_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Student baseline training failed!"
    exit 1
fi
echo "✅ Student baseline training completed"
echo ""

# ========================================
# PHASE 3: Knowledge Distillation
# ========================================
echo "=================================================="
echo "🧠 Phase 3/3: Knowledge Distillation ($TEACHER → $STUDENT)"
echo "=================================================="
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
mkdir -p "$PHASE3_DIR"

python distillation/scripts/distill_students.py \
    --teacher $TEACHER \
    --student $STUDENT \
    --all-patients \
    --dataset ohiot1dm \
    --seed $SEED \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --alpha $ALPHA \
    --beta $BETA \
    --distill-epochs $DISTILL_EPOCHS \
    --teacher-checkpoint-dir "$PHASE1_DIR" \
    --student-config-dir "$PHASE2_DIR" \
    --output-dir "$PHASE3_DIR" \
    --config-output-dir "$PHASE3_DIR" \
    --pipeline-dir "$PIPELINE_DIR" \
    ${FAIRNESS_WEIGHT:+--fairness-weight $FAIRNESS_WEIGHT} \
    ${FAIRNESS_FEATURE:+--fairness-feature $FAIRNESS_FEATURE} \
    ${FAIRNESS_FEATURE:+--target-threshold $TARGET_THRESHOLD} \
    ${FAIRNESS_FEATURE:+--pred-threshold $PRED_THRESHOLD}

if [ $? -ne 0 ]; then
    echo "❌ Knowledge distillation failed!"
    exit 1
fi
echo "✅ Knowledge distillation completed"
echo ""

fi  # End of RUN_TRAINING

# ========================================
# STAGE 2: INFERENCE
# ========================================
if [ "$RUN_INFERENCE" = true ]; then
echo "========================================="
echo "🔍 STAGE 2: ALL INFERENCES"
echo "========================================="
echo ""

# Define phase directories
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"

# ========================================
# INFERENCE 1: Teacher Model
# ========================================
echo "=================================================="
echo "📊 Inference 1/3: Per-Patient Inference on Teacher"
echo "=================================================="

# Find the teacher checkpoint
TEACHER_CHECKPOINT=$(find "$PHASE1_DIR" -name "checkpoint.pth" -type f | head -1)

if [ -z "$TEACHER_CHECKPOINT" ]; then
    echo "⚠️  No checkpoint found in $PHASE1_DIR (skipping per-patient inference)"
else
    echo "Found checkpoint: $TEACHER_CHECKPOINT"
    
    # Create per-patient inference directory inside phase 1
    # Include time_llm_* prefix so the runner can find it
    TEACHER_INFERENCE_DIR="$PHASE1_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
    
    # Clean up old configs from previous runs
    if [ -d "$TEACHER_INFERENCE_DIR" ]; then
        echo "🧹 Cleaning up old configs from previous runs..."
        rm -rf "$TEACHER_INFERENCE_DIR"/*
    fi
    
    mkdir -p "$TEACHER_INFERENCE_DIR"
    
    # Use unified config generator with per_patient_inference mode
    python scripts/time_llm/config_generator.py \
        --mode per_patient_inference \
        --checkpoint-path "$TEACHER_CHECKPOINT" \
        --llm_models BERT \
        --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
        --seeds $SEED \
        --dataset ohiot1dm \
        --data_scenario standardized \
        --pred-lengths 9 \
        --torch-dtype float32 \
        --output_dir "$TEACHER_INFERENCE_DIR"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Config generation failed for teacher (continuing...)"
    else
        # Run experiments using unified runner - search in the per_patient_inference parent directory
        python scripts/time_llm/run_experiments.py \
            --experiments_dir "$PHASE1_DIR/per_patient_inference"
        
        if [ $? -ne 0 ]; then
            echo "⚠️  Per-patient inference failed for teacher (continuing...)"
        else
            echo "✅ Per-patient inference completed for teacher"
        fi
    fi
fi
echo ""

# ========================================
# INFERENCE 2: Student Baseline
# ========================================
echo "=================================================="
echo "📊 Inference 2/3: Per-Patient Inference on Student Baseline"
echo "=================================================="

# Find the student checkpoint
STUDENT_CHECKPOINT=$(find "$PHASE2_DIR" -name "checkpoint.pth" -type f | head -1)

if [ -z "$STUDENT_CHECKPOINT" ]; then
    echo "⚠️  No checkpoint found in $PHASE2_DIR (skipping per-patient inference)"
else
    echo "Found checkpoint: $STUDENT_CHECKPOINT"
    
    # Create per-patient inference directory inside phase 2
    # Include time_llm_* prefix so the runner can find it
    STUDENT_INFERENCE_DIR="$PHASE2_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
    
    # Clean up old configs from previous runs
    if [ -d "$STUDENT_INFERENCE_DIR" ]; then
        echo "🧹 Cleaning up old configs from previous runs..."
        rm -rf "$STUDENT_INFERENCE_DIR"/*
    fi
    
    mkdir -p "$STUDENT_INFERENCE_DIR"
    
    # Use unified config generator with per_patient_inference mode
    python scripts/time_llm/config_generator.py \
        --mode per_patient_inference \
        --checkpoint-path "$STUDENT_CHECKPOINT" \
        --llm_models BERT-tiny \
        --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
        --seeds $SEED \
        --dataset ohiot1dm \
        --data_scenario standardized \
        --pred-lengths 9 \
        --torch-dtype float32 \
        --output_dir "$STUDENT_INFERENCE_DIR"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Config generation failed for student baseline (continuing...)"
    else
        # Run experiments using unified runner - search in the per_patient_inference parent directory
        python scripts/time_llm/run_experiments.py \
            --experiments_dir "$PHASE2_DIR/per_patient_inference"
        
        if [ $? -ne 0 ]; then
            echo "⚠️  Per-patient inference failed for student baseline (continuing...)"
        else
            echo "✅ Per-patient inference completed for student baseline"
        fi
    fi
fi
echo ""

# ========================================
# INFERENCE 3: Distilled Student
# ========================================
echo "=================================================="
echo "📊 Inference 3/3: Per-Patient Inference on Distilled Student"
echo "=================================================="

# Find the checkpoint (try checkpoint.pth first, then student_distilled.pth)
DISTILLED_CHECKPOINT=$(find "$PHASE3_DIR" -name "checkpoint.pth" -type f | head -1)
if [ -z "$DISTILLED_CHECKPOINT" ]; then
    DISTILLED_CHECKPOINT=$(find "$PHASE3_DIR" -name "student_distilled.pth" -type f | head -1)
fi

if [ -z "$DISTILLED_CHECKPOINT" ]; then
    echo "⚠️  No checkpoint found in $PHASE3_DIR (skipping per-patient inference)"
else
    echo "Found checkpoint: $DISTILLED_CHECKPOINT"
    
    # Create per-patient inference directory inside phase 3
    # Include time_llm_* prefix so the runner can find it
    DISTILLED_INFERENCE_DIR="$PHASE3_DIR/per_patient_inference/time_llm_per_patient_inference_ohiot1dm"
    
    # Clean up old configs from previous runs
    if [ -d "$DISTILLED_INFERENCE_DIR" ]; then
        echo "🧹 Cleaning up old configs from previous runs..."
        rm -rf "$DISTILLED_INFERENCE_DIR"/*
    fi
    
    mkdir -p "$DISTILLED_INFERENCE_DIR"
    
    # Use unified config generator with per_patient_inference mode
    python scripts/time_llm/config_generator.py \
        --mode per_patient_inference \
        --checkpoint-path "$DISTILLED_CHECKPOINT" \
        --llm_models BERT-tiny \
        --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
        --seeds $SEED \
        --dataset ohiot1dm \
        --data_scenario standardized \
        --pred-lengths 9 \
        --torch-dtype float32 \
        --output_dir "$DISTILLED_INFERENCE_DIR"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Config generation failed for distilled student (continuing...)"
    else
        # Run experiments using unified runner - search in the per_patient_inference parent directory
        python scripts/time_llm/run_experiments.py \
            --experiments_dir "$PHASE3_DIR/per_patient_inference"
        
        if [ $? -ne 0 ]; then
            echo "⚠️  Per-patient inference failed for distilled student (continuing...)"
        else
            echo "✅ Per-patient inference completed for distilled student"
        fi
    fi
fi
echo ""

fi  # End of RUN_INFERENCE

# ========================================
# Calculate runtime and display summary
# ========================================
PIPELINE_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo "🎉 SUCCESS: All-Patients Distillation Pipeline Completed!"
echo "==========================================================="
if [ "$RUN_TRAINING" = true ]; then
    echo "✅ Phase 1: Teacher trained ($TEACHER_EPOCHS epochs)"
    echo "✅ Phase 2: Student baseline trained ($STUDENT_EPOCHS epochs)"
    echo "✅ Phase 3: Knowledge distilled ($DISTILL_EPOCHS epochs)"
fi
if [ "$RUN_INFERENCE" = true ]; then
    echo "✅ Inference: Per-patient evaluations completed for all models"
fi
echo ""
echo "📁 Results saved in: $PIPELINE_DIR"
echo "⏱️  Total Runtime: ${TOTAL_RUNTIME}s ($(($TOTAL_RUNTIME / 60))m $(($TOTAL_RUNTIME % 60))s)"
echo ""
if [ "$RUN_TRAINING" = true ] && [ "$RUN_INFERENCE" = false ]; then
    echo "Next steps:"
    echo "  - Run inference: $0 --inference-only --pipeline-dir $PIPELINE_DIR"
    echo "  - Check training logs: find $PIPELINE_DIR -name '*.log'"
    echo "  - View training metrics: find $PIPELINE_DIR -name '*summary.json'"
elif [ "$RUN_INFERENCE" = true ]; then
    echo "Next steps:"
    echo "  - Check inference logs: find $PIPELINE_DIR -name '*.log'"
    echo "  - View per-patient metrics: find $PIPELINE_DIR/*/per_patient_inference -name '*.json'"
    echo "  - Compare teacher vs student vs distilled performance"
    echo "  - Run fairness analysis on combined model results"
else
    echo "Next steps:"
    echo "  - Check logs: find $PIPELINE_DIR -name '*.log'"
    echo "  - View metrics: find $PIPELINE_DIR -name '*summary.json'"
    echo "  - Compare teacher vs student performance"
    echo "  - Run fairness analysis on combined model results"
fi
echo ""
echo "🎯 Ready to test on production data!"
