#!/bin/bash
# Complete 3-Step Knowledge Distillation Pipeline
# Step 1: Train teacher model
# Step 2: Train student model (baseline)  
# Step 3: Knowledge distillation (student learns from teacher)

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

# Record pipeline start time for runtime calculation
PIPELINE_START_TIME=$(date +%s)

echo "🧠 Time-LLM Knowledge Distillation Pipeline"
echo "============================================="

# Parse command line arguments
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
        --patients)
            PATIENTS="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --no-checkpoints)
            NO_CHECKPOINTS="--remove_checkpoints"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --teacher <model> --student <model> --patients <patient_ids> --dataset <dataset_name> --seed <seed> --teacher-epochs <n> --student-epochs <n> --distill-epochs <n> [--lr <rate>] [--batch-size <size>] [--alpha <weight>] [--beta <weight>] [--dry-run] [--no-checkpoints]"
            exit 1
            ;;
    esac
done

# Set defaults
TEACHER=${TEACHER:-bert}
STUDENT=${STUDENT:-tinybert}
PATIENTS=${PATIENTS:-570}
DATASET_NAME=${DATASET_NAME:-ohiot1dm}
SEED=${SEED:-238822}

# Auto-detect data path based on dataset name
DATA_PATH="./data/${DATASET_NAME}"
TEACHER_EPOCHS=${TEACHER_EPOCHS:-1}
STUDENT_EPOCHS=${STUDENT_EPOCHS:-1}
DISTILL_EPOCHS=${DISTILL_EPOCHS:-1}
LR=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-32}
ALPHA=${ALPHA:-0.5}
BETA=${BETA:-0.5}
NO_CHECKPOINTS=${NO_CHECKPOINTS:-""}
DATA_TYPE="raw_standardized"

# Create organized pipeline directory structure
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PIPELINE_DIR="distillation_experiments/pipeline_runs/pipeline_${TIMESTAMP}"
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"

# Validate required parameters
if [[ -z "$TEACHER" || -z "$STUDENT" || -z "$PATIENTS" ]]; then
    echo "❌ Missing required parameters!"
    echo "Usage: $0 --teacher <model> --student <model> --patients <patient_ids> --dataset <dataset_name> --seed <seed> --teacher-epochs <n> --student-epochs <n> --distill-epochs <n> [--lr <rate>] [--batch-size <size>] [--alpha <weight>] [--beta <weight>] [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset ohiot1dm --seed 42 --teacher-epochs 1 --student-epochs 1 --distill-epochs 1"
    echo "  $0 --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570,584 --dataset ohiot1dm --seed 42 --teacher-epochs 1 --student-epochs 1 --distill-epochs 1"
    echo ""
    exit 1
fi

echo "Starting 3-Step Knowledge Distillation Pipeline:"
echo "  Patient IDs: $PATIENTS"
echo "  Dataset Name: $DATASET_NAME"
echo "  Data Path: $DATA_PATH"
echo "  Seed: $SEED"
echo "  Teacher Model: $TEACHER"
echo "  Student Model: $STUDENT"
echo "  Teacher Epochs: $TEACHER_EPOCHS"
echo "  Student Epochs: $STUDENT_EPOCHS"
echo "  Distill Epochs: $DISTILL_EPOCHS"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Distillation Alpha: $ALPHA"
echo "  Distillation Beta: $BETA"
echo "  Pipeline Directory: $PIPELINE_DIR"

# Create main pipeline directory
mkdir -p "$PIPELINE_DIR"

# Convert comma-separated patients to array
IFS=',' read -ra PATIENT_ARRAY <<< "$PATIENTS"

echo ""
echo "=========================================="
echo "🧠 Multi-Patient Pipeline Execution"
echo "=========================================="
echo "Total patients to process: ${#PATIENT_ARRAY[@]}"
echo "Patients: $(IFS=','; echo "${PATIENT_ARRAY[*]}")"
echo ""

# Loop through each patient and run complete 3-phase pipeline
for CURRENT_PATIENT in "${PATIENT_ARRAY[@]}"; do
    echo "=================================================="
    echo "🔄 Processing Patient: $CURRENT_PATIENT"
    echo "=================================================="
    
    # Create patient-specific directories
    PATIENT_PIPELINE_DIR="$PIPELINE_DIR/patient_$CURRENT_PATIENT"
    PHASE1_DIR="$PATIENT_PIPELINE_DIR/phase_1_teacher"
    PHASE2_DIR="$PATIENT_PIPELINE_DIR/phase_2_student"
    PHASE3_DIR="$PATIENT_PIPELINE_DIR/phase_3_distillation"
    
    # Create patient-specific directories
    mkdir -p "$PHASE1_DIR" "$PHASE2_DIR" "$PHASE3_DIR"
    
    echo "Patient Directory: $PATIENT_PIPELINE_DIR"
    echo ""
    
    # PHASE 1: Teacher Training
    echo "=========================================="
    echo "🎓 Phase 1/3: Training Teacher Model ($TEACHER)"
    echo "=========================================="
    echo "Training $TEACHER on patient $CURRENT_PATIENT for $TEACHER_EPOCHS epochs..."
    echo "Output directory: $PHASE1_DIR"
    python distillation/scripts/train_teachers.py --model $TEACHER --patients $CURRENT_PATIENT --dataset $DATASET_NAME --seed $SEED --lr $LR --batch-size $BATCH_SIZE --epochs $TEACHER_EPOCHS --output-dir "$PHASE1_DIR" --config-dir "$PHASE1_DIR" $([ "$NO_CHECKPOINTS" != "" ] && echo "--remove-checkpoints")
    if [ $? -ne 0 ]; then
        echo "❌ Teacher training failed for patient $CURRENT_PATIENT!"
        exit 1
    fi
    echo "✅ Teacher training completed for patient $CURRENT_PATIENT"
    echo ""
    
    # PHASE 2: Student Baseline Training
    echo "=========================================="
    echo "👨‍🎓 Phase 2/3: Training Student Baseline ($STUDENT)"
    echo "=========================================="
    echo "Training baseline $STUDENT on patient $CURRENT_PATIENT for $STUDENT_EPOCHS epochs..."
    echo "Output directory: $PHASE2_DIR"
    python distillation/scripts/train_students.py \
        --model $STUDENT \
        --patients $CURRENT_PATIENT \
        --dataset $DATASET_NAME \
        --seed $SEED \
        --lr $LR \
        --batch-size $BATCH_SIZE \
        --epochs $STUDENT_EPOCHS \
        --output-dir "$PHASE2_DIR" \
        --config-dir "$PHASE2_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Student baseline training failed for patient $CURRENT_PATIENT!"
        exit 1
    fi
    echo "✅ Student baseline training completed for patient $CURRENT_PATIENT"
    echo ""
    
    # PHASE 3: Knowledge Distillation
    echo "=========================================="
    echo "🧠 Phase 3/3: Knowledge Distillation ($TEACHER → $STUDENT)"
    echo "=========================================="
    echo "Distilling knowledge from $TEACHER to $STUDENT for patient $CURRENT_PATIENT ($DISTILL_EPOCHS epochs)..."
    echo "Output directory: $PHASE3_DIR"
    python distillation/scripts/distill_students.py \
        --teacher $TEACHER \
        --student $STUDENT \
        --patients $CURRENT_PATIENT \
        --dataset $DATASET_NAME \
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
        --pipeline-dir "$PATIENT_PIPELINE_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Knowledge distillation failed for patient $CURRENT_PATIENT!"
        exit 1
    fi
    echo "✅ Knowledge distillation completed for patient $CURRENT_PATIENT"
    echo ""
    
    # Calculate runtime for this patient
    PATIENT_END_TIME=$(date +%s)
    PATIENT_RUNTIME=$((PATIENT_END_TIME - PIPELINE_START_TIME))
    
    # Log this patient's results to CSV
    echo "📊 Logging patient $CURRENT_PATIENT results to CSV..."
    python distillation/scripts/pipeline_csv_logger.py \
        --pipeline-dir "$PATIENT_PIPELINE_DIR" \
        --patients "$CURRENT_PATIENT" \
        --dataset "$DATASET_NAME" \
        --seed "$SEED" \
        --teacher "$TEACHER" \
        --student "$STUDENT" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --teacher-epochs "$TEACHER_EPOCHS" \
        --student-epochs "$STUDENT_EPOCHS" \
        --distill-epochs "$DISTILL_EPOCHS" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --teacher-metrics "$PHASE1_DIR/teacher_training_summary.json" \
        --student-metrics "$PHASE2_DIR/student_baseline_summary.json" \
        --distillation-metrics "$PHASE3_DIR/distillation_summary.json" \
        --total-runtime "$PATIENT_RUNTIME" \
        --notes "Patient $CURRENT_PATIENT complete 3-phase pipeline run"
    
    if [ $? -eq 0 ]; then
        echo "✅ Patient $CURRENT_PATIENT results successfully logged to CSV!"
    else
        echo "⚠️  CSV logging failed for patient $CURRENT_PATIENT, but pipeline completed successfully"
    fi
    
    echo ""
    echo "🎉 Patient $CURRENT_PATIENT: Complete 3-Phase Pipeline Finished!"
    echo "✅ Teacher trained: $TEACHER ($TEACHER_EPOCHS epochs)"
    echo "✅ Student baseline: $STUDENT ($STUDENT_EPOCHS epochs)"  
    echo "✅ Knowledge distilled: $TEACHER → $STUDENT ($DISTILL_EPOCHS epochs)"
    echo "📁 Results saved in: $PATIENT_PIPELINE_DIR"
    echo ""
    
done

# Calculate total pipeline runtime
PIPELINE_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo ""
echo "🎉🎉 SUCCESS: Multi-Patient Pipeline Completed! 🎉🎉"
echo "============================================="
echo "✅ Total patients processed: ${#PATIENT_ARRAY[@]}"
echo "✅ Patients: $(IFS=','; echo "${PATIENT_ARRAY[*]}")"
echo "✅ Each patient completed all 3 phases successfully"
echo ""
echo "📁 Results saved in: $PIPELINE_DIR"
echo "📊 CSV Results: distillation_experiments/pipeline_results.csv"
echo "📊 Total Pipeline Runtime: ${TOTAL_RUNTIME}s"