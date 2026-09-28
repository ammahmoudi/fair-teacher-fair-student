#!/bin/bash
# ============================================================================
# MIT-BIH ECG Time-LLM Fairness Distillation Pipeline
# ============================================================================
# This is the ECG-classification analogue of
# scripts/pipelines/run_fairness_distillation_experiments.sh (the BG pipeline).
#
# Runs, end to end, on a fresh checkout / fresh machine:
#   Phase 0: data prep        (metadata/demographics/beat-index CSVs) — once
#   Phase 1: teacher training (time_llm_ecg_classifier, e.g. BERT)     \
#   Phase 2: student baseline (time_llm_ecg_classifier, no distillation) } per seed
#   Phase 3: distillation variants (distillation_ecg_classifier):       |
#              - baseline (no fixes)                                   |
#              - T1 fair teacher sampling                              |
#              - O2 student calibration                                |
#              - T1 + O2 (best combo in the BG work)                   |
#              - optionally (RUN_ALL_FIXES=1): K1, O1, K3, K4, O3      |
#   Phase 4: fairness comparison across every trained variant         /
#
# Every phase is skipped automatically if its checkpoint + predictions already
# exist, so the script is safe to re-run/resume after an interruption.
#
# Usage:
#   cd /path/to/fair-teacher-fair-student
#   bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
#
# Configuration is via environment variables (all optional, shown with
# defaults below). Example overriding a few:
#   TEACHER_MODEL=BERT STUDENT_MODEL=BERT-tiny TEACHER_EPOCHS=15 \
#     bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
#
# Seed sweeps: by default a single SEED=42 is used. To sweep across seeds:
#   SEEDS=42,123,456 bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
# Or to use every seed from scripts/utilities/seeds.py::fixed_seeds:
#   ALL_SEEDS=1 bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
# Each seed gets its own subdirectory under PIPELINE_DIR (seed_<value>/), with
# its own checkpoints/predictions/fairness report, so results are not mixed.
#
# To bootstrap a brand-new machine (create venv + install requirements) before
# running the pipeline, set SETUP_ENV=1:
#   SETUP_ENV=1 bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
#
# All output is tee'd to a timestamped log file under the pipeline directory.
# ============================================================================

set -euo pipefail

# ── Resolve project root and cd into it ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration (override via environment variables) ──────────────────────
TEACHER_MODEL="${TEACHER_MODEL:-BERT}"
STUDENT_MODEL="${STUDENT_MODEL:-TinyBERT}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-}"                                     # comma-separated seed list, overrides SEED
ALL_SEEDS="${ALL_SEEDS:-0}"                            # 1 = use every seed in utilities/seeds.py::fixed_seeds
TEACHER_EPOCHS="${TEACHER_EPOCHS:-10}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-10}"
FAIR_FEATURE="${FAIR_FEATURE:-sex}"
LABEL_MODE="${LABEL_MODE:-aami5}"
USE_RR_FEATURES="${USE_RR_FEATURES:-0}"
RR_FUSION_WEIGHT="${RR_FUSION_WEIGHT:-1.0}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
PATCH_LENGTH="${PATCH_LENGTH:-16}"
BEAT_INDEX_CSV="${BEAT_INDEX_CSV:-./data/mit-bih-arrhythmia/beat_index.csv}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
INCLUDE_DUPLICATE_202="${INCLUDE_DUPLICATE_202:-0}"   # 1 = include record 202
PIPELINE_DIR="${PIPELINE_DIR:-experiments/mitbih_fairness_pipeline}"
RUN_ALL_FIXES="${RUN_ALL_FIXES:-0}"                    # 1 = also run K1/O1/K3/K4/O3 individually
DISTILL_VARIANTS="${DISTILL_VARIANTS:-baseline,t1,o2,t1_o2}"
SETUP_ENV="${SETUP_ENV:-0}"                            # 1 = create venv + pip install first
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CLASS_BALANCED="${CLASS_BALANCED:-1}"                   # 1 = class-balanced oversampling for teacher/student
                                                        # training (default ON: MIT-BIH's extreme AAMI class
                                                        # imbalance otherwise causes majority-class collapse)
CLASS_BALANCED_MAX_OVERSAMPLE="${CLASS_BALANCED_MAX_OVERSAMPLE:-50.0}"
FAIR_MAX_OVERSAMPLE="${FAIR_MAX_OVERSAMPLE:-4.0}"       # T1 group/class sampling cap
FREEZE_LLM="${FREEZE_LLM:-1}"                         # 1 = BG-style frozen pretrained backbone
LEARNING_RATE="${LEARNING_RATE:-0.0001}"              # ECG patch/projection/classifier modules
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-0.00001}" # only used when FREEZE_LLM=0
O2_LEARNING_RATE="${O2_LEARNING_RATE:-$LEARNING_RATE}"
O2_SCALE_REGULARIZATION="${O2_SCALE_REGULARIZATION:-0.0}"
O2_BIAS_REGULARIZATION="${O2_BIAS_REGULARIZATION:-0.0}"
KD_ALPHA="${KD_ALPHA:-0.5}"
KD_BETA="${KD_BETA:-0.5}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-loss}"
SELECTION_MIN_S_RECALL="${SELECTION_MIN_S_RECALL:-0.05}"
SELECTION_MACRO_F1_TOLERANCE="${SELECTION_MACRO_F1_TOLERANCE:-0.01}"
SELECTION_REQUIRE_S_RECALL="${SELECTION_REQUIRE_S_RECALL:-0}"
MIN_TEACHER_MACRO_F1="${MIN_TEACHER_MACRO_F1:-0.35}"
DEFAULT_MIN_TEACHER_PREDICTED_CLASSES=4
DEFAULT_SELECTION_FAIRNESS_CLASSES="0,2"
if [[ "$LABEL_MODE" != "aami5" ]]; then
    DEFAULT_MIN_TEACHER_PREDICTED_CLASSES=2
    DEFAULT_SELECTION_FAIRNESS_CLASSES="1"
fi
MIN_TEACHER_PREDICTED_CLASSES="${MIN_TEACHER_PREDICTED_CLASSES:-$DEFAULT_MIN_TEACHER_PREDICTED_CLASSES}"
SELECTION_FAIRNESS_CLASSES="${SELECTION_FAIRNESS_CLASSES:-$DEFAULT_SELECTION_FAIRNESS_CLASSES}"

DUP_202_FLAG=""
if [[ "$INCLUDE_DUPLICATE_202" == "1" ]]; then
    DUP_202_FLAG="--include-duplicate-202"
fi

CLASS_BALANCED_FLAG=""
if [[ "$CLASS_BALANCED" == "1" ]]; then
    CLASS_BALANCED_FLAG="--class-balanced --class-balanced-max-oversample $CLASS_BALANCED_MAX_OVERSAMPLE"
fi

UNFREEZE_LLM_FLAG=""
if [[ "$FREEZE_LLM" == "0" ]]; then
    UNFREEZE_LLM_FLAG="--unfreeze-llm"
fi

RR_FLAG=""
DISTILL_RR_FLAG=""
if [[ "$USE_RR_FEATURES" == "1" ]]; then
    RR_FLAG="--use-rr-features --rr-fusion-weight $RR_FUSION_WEIGHT"
    DISTILL_RR_FLAG="--use-rr-features --teacher-use-rr-features --rr-fusion-weight $RR_FUSION_WEIGHT"
fi

SELECTION_REQUIRE_S_FLAG=""
if [[ "$SELECTION_REQUIRE_S_RECALL" == "1" ]]; then
    SELECTION_REQUIRE_S_FLAG="--checkpoint-selection-require-s-recall"
fi

mkdir -p "$PIPELINE_DIR"
LOG_FILE="$PIPELINE_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Optional environment bootstrap for a brand-new machine ──────────────────
if [[ "$SETUP_ENV" == "1" ]]; then
    echo "🧰 SETUP_ENV=1 — bootstrapping Python environment..."
    if [[ ! -d venv ]]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Environment ready: $(python3 --version), venv=$VIRTUAL_ENV"
    echo ""
fi

# ── Activate venv if present (matches scripts/run_main.sh's own logic) ──────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d venv ]]; then
        source venv/bin/activate
    elif [[ -d .venv ]]; then
        source .venv/bin/activate
    fi
fi

# ── Resolve which seed(s) to run ────────────────────────────────────────────
SEED_LIST=()
if [[ "$ALL_SEEDS" == "1" ]]; then
    ALL_SEEDS_CSV=$(cd "$PROJECT_ROOT/scripts" && python3 -c "from utilities.seeds import fixed_seeds; print(','.join(map(str, fixed_seeds)))")
    IFS=',' read -r -a SEED_LIST <<< "$ALL_SEEDS_CSV"
elif [[ -n "$SEEDS" ]]; then
    IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
else
    SEED_LIST=("$SEED")
fi

echo "========================================================================"
echo "🔬 MIT-BIH ECG Fairness Distillation Pipeline"
echo "   Started: $(date)"
echo "   Project root: $PROJECT_ROOT"
echo "   Pipeline dir:  $PIPELINE_DIR"
echo "   Log file:      $LOG_FILE"
echo "   venv:          ${VIRTUAL_ENV:-<none, using system python>}"
echo "   Teacher model: $TEACHER_MODEL (epochs=$TEACHER_EPOCHS)"
echo "   Student model: $STUDENT_MODEL (epochs=$STUDENT_EPOCHS, distill_epochs=$DISTILL_EPOCHS)"
echo "   Fair feature:  $FAIR_FEATURE"
echo "   Label mode:    $LABEL_MODE"
echo "   RR features:   $USE_RR_FEATURES (fusion_weight=$RR_FUSION_WEIGHT)"
echo "   ECG context:   sequence=$SEQUENCE_LENGTH, patch=$PATCH_LENGTH, index=$BEAT_INDEX_CSV"
echo "   Run all fixes: $RUN_ALL_FIXES"
echo "   Distillation variants: $DISTILL_VARIANTS"
echo "   Class-balanced sampling (teacher/student): $CLASS_BALANCED (max_oversample=$CLASS_BALANCED_MAX_OVERSAMPLE)"
echo "   T1 group/class max oversample: $FAIR_MAX_OVERSAMPLE"
echo "   O2 calibration: lr=$O2_LEARNING_RATE, scale_reg=$O2_SCALE_REGULARIZATION, bias_reg=$O2_BIAS_REGULARIZATION"
echo "   KD objective: alpha=$KD_ALPHA, beta=$KD_BETA, temperature=$KD_TEMPERATURE"
echo "   Checkpoint selection: $CHECKPOINT_SELECTION (min_s_recall=$SELECTION_MIN_S_RECALL, macro_f1_tolerance=$SELECTION_MACRO_F1_TOLERANCE, require_s=$SELECTION_REQUIRE_S_RECALL)"
echo "   LLM backbone frozen: $FREEZE_LLM (task_lr=$LEARNING_RATE, unfrozen_backbone_lr=$BACKBONE_LEARNING_RATE)"
echo "   Teacher acceptance: macro_f1 >= $MIN_TEACHER_MACRO_F1, predicted_classes >= $MIN_TEACHER_PREDICTED_CLASSES"
echo "   Seeds:         ${SEED_LIST[*]} (${#SEED_LIST[@]} total)"
echo "========================================================================"
echo ""

# ── Helper: run a single generated config if not already trained ───────────
# Expects the exact experiment folder (the one containing dataset_mitbih/).
# NOTE: every run of main.py creates a fresh timestamped subdirectory
# (logs/logs_<timestamp>/) via utils/logger.py's setup_logging(), so
# checkpoint/prediction paths are NOT fixed — always glob for the latest one.
latest_checkpoint_path() {
    find "$1/dataset_mitbih/logs" -path "*/checkpoints/checkpoint_best.pth" 2>/dev/null | sort | tail -1
}

latest_predictions_path() {
    find "$1/dataset_mitbih/logs" -maxdepth 2 -name "test_predictions.csv" 2>/dev/null | sort | tail -1
}

latest_val_predictions_path() {
    find "$1/dataset_mitbih/logs" -maxdepth 2 -name "val_predictions.csv" 2>/dev/null | sort | tail -1
}

run_experiment_if_needed() {
    local exp_dir="$1"
    local label="$2"
    local config_path="$exp_dir/dataset_mitbih/config.gin"
    local ckpt
    ckpt=$(latest_checkpoint_path "$exp_dir")
    local preds
    preds=$(latest_predictions_path "$exp_dir")

    if [[ -n "$ckpt" && -n "$preds" ]]; then
        echo "⏭️  Skipping $label — already trained:"
        echo "     checkpoint:  $ckpt"
        echo "     predictions: $preds"
        return 0
    fi

    if [[ ! -f "$config_path" ]]; then
        echo "❌ Config not found for $label: $config_path"
        exit 1
    fi

    echo "▶ Running $label"
    echo "   Config: $config_path"
    ./scripts/run_main.sh --config_path "$config_path" --log_level "$LOG_LEVEL" --remove_checkpoints False

    ckpt=$(latest_checkpoint_path "$exp_dir")
    if [[ -z "$ckpt" ]]; then
        echo "❌ $label finished but no checkpoint was found under $exp_dir/dataset_mitbih/logs"
        exit 1
    fi
    echo "✅ $label complete"
    echo ""
}

validate_teacher_predictions() {
    local predictions_path="$1"
    python3 - "$predictions_path" "$MIN_TEACHER_MACRO_F1" "$MIN_TEACHER_PREDICTED_CLASSES" <<'PY'
import sys

import pandas as pd
from sklearn.metrics import f1_score, recall_score

predictions_path, min_macro_f1, min_predicted_classes = sys.argv[1:]
min_macro_f1 = float(min_macro_f1)
min_predicted_classes = int(min_predicted_classes)
predictions = pd.read_csv(predictions_path)
required_columns = {"y_true", "y_pred"}
missing_columns = required_columns.difference(predictions.columns)
if missing_columns:
    raise SystemExit(f"Teacher acceptance failed: missing columns {sorted(missing_columns)}")

y_true = predictions["y_true"].to_numpy()
y_pred = predictions["y_pred"].to_numpy()
labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
predicted_classes = sorted(set(y_pred.tolist()))
recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
recall_summary = {int(label): round(float(recall), 4) for label, recall in zip(labels, recalls)}
print(
    f"Teacher acceptance metrics: macro_f1={macro_f1:.6f}, "
    f"predicted_classes={predicted_classes}, class_recalls={recall_summary}"
)
if macro_f1 < min_macro_f1 or len(predicted_classes) < min_predicted_classes:
    raise SystemExit(
        "Teacher acceptance failed: refusing to run distillation with a collapsed or weak teacher "
        f"(required macro_f1 >= {min_macro_f1} and at least {min_predicted_classes} predicted classes)."
    )
PY
}

# ── Helper: return the selected sequence/patch experiment folder ────────────
locate_selected_experiment_dir() {
    local gen_dir="$1"
    local config_path
    config_path=$(find "$gen_dir" -name config.gin | grep "seq_${SEQUENCE_LENGTH}_patch_${PATCH_LENGTH}" | head -1)
    if [[ -z "$config_path" ]]; then
        echo "❌ No seq_${SEQUENCE_LENGTH}_patch_${PATCH_LENGTH} config generated under $gen_dir" >&2
        exit 1
    fi
    # config_path = <experiment_folder>/dataset_mitbih/config.gin
    dirname "$(dirname "$config_path")"
}

# ── Phase 0: data prep (idempotent, shared across all seeds) ───────────────
echo "========================================================================"
echo "▶ Phase 0: MIT-BIH data preparation"
echo "========================================================================"
DATA_DIR="data/mit-bih-arrhythmia"
if [[ -f "$DATA_DIR/metadata_records.csv" && -f "$DATA_DIR/demographics_records.csv" ]]; then
    echo "⏭️  Metadata CSVs already present in $DATA_DIR"
else
    python scripts/mitbih/build_metadata.py $DUP_202_FLAG
    python scripts/mitbih/build_demographics.py $DUP_202_FLAG
fi
python scripts/mitbih/prepare_beat_dataset.py $DUP_202_FLAG \
    --window-size "$SEQUENCE_LENGTH" --output-path "$BEAT_INDEX_CSV" \
    --resplit-existing --split-strategy stratified
if [[ -f "$PIPELINE_DIR/data_split_manifest.json" ]]; then
    if ! cmp -s "$DATA_DIR/split_manifest.json" "$PIPELINE_DIR/data_split_manifest.json"; then
        echo "❌ Data split differs from the split recorded in $PIPELINE_DIR; use a fresh PIPELINE_DIR"
        exit 1
    fi
elif compgen -G "$PIPELINE_DIR/seed_*" > /dev/null; then
    echo "❌ Existing seed outputs have no split manifest; use a fresh PIPELINE_DIR to avoid split leakage"
    exit 1
else
    cp "$DATA_DIR/split_manifest.json" "$PIPELINE_DIR/data_split_manifest.json"
fi
echo "✅ Phase 0 complete — $(date)"
echo ""

# ── Phases 1-4, run once per seed ───────────────────────────────────────────
run_pipeline_for_seed() {
    local seed="$1"
    local seed_dir="$PIPELINE_DIR/seed_${seed}"
    mkdir -p "$seed_dir"

    echo "########################################################################"
    echo "# Seed $seed  (output: $seed_dir)"
    echo "########################################################################"
    echo ""

    # Phase 1: teacher training
    echo "========================================================================"
    echo "▶ Phase 1/4 [seed $seed]: Teacher training ($TEACHER_MODEL)"
    echo "========================================================================"
    local teacher_gen_dir="$seed_dir/teacher_gen"
    python scripts/time_llm/config_generator_mitbih.py \
        --mode train_inference --llm_models "$TEACHER_MODEL" --seeds "$seed" \
        --epochs "$TEACHER_EPOCHS" --torch-dtype "$TORCH_DTYPE" $DUP_202_FLAG $CLASS_BALANCED_FLAG $UNFREEZE_LLM_FLAG \
        --learning-rate "$LEARNING_RATE" --backbone-learning-rate "$BACKBONE_LEARNING_RATE" \
        --label-mode "$LABEL_MODE" $RR_FLAG \
        --beat-index-csv "$BEAT_INDEX_CSV" --length-configs "$SEQUENCE_LENGTH:$PATCH_LENGTH" \
        --output_dir "$teacher_gen_dir"
    local teacher_exp_dir
    teacher_exp_dir=$(locate_selected_experiment_dir "$teacher_gen_dir")
    run_experiment_if_needed "$teacher_exp_dir" "teacher ($TEACHER_MODEL) [seed $seed]"
    local teacher_ckpt
    teacher_ckpt=$(latest_checkpoint_path "$teacher_exp_dir")
    local teacher_predictions
    teacher_predictions=$(latest_predictions_path "$teacher_exp_dir")
    local teacher_val_predictions
    teacher_val_predictions=$(latest_val_predictions_path "$teacher_exp_dir")
    echo "   Teacher checkpoint:  $teacher_ckpt"
    echo "   Teacher predictions: $teacher_predictions"
    echo "   Teacher validation predictions: $teacher_val_predictions"
    if [[ -z "$teacher_val_predictions" ]]; then
        echo "❌ Teacher validation predictions were not found; refusing to gate on the test set"
        exit 1
    fi
    validate_teacher_predictions "$teacher_val_predictions"
    echo "✅ Teacher acceptance gate passed"
    echo ""

    # Phase 2: student baseline (no distillation)
    echo "========================================================================"
    echo "▶ Phase 2/4 [seed $seed]: Student baseline training ($STUDENT_MODEL, no distillation)"
    echo "========================================================================"
    local student_gen_dir="$seed_dir/student_baseline_gen"
    python scripts/time_llm/config_generator_mitbih.py \
        --mode train_inference --llm_models "$STUDENT_MODEL" --seeds "$seed" \
        --epochs "$STUDENT_EPOCHS" --torch-dtype "$TORCH_DTYPE" $DUP_202_FLAG $CLASS_BALANCED_FLAG $UNFREEZE_LLM_FLAG \
        --learning-rate "$LEARNING_RATE" --backbone-learning-rate "$BACKBONE_LEARNING_RATE" \
        --label-mode "$LABEL_MODE" $RR_FLAG \
        --beat-index-csv "$BEAT_INDEX_CSV" --length-configs "$SEQUENCE_LENGTH:$PATCH_LENGTH" \
        --output_dir "$student_gen_dir"
    local student_exp_dir
    student_exp_dir=$(locate_selected_experiment_dir "$student_gen_dir")
    run_experiment_if_needed "$student_exp_dir" "student baseline ($STUDENT_MODEL) [seed $seed]"
    local student_predictions
    student_predictions=$(latest_predictions_path "$student_exp_dir")
    echo "   Student baseline predictions: $student_predictions"
    echo ""

    # Phase 3: distillation variants
    echo "========================================================================"
    echo "▶ Phase 3/4 [seed $seed]: Distillation variants ($TEACHER_MODEL -> $STUDENT_MODEL)"
    echo "========================================================================"

    declare -A distill_predictions

    _run_distillation_variant() {
        local label="$1"; shift
        local extra_flags=("$@")
        local gen_dir="$seed_dir/distill_${label}_gen"

        python scripts/time_llm/config_generator_mitbih_distillation.py \
            --mode train_inference --teacher-model "$TEACHER_MODEL" \
            --student-models "$STUDENT_MODEL" --teacher-checkpoint-path "$teacher_ckpt" \
            --seeds "$seed" --epochs "$DISTILL_EPOCHS" --torch-dtype "$TORCH_DTYPE" \
            $DUP_202_FLAG $CLASS_BALANCED_FLAG $UNFREEZE_LLM_FLAG \
            --learning-rate "$LEARNING_RATE" --backbone-learning-rate "$BACKBONE_LEARNING_RATE" \
            --label-mode "$LABEL_MODE" $DISTILL_RR_FLAG \
            --beat-index-csv "$BEAT_INDEX_CSV" --length-configs "$SEQUENCE_LENGTH:$PATCH_LENGTH" \
            --alpha "$KD_ALPHA" --beta "$KD_BETA" --temperature "$KD_TEMPERATURE" \
            --fair-teacher-max-oversample "$FAIR_MAX_OVERSAMPLE" \
            --student-calibration-learning-rate "$O2_LEARNING_RATE" \
            --student-calibration-scale-regularization "$O2_SCALE_REGULARIZATION" \
            --student-calibration-bias-regularization "$O2_BIAS_REGULARIZATION" \
            --checkpoint-selection "$CHECKPOINT_SELECTION" \
            --checkpoint-selection-feature "$FAIR_FEATURE" \
            --checkpoint-selection-fairness-classes "$SELECTION_FAIRNESS_CLASSES" \
            --checkpoint-selection-min-s-recall "$SELECTION_MIN_S_RECALL" \
            --checkpoint-selection-macro-f1-tolerance "$SELECTION_MACRO_F1_TOLERANCE" \
            $SELECTION_REQUIRE_S_FLAG \
            --output_dir "$gen_dir" "${extra_flags[@]}"

        local exp_dir
        exp_dir=$(locate_selected_experiment_dir "$gen_dir")
        run_experiment_if_needed "$exp_dir" "distillation: $label [seed $seed]"
        distill_predictions["$label"]=$(latest_predictions_path "$exp_dir")
    }

    _variant_enabled() {
        [[ ",$DISTILL_VARIANTS," == *",$1,"* ]]
    }

    if _variant_enabled "baseline"; then
        _run_distillation_variant "baseline"
    fi
    if _variant_enabled "t1"; then
        _run_distillation_variant "t1" --fair-teacher --fair-teacher-feature "$FAIR_FEATURE"
    fi
    if _variant_enabled "o2"; then
        _run_distillation_variant "o2" --student-calibration --student-calibration-feature "$FAIR_FEATURE"
    fi
    if _variant_enabled "t1_o2"; then
        _run_distillation_variant "t1_o2" \
            --fair-teacher --fair-teacher-feature "$FAIR_FEATURE" \
            --student-calibration --student-calibration-feature "$FAIR_FEATURE"
    fi

    if [[ "$RUN_ALL_FIXES" == "1" ]]; then
        echo "🧪 RUN_ALL_FIXES=1 — also running K1, O1, K3, K4, O3 individually"
        _run_distillation_variant "k1" \
            --teacher-calibration --teacher-calibration-feature "$FAIR_FEATURE" \
            --teacher-calibration-offsets '{"M": [0,0,0,0,0], "F": [0,0.3,0.3,0.3,0.3]}'
        _run_distillation_variant "o1" \
            --fairness-constraint --fairness-constraint-feature "$FAIR_FEATURE"
        _run_distillation_variant "k3" \
            --feature-alignment --feature-alignment-feature "$FAIR_FEATURE"
        _run_distillation_variant "k4" \
            --kd-replay --kd-replay-feature "$FAIR_FEATURE" --kd-replay-minority-group F
        _run_distillation_variant "o3" \
            --adv-erasure --adv-erasure-feature "$FAIR_FEATURE"
        # T2 (multi-teacher) needs a second, group-specialized teacher checkpoint,
        # which this pipeline does not train by default. Skipped unless you supply
        # SECOND_TEACHER_CHECKPOINT_PATH explicitly.
        if [[ -n "${SECOND_TEACHER_CHECKPOINT_PATH:-}" ]]; then
            _run_distillation_variant "t2" \
                --multi-teacher --multi-teacher-feature "$FAIR_FEATURE" --multi-teacher-group0 F \
                --second-teacher-checkpoint-path "$SECOND_TEACHER_CHECKPOINT_PATH"
        else
            echo "⏭️  Skipping T2 (multi-teacher) — set SECOND_TEACHER_CHECKPOINT_PATH to enable"
        fi
    fi

    echo "✅ Phase 3 complete [seed $seed] — $(date)"
    echo ""

    # Phase 4: fairness comparison across every trained variant
    echo "========================================================================"
    echo "▶ Phase 4/4 [seed $seed]: Fairness comparison"
    echo "========================================================================"

    local pred_csv_arg="teacher=$teacher_predictions,student_baseline=$student_predictions"
    for label in "${!distill_predictions[@]}"; do
        pred_csv_arg="$pred_csv_arg,distilled_${label}=${distill_predictions[$label]}"
    done

    local fairness_report="$seed_dir/fairness_comparison_${FAIR_FEATURE}.json"
    python scripts/fairness/run_mitbih_classifier_fairness.py \
        --prediction-csvs "$pred_csv_arg" \
        --group-column "$FAIR_FEATURE" \
        --output-json "$fairness_report"

    echo "✅ Phase 4 complete [seed $seed] — $(date)"
    echo ""

    echo "------------------------------------------------------------------------"
    echo "🎉 Seed $seed complete — $(date)"
    echo "   Teacher checkpoint:      $teacher_ckpt"
    echo "   Student baseline preds:  $student_predictions"
    for label in "${!distill_predictions[@]}"; do
        echo "   Distilled ($label) preds: ${distill_predictions[$label]}"
    done
    echo "   Fairness comparison:     $fairness_report"
    echo "------------------------------------------------------------------------"
    echo ""

    unset -f _run_distillation_variant
}

for seed in "${SEED_LIST[@]}"; do
    run_pipeline_for_seed "$seed"
done

echo "========================================================================"
echo "🎉🎉 All seeds complete — $(date)"
echo "   Seeds run: ${SEED_LIST[*]}"
echo "   Per-seed outputs under: $PIPELINE_DIR/seed_<seed>/"
echo "   Full log: $LOG_FILE"
echo "========================================================================"
