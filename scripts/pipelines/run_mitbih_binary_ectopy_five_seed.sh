#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PIPELINE_DIR="${PIPELINE_DIR:-experiments/mitbih_binary_ectopy_five_seed}"
SEEDS="831363,809906,427368,238822,247659"
EPOCHS="10"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
TEACHER_MODEL="BERT"
STUDENT_MODEL="TinyBERT"
AAMI5_ROOT="${AAMI5_ROOT:-experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712}"
BG_ROOT="${BG_ROOT:-distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17}"
KD_PROTOCOL_FILE="${KD_PROTOCOL_FILE:-experiments/mitbih_binary_ectopy_kd_grid/selected_kd_protocol.json}"
DRY_RUN="${DRY_RUN:-0}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

mkdir -p "$PIPELINE_DIR"
PIPELINE_DIR="$(cd "$PIPELINE_DIR" && pwd)"
RUN_LOG="$PIPELINE_DIR/five_seed_runner.log"
exec > >(tee -a "$RUN_LOG") 2>&1

check_disk() {
    local free_kb free_gb
    free_kb=$(df -Pk "$PIPELINE_DIR" | awk 'NR==2 {print $4}')
    free_gb=$((free_kb / 1024 / 1024))
    if (( free_gb < MIN_FREE_GB )); then
        echo "Insufficient free disk: ${free_gb}GB available, ${MIN_FREE_GB}GB required" >&2
        exit 1
    fi
}

load_kd_protocol() {
    [[ -f "$KD_PROTOCOL_FILE" ]] || {
        echo "Missing validation-selected KD protocol: $KD_PROTOCOL_FILE" >&2
        echo "Run scripts/pipelines/run_mitbih_binary_ectopy_kd_grid.sh first" >&2
        exit 1
    }
    read -r KD_ALPHA KD_BETA KD_TEMPERATURE < <(
        "$PYTHON_BIN" - "$KD_PROTOCOL_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("selection_split") != "validation":
    raise SystemExit("KD protocol was not selected on validation")
if payload.get("label_mode") != "binary_ectopy":
    raise SystemExit("KD protocol is not for binary_ectopy")
if payload.get("selection_rule", {}).get("test_metrics_used") is not False:
    raise SystemExit("KD protocol does not certify test-independent selection")
selected = payload["selected"]
if int(selected.get("n_seeds", 0)) < 1:
    raise SystemExit("KD protocol has no completed development seed")
print(selected["alpha"], selected["beta"], selected["temperature"])
PY
    )
}

write_protocol_manifest() {
    "$PYTHON_BIN" - "$PIPELINE_DIR/protocol_manifest.json" "$SEEDS" "$EPOCHS" \
        "$TEACHER_MODEL" "$STUDENT_MODEL" "$KD_ALPHA" "$KD_BETA" \
        "$KD_TEMPERATURE" "$KD_PROTOCOL_FILE" <<'PY'
import json
import os
import sys

path, seeds, epochs, teacher, student, alpha, beta, temperature, kd_protocol = sys.argv[1:]
payload = {
    "protocol": "mitbih_binary_ectopy_five_seed_v2",
    "seeds": [int(value) for value in seeds.split(",")],
    "epochs": int(epochs),
    "teacher_model": teacher,
    "student_model": student,
    "label_mode": "binary_ectopy",
    "class_mapping": {"0": "N", "1": "S/V/F"},
    "excluded_original_classes": ["Q"],
    "record_split": {"train": 28, "validation": 9, "test": 10},
    "sequence_length": 256,
    "patch_length": 16,
    "use_rr_features": True,
    "rr_fusion_weight": 1.0,
    "freeze_llm": True,
    "class_balanced_sampling": True,
    "class_balanced_max_oversample": 50.0,
    "fairness_feature": "sex",
    "fair_max_oversample": 4.0,
    "distillation_variants": ["baseline", "t1", "o2", "t1_o2"],
    "distillation_alpha": float(alpha),
    "distillation_beta": float(beta),
    "distillation_temperature": float(temperature),
    "kd_protocol_file": os.path.abspath(kd_protocol),
    "o2_learning_rate": 0.0001,
    "o2_scale_regularization": 0.0,
    "o2_bias_regularization": 0.0,
    "checkpoint_selection": "utility_fairness",
    "checkpoint_selection_split": "validation",
    "checkpoint_selection_fairness_classes": [1],
    "checkpoint_selection_min_positive_recall": 0.05,
    "checkpoint_selection_macro_f1_tolerance": 0.01,
    "min_teacher_macro_f1": 0.35,
    "min_teacher_predicted_classes": 2,
}
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        existing = json.load(handle)
    if existing != payload:
        raise SystemExit(
            "Protocol manifest mismatch; use a new PIPELINE_DIR or restore the locked settings"
        )
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

aggregate_results() {
    "$PYTHON_BIN" scripts/fairness/aggregate_mitbih_multiseed_classification.py \
        --pipeline-dir "$PIPELINE_DIR" \
        --label-mode binary_ectopy \
        --fairness-feature sex \
        --fairness-classes 1 \
        --seeds "$SEEDS"

    if [[ -f "$AAMI5_ROOT/multiseed_aggregate_summary.csv" \
        && -f "$BG_ROOT/multiseed_robustness_results.csv" ]]; then
        "$PYTHON_BIN" scripts/fairness/aggregate_bg_ecg_task_comparison.py \
            --bg-root "$BG_ROOT" \
            --aami5-root "$AAMI5_ROOT" \
            --binary-root "$PIPELINE_DIR" \
            --output-dir "$PIPELINE_DIR"
        "$PYTHON_BIN" scripts/fairness/aggregate_bg_ecg_comparison.py \
            --bg-root "$BG_ROOT" \
            --ecg-root "$AAMI5_ROOT" \
            --binary-root "$PIPELINE_DIR"
    else
        echo "Comparison skipped: canonical BG or AAMI-5 aggregate is missing" >&2
    fi
}

echo "MIT-BIH locked binary-ectopy five-seed run"
echo "pipeline_dir=$PIPELINE_DIR"
echo "seeds=$SEEDS epochs=$EPOCHS"
echo "teacher=$TEACHER_MODEL student=$STUDENT_MODEL"
echo "kd_protocol_file=$KD_PROTOCOL_FILE"
echo "dry_run=$DRY_RUN aggregate_only=$AGGREGATE_ONLY"
echo "runner_log=$RUN_LOG"

check_disk
load_kd_protocol
echo "selected_kd=alpha:$KD_ALPHA beta:$KD_BETA temperature:$KD_TEMPERATURE"
write_protocol_manifest

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN complete: locked protocol manifest validated; no training started"
    exit 0
fi

if [[ "$AGGREGATE_ONLY" == "1" ]]; then
    aggregate_results
    exit 0
fi

if [[ ! -f "$PIPELINE_DIR/.training_complete" ]]; then
    env \
        PIPELINE_DIR="$PIPELINE_DIR" \
        SEEDS="$SEEDS" \
        TEACHER_MODEL="$TEACHER_MODEL" \
        STUDENT_MODEL="$STUDENT_MODEL" \
        TEACHER_EPOCHS="$EPOCHS" \
        STUDENT_EPOCHS="$EPOCHS" \
        DISTILL_EPOCHS="$EPOCHS" \
        LABEL_MODE=binary_ectopy \
        USE_RR_FEATURES=1 \
        RR_FUSION_WEIGHT=1.0 \
        SEQUENCE_LENGTH=256 \
        PATCH_LENGTH=16 \
        BEAT_INDEX_CSV=./data/mit-bih-arrhythmia/beat_index.csv \
        FAIR_FEATURE=sex \
        CLASS_BALANCED=1 \
        CLASS_BALANCED_MAX_OVERSAMPLE=50.0 \
        FAIR_MAX_OVERSAMPLE=4.0 \
        FREEZE_LLM=1 \
        LEARNING_RATE=0.0001 \
        BACKBONE_LEARNING_RATE=0.00001 \
        KD_ALPHA="$KD_ALPHA" \
        KD_BETA="$KD_BETA" \
        KD_TEMPERATURE="$KD_TEMPERATURE" \
        O2_LEARNING_RATE=0.0001 \
        O2_SCALE_REGULARIZATION=0.0 \
        O2_BIAS_REGULARIZATION=0.0 \
        DISTILL_VARIANTS=baseline,t1,o2,t1_o2 \
        CHECKPOINT_SELECTION=utility_fairness \
        SELECTION_FAIRNESS_CLASSES=1 \
        SELECTION_MIN_S_RECALL=0.05 \
        SELECTION_MACRO_F1_TOLERANCE=0.01 \
        SELECTION_REQUIRE_S_RECALL=0 \
        MIN_TEACHER_MACRO_F1=0.35 \
        MIN_TEACHER_PREDICTED_CLASSES=2 \
        bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
    for seed in ${SEEDS//,/ }; do
        seed_dir="$PIPELINE_DIR/seed_${seed}"
        [[ -f "$seed_dir/fairness_comparison_sex.json" ]] || {
            echo "Missing final fairness report for seed $seed" >&2
            exit 1
        }
        for directory in teacher_gen student_baseline_gen distill_baseline_gen distill_t1_gen distill_o2_gen distill_t1_o2_gen; do
            if [[ -z "$(find "$seed_dir/$directory" -name test_predictions.csv -print -quit)" ]]; then
                echo "Missing test predictions for seed=$seed directory=$directory" >&2
                exit 1
            fi
            if [[ -z "$(find "$seed_dir/$directory" -name val_predictions.csv -print -quit)" ]]; then
                echo "Missing validation predictions for seed=$seed directory=$directory" >&2
                exit 1
            fi
        done
    done
    touch "$PIPELINE_DIR/.training_complete"
else
    echo "SKIP training: .training_complete marker exists"
fi

aggregate_results
touch "$PIPELINE_DIR/.complete"
echo "Binary-ectopy five-seed run complete"
echo "Binary report: $PIPELINE_DIR/MULTISEED_ANALYSIS.md"
echo "Three-task report: $PIPELINE_DIR/BG_AAMI5_BINARY_COMPARISON.md"
