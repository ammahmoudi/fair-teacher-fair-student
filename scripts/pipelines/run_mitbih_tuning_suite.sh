#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PROFILE="${PROFILE:-full}"                    # screen | full
SUITE_DIR="${SUITE_DIR:-experiments/mitbih_tuning_suite_$(date +%Y%m%d_%H%M%S)}"
PHASES="${PHASES:-kd,t1,o2,rr,binary,age}"
CASES="${CASES:-}"                            # optional comma-separated case names
SUITE_SEEDS="${SUITE_SEEDS:-}"
SUITE_EPOCHS="${SUITE_EPOCHS:-}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
TEACHER_MODEL="${TEACHER_MODEL:-BERT}"
STUDENT_MODEL="${STUDENT_MODEL:-TinyBERT}"
AGGREGATION_SPLIT="${AGGREGATION_SPLIT:-validation}"
KD_SELECTION_MIN_SEEDS="${KD_SELECTION_MIN_SEEDS:-3}"

if [[ "$AGGREGATION_SPLIT" != "validation" && "$AGGREGATION_SPLIT" != "test" ]]; then
    echo "AGGREGATION_SPLIT must be validation or test" >&2
    exit 2
fi
export AGGREGATION_SPLIT

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if [[ "$PROFILE" == "screen" ]]; then
    SUITE_SEEDS="${SUITE_SEEDS:-831363}"
    SUITE_EPOCHS="${SUITE_EPOCHS:-2}"
elif [[ "$PROFILE" == "full" ]]; then
    SUITE_SEEDS="${SUITE_SEEDS:-831363,809906,427368}"
    SUITE_EPOCHS="${SUITE_EPOCHS:-10}"
else
    echo "PROFILE must be screen or full" >&2
    exit 2
fi

mkdir -p "$SUITE_DIR/cases"
SUITE_DIR="$(cd "$SUITE_DIR" && pwd)"
SUITE_LOG="$SUITE_DIR/suite.log"
exec > >(tee -a "$SUITE_LOG") 2>&1

phase_enabled() {
    [[ ",$PHASES," == *",$1,"* ]]
}

case_enabled() {
    [[ -z "$CASES" || ",$CASES," == *",$1,"* ]]
}

check_disk() {
    local free_kb free_gb
    free_kb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    free_gb=$((free_kb / 1024 / 1024))
    if (( free_gb < MIN_FREE_GB )); then
        echo "Insufficient free disk: ${free_gb}GB available, ${MIN_FREE_GB}GB required" >&2
        exit 1
    fi
}

write_manifest() {
    local case_dir="$1" case_name="$2" phase="$3" label_mode="$4"
    local fairness_feature="$5" sequence_length="$6" patch_length="$7" use_rr="$8"
    local fairness_classes="$9" variants="${10}"
    shift 10
    "$PYTHON_BIN" - "$case_dir/case_manifest.json" "$case_name" "$phase" \
        "$label_mode" "$fairness_feature" "$sequence_length" "$patch_length" \
        "$use_rr" "$fairness_classes" "$variants" "$SUITE_SEEDS" \
        "$SUITE_EPOCHS" "$TEACHER_MODEL" "$STUDENT_MODEL" "$@" <<'PY'
import json
import os
import sys

(
    path, case_name, phase, label_mode, feature, sequence, patch, rr, classes,
    variants, seeds, epochs, teacher_model, student_model, *assignments
) = sys.argv[1:]
variables = dict(item.split("=", 1) for item in assignments)
payload = {
    "case_name": case_name,
    "phase": phase,
    "aggregation_split": os.environ["AGGREGATION_SPLIT"],
    "label_mode": label_mode,
    "fairness_feature": feature,
    "sequence_length": int(sequence),
    "patch_length": int(patch),
    "use_rr_features": rr == "1",
    "fairness_classes": [int(value) for value in classes.split(",")],
    "distill_variants": [value for value in variants.split(",") if value],
    "seeds": [int(value) for value in seeds.split(",")],
    "epochs": int(epochs),
    "teacher_model": teacher_model,
    "student_model": student_model,
    "variables": variables,
}
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        existing = json.load(handle)
    if existing != payload:
        raise SystemExit(
            f"Case manifest mismatch for {case_name}; use a new SUITE_DIR or restore "
            "the original profile, seeds, models, and case settings"
        )
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY
}

aggregate_suite() {
    "$PYTHON_BIN" scripts/fairness/aggregate_mitbih_tuning_suite.py \
        --suite-dir "$SUITE_DIR"
}

run_case() {
    local case_name="$1" phase="$2" label_mode="$3" fairness_feature="$4"
    local sequence_length="$5" patch_length="$6" use_rr="$7" fairness_classes="$8"
    local variants="$9" beat_index_csv="${10}"
    shift 10
    local assignments=("$@")
    if ! case_enabled "$case_name"; then
        echo "SKIP filtered case: $case_name"
        return
    fi
    local case_dir="$SUITE_DIR/cases/$case_name"
    mkdir -p "$case_dir"
    write_manifest "$case_dir" "$case_name" "$phase" "$label_mode" \
        "$fairness_feature" "$sequence_length" "$patch_length" "$use_rr" \
        "$fairness_classes" "$variants" "${assignments[@]}"

    if [[ -f "$case_dir/.complete" ]]; then
        echo "SKIP complete case: $case_name"
        aggregate_suite
        return
    fi

    echo "========================================================================"
    echo "CASE $case_name | phase=$phase | seeds=$SUITE_SEEDS | epochs=$SUITE_EPOCHS"
    echo "========================================================================"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN env PIPELINE_DIR=%q ' "$case_dir"
        printf '%q ' "${assignments[@]}"
        echo
        return
    fi

    local reuse_case=""
    local pipeline_assignments=()
    local assignment
    for assignment in "${assignments[@]}"; do
        if [[ "$assignment" == REUSE_BASELINES_FROM_CASE=* ]]; then
            reuse_case="${assignment#*=}"
        else
            pipeline_assignments+=("$assignment")
        fi
    done
    if [[ -n "$reuse_case" ]]; then
        local source_case_dir source_split_manifest target_split_manifest
        local seed source_seed_dir target_seed_dir directory
        source_case_dir="$SUITE_DIR/cases/$reuse_case"
        source_split_manifest="$source_case_dir/data_split_manifest.json"
        target_split_manifest="$case_dir/data_split_manifest.json"
        [[ -f "$source_split_manifest" ]] || {
            echo "Cannot reuse baselines without $source_split_manifest" >&2
            exit 1
        }
        if [[ -f "$target_split_manifest" ]]; then
            cmp -s "$source_split_manifest" "$target_split_manifest" || {
                echo "Reused case split manifest differs from $reuse_case" >&2
                exit 1
            }
        else
            cp "$source_split_manifest" "$target_split_manifest"
        fi
        IFS=',' read -r -a reuse_seeds <<< "$SUITE_SEEDS"
        for seed in "${reuse_seeds[@]}"; do
            source_seed_dir="$SUITE_DIR/cases/$reuse_case/seed_$seed"
            target_seed_dir="$case_dir/seed_$seed"
            mkdir -p "$target_seed_dir"
            for directory in teacher_gen student_baseline_gen; do
                if [[ -L "$target_seed_dir/$directory" ]]; then
                    unlink "$target_seed_dir/$directory"
                fi
                if [[ ! -e "$target_seed_dir/$directory" ]]; then
                    [[ -d "$source_seed_dir/$directory" ]] || {
                        echo "Cannot reuse missing $source_seed_dir/$directory" >&2
                        exit 1
                    }
                    cp -a "$source_seed_dir/$directory" "$target_seed_dir/$directory"
                fi
            done
        done
    fi

    check_disk
    if env \
        PIPELINE_DIR="$case_dir" \
        SEEDS="$SUITE_SEEDS" \
        TEACHER_MODEL="$TEACHER_MODEL" \
        STUDENT_MODEL="$STUDENT_MODEL" \
        TEACHER_EPOCHS="$SUITE_EPOCHS" \
        STUDENT_EPOCHS="$SUITE_EPOCHS" \
        DISTILL_EPOCHS="$SUITE_EPOCHS" \
        LABEL_MODE="$label_mode" \
        FAIR_FEATURE="$fairness_feature" \
        SEQUENCE_LENGTH="$sequence_length" \
        PATCH_LENGTH="$patch_length" \
        USE_RR_FEATURES="$use_rr" \
        BEAT_INDEX_CSV="$beat_index_csv" \
        DISTILL_VARIANTS="$variants" \
        CHECKPOINT_SELECTION="utility_fairness" \
        SELECTION_FAIRNESS_CLASSES="$fairness_classes" \
        CLASS_BALANCED=1 \
        "${pipeline_assignments[@]}" \
        bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh; then
        touch "$case_dir/.complete"
        rm -f "$case_dir/.failed"
    else
        touch "$case_dir/.failed"
        echo "FAILED case: $case_name" >&2
        if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
            exit 1
        fi
    fi
    aggregate_suite
}

echo "MIT-BIH tuning suite"
echo "suite_dir=$SUITE_DIR"
echo "profile=$PROFILE seeds=$SUITE_SEEDS epochs=$SUITE_EPOCHS phases=$PHASES"
echo "cases=${CASES:-<all phase cases>}"
echo "aggregation_split=$AGGREGATION_SPLIT"
echo "dry_run=$DRY_RUN continue_on_error=$CONTINUE_ON_ERROR"

if phase_enabled kd; then
    run_case kd_a07_b03_t2 kd aami5 sex 256 16 0 "0,2" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.7 KD_BETA=0.3 KD_TEMPERATURE=2.0
    run_case kd_a05_b05_t2 kd aami5 sex 256 16 0 "0,2" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.5 KD_BETA=0.5 KD_TEMPERATURE=2.0
    run_case kd_a03_b07_t2 kd aami5 sex 256 16 0 "0,2" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.3 KD_BETA=0.7 KD_TEMPERATURE=2.0
    run_case kd_a03_b07_t4 kd aami5 sex 256 16 0 "0,2" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.3 KD_BETA=0.7 KD_TEMPERATURE=4.0
fi

if phase_enabled binary_kd; then
    run_case binary_kd_a03_b03_t2 binary_kd binary_ectopy sex 256 16 1 "1" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.3 KD_BETA=0.3 KD_TEMPERATURE=2.0 \
        MIN_TEACHER_PREDICTED_CLASSES=2
    run_case binary_kd_a07_b03_t2 binary_kd binary_ectopy sex 256 16 1 "1" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.7 KD_BETA=0.3 KD_TEMPERATURE=2.0 \
        MIN_TEACHER_PREDICTED_CLASSES=2 REUSE_BASELINES_FROM_CASE=binary_kd_a03_b03_t2
    run_case binary_kd_a05_b05_t2 binary_kd binary_ectopy sex 256 16 1 "1" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.5 KD_BETA=0.5 KD_TEMPERATURE=2.0 \
        MIN_TEACHER_PREDICTED_CLASSES=2 REUSE_BASELINES_FROM_CASE=binary_kd_a03_b03_t2
    run_case binary_kd_a03_b07_t2 binary_kd binary_ectopy sex 256 16 1 "1" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.3 KD_BETA=0.7 KD_TEMPERATURE=2.0 \
        MIN_TEACHER_PREDICTED_CLASSES=2 REUSE_BASELINES_FROM_CASE=binary_kd_a03_b03_t2
    run_case binary_kd_a05_b05_t1 binary_kd binary_ectopy sex 256 16 1 "1" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.5 KD_BETA=0.5 KD_TEMPERATURE=1.0 \
        MIN_TEACHER_PREDICTED_CLASSES=2 REUSE_BASELINES_FROM_CASE=binary_kd_a03_b03_t2
    run_case binary_kd_a05_b05_t4 binary_kd binary_ectopy sex 256 16 1 "1" baseline \
        ./data/mit-bih-arrhythmia/beat_index.csv KD_ALPHA=0.5 KD_BETA=0.5 KD_TEMPERATURE=4.0 \
        MIN_TEACHER_PREDICTED_CLASSES=2 REUSE_BASELINES_FROM_CASE=binary_kd_a03_b03_t2
fi

if phase_enabled t1; then
    for cap in 2 4 8; do
        run_case "t1_cap${cap}" t1 aami5 sex 256 16 0 "0,2" baseline,t1 \
            ./data/mit-bih-arrhythmia/beat_index.csv FAIR_MAX_OVERSAMPLE="$cap"
    done
fi

if phase_enabled o2; then
    run_case o2_lr25_reg1 o2 aami5 sex 256 16 0 "0,2" baseline,o2,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index.csv O2_LEARNING_RATE=0.000025 \
        O2_SCALE_REGULARIZATION=0.001 O2_BIAS_REGULARIZATION=0.001
    run_case o2_lr50_reg1 o2 aami5 sex 256 16 0 "0,2" baseline,o2,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index.csv O2_LEARNING_RATE=0.00005 \
        O2_SCALE_REGULARIZATION=0.001 O2_BIAS_REGULARIZATION=0.001
    run_case o2_lr100_reg0 o2 aami5 sex 256 16 0 "0,2" baseline,o2,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index.csv O2_LEARNING_RATE=0.0001 \
        O2_SCALE_REGULARIZATION=0.0 O2_BIAS_REGULARIZATION=0.0
fi

if phase_enabled rr; then
    run_case rr_seq256 rr aami5 sex 256 16 1 "0,2" baseline,t1,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index.csv RR_FUSION_WEIGHT=1.0
    run_case rr_seq720 rr aami5 sex 720 32 1 "0,2" baseline,t1,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index_720.csv RR_FUSION_WEIGHT=1.0
    run_case rr_seq1080 rr aami5 sex 1080 48 1 "0,2" baseline,t1,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index_1080.csv RR_FUSION_WEIGHT=1.0
fi

if phase_enabled binary; then
    run_case binary_ectopy binary binary_ectopy sex 256 16 1 "1" baseline,t1,o2,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index.csv MIN_TEACHER_PREDICTED_CLASSES=2
    run_case binary_non_n binary binary_non_n sex 256 16 1 "1" baseline,t1,o2,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index.csv MIN_TEACHER_PREDICTED_CLASSES=2
fi

if phase_enabled age; then
    run_case age_targeted age aami5 age_group 256 16 1 "0,2" baseline,t1,o2,t1_o2 \
        ./data/mit-bih-arrhythmia/beat_index.csv FAIR_MAX_OVERSAMPLE=4
fi

aggregate_suite
if phase_enabled binary_kd && [[ "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" scripts/fairness/select_mitbih_binary_kd.py \
        --suite-dir "$SUITE_DIR" \
        --min-seeds "$KD_SELECTION_MIN_SEEDS"
fi
echo "Suite finished. Results: $SUITE_DIR/SUITE_RESULTS.md"
