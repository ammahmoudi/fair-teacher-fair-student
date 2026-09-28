#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SUITE_DIR="${SUITE_DIR:-experiments/mitbih_binary_ectopy_kd_grid}"
DRY_RUN="${DRY_RUN:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

mkdir -p "$SUITE_DIR"
SUITE_DIR="$(cd "$SUITE_DIR" && pwd)"

exec env \
    SUITE_DIR="$SUITE_DIR" \
    PROFILE=full \
    PHASES=binary_kd \
    SUITE_SEEDS=831363 \
    SUITE_EPOCHS=5 \
    KD_SELECTION_MIN_SEEDS=1 \
    DRY_RUN="$DRY_RUN" \
    CONTINUE_ON_ERROR=0 \
    MIN_FREE_GB="$MIN_FREE_GB" \
    AGGREGATION_SPLIT=validation \
    TEACHER_MODEL=BERT \
    STUDENT_MODEL=TinyBERT \
    bash scripts/pipelines/run_mitbih_tuning_suite.sh
