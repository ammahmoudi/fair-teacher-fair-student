#!/usr/bin/env bash
set -euo pipefail

# Package a minimal, transferable archive of this project — a single file you can
# copy to another machine. Each run produces a fresh timestamped archive:
#     ../fair-teacher-fair-student_YYYYMMDD_HHMMSS.zip   (or .tar.gz if `zip` is not installed)
#
# Contents (only what the multi-seed robustness run needs):
#   - all code: git-tracked files + new untracked-but-not-ignored files
#     (so freshly-written runners/scripts are always included)
#   - data/                       (git-ignored, but required — added explicitly)
#   - the 2 teacher checkpoints    (the run distills FROM these — added explicitly)
#
# Everything .gitignore excludes (venv/, results/, logs/, past experiments, the
# 17 GB of distillation_experiments student runs) is left out automatically.
# The HuggingFace cache is NOT included — BERT/BERT-tiny download on first run.
#
# Usage:
#   ./scripts/utilities/prepare_for_transfer.sh                 # -> ../fair-teacher-fair-student_<ts>.zip
#   ./scripts/utilities/prepare_for_transfer.sh /some/dir       # archive written into /some/dir
#   TS=20260620_1200 ./scripts/utilities/prepare_for_transfer.sh   # override timestamp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="${1:-$(dirname "$SRC")}"
NAME="fair-teacher-fair-student"
# Timestamp: overridable via $TS (the shell can't call date() in some sandboxes).
TS="${TS:-$(date +%Y%m%d_%H%M%S 2>/dev/null || echo manual)}"

cd "$SRC"
mkdir -p "$OUT_DIR"

# ── 1. Verify the two teacher checkpoints exist BEFORE building anything ──
PIPE="distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/phase_1_teacher"
BASE_CKPT="$PIPE/bert_all_patients_10epochs/logs/logs_2025-10-28_14-20-20/checkpoints/checkpoint.pth"
FAIR_CKPT="$PIPE/bert_all_patients_10epochs_fair_gender/bert_all_patients_10epochs/logs/logs_2026-06-13_11-13-20/checkpoints/checkpoint.pth"
for c in "$BASE_CKPT" "$FAIR_CKPT"; do
  if [[ ! -f "$c" ]]; then
    echo "❌ Required teacher checkpoint missing — aborting so we don't ship a broken bundle:"
    echo "   $c"
    exit 1
  fi
done
echo "✅ Both teacher checkpoints present."

# ── 2. Build the file list ──
#   a) git-tracked code  b) new untracked-but-not-ignored files  c) data/  d) the 2 checkpoints
echo "── Collecting file list ──"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
{
  git ls-files
  git ls-files --others --exclude-standard
  # data/ is git-ignored, add it explicitly (only existing files)
  find data -type f 2>/dev/null
  echo "$BASE_CKPT"
  echo "$FAIR_CKPT"
} | sort -u > "$LIST"
N=$(wc -l < "$LIST")
echo "   $N files"

# ── 3. Archive (prefer zip; fall back to tar.gz) ──
if command -v zip >/dev/null 2>&1; then
  ARCHIVE="$OUT_DIR/${NAME}_${TS}.zip"
  echo "── Creating $ARCHIVE ──"
  # zip with a prefix dir so it extracts into fair-teacher-fair-student_<ts>/, not loose into cwd.
  rm -f "$ARCHIVE"
  # -@ reads the file list from stdin; relative paths preserved.
  zip -q "$ARCHIVE" -@ < "$LIST"
else
  ARCHIVE="$OUT_DIR/${NAME}_${TS}.tar.gz"
  echo "── 'zip' not installed; creating $ARCHIVE instead ──"
  echo "   (tar.gz is smaller and unzips with: tar xzf <file>)"
  tar -czf "$ARCHIVE" -T "$LIST"
fi

# ── 4. Report ──
echo
echo "✅ Archive ready: $ARCHIVE"
echo "   Size: $(du -sh "$ARCHIVE" 2>/dev/null | cut -f1)"
echo
echo "On the TARGET machine:"
case "$ARCHIVE" in
  *.zip)    echo "  mkdir -p fair-teacher-fair-student && cd fair-teacher-fair-student && unzip ../${NAME}_${TS}.zip" ;;
  *.tar.gz) echo "  mkdir -p fair-teacher-fair-student && cd fair-teacher-fair-student && tar xzf ../${NAME}_${TS}.tar.gz" ;;
esac
echo "  ./scripts/setup/setup_5090.sh                      # see scripts/setup/SETUP_NEW_PC.md for Windows/driver steps"
echo "  source venv/bin/activate"
echo "  ./scripts/pipelines/run_multiseed_robustness_experiment.sh"
