#!/usr/bin/env bash
# Validate that an older LLM-TIME checkout can run the BG standalone O2 suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

required_files=(
  "data/ohiot1dm/all_patients_combined/all_patients_training.csv"
  "data/ohiot1dm/all_patients_combined/all_patients_testing.csv"
  "data/ohiot1dm/raw_standardized/t1dm_prompt.txt"
  "distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/phase_1_teacher/bert_all_patients_10epochs/logs/logs_2025-10-28_14-20-20/checkpoints/checkpoint.pth"
  "scripts/pipelines/run_bg_o2_only_multiseed.sh"
  "scripts/fairness/aggregate_bg_o2_only_multiseed.py"
)

missing=0
for required_file in "${required_files[@]}"; do
  if [[ -f "$required_file" ]]; then
    printf 'FOUND   %s\n' "$required_file"
  else
    printf 'MISSING %s\n' "$required_file" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Missing required files. Restore the OhioT1DM data or extract the transfer archive again." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

modules = ("torch", "accelerate", "gin", "numpy", "pandas", "transformers")
missing = [module for module in modules if importlib.util.find_spec(module) is None]
if missing:
    print(f"Missing Python modules: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(1)

import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start the full five-seed suite on CPU.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "BG O2 transfer verification passed."