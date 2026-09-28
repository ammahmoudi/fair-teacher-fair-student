#!/usr/bin/env bash
set -euo pipefail

# One-shot environment setup for fair-teacher-fair-student on an RTX 5090 (Blackwell) under WSL2.
#
# Prereqs (on the WINDOWS side, not inside WSL):
#   - Up-to-date NVIDIA GeForce driver (570+ / Blackwell). WSL inherits it.
#   - Verify from inside WSL with:  nvidia-smi   (should show RTX 5090)
#
# This script creates a venv, installs Blackwell-capable torch (cu128), then the
# rest of the deps, and verifies the GPU actually works before you start a run.

# This script lives in scripts/setup/ — resolve the repo root (two levels up)
# and run everything from there, so the venv, requirements, and script-tree
# paths below are correct regardless of where the script is invoked from.
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SETUP_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "── 0a. Check WSL2 kernel version (need >= 5.10.43.3 for CUDA passthrough) ──"
KVER="$(uname -r)"
echo "   kernel: $KVER"
# Crude major.minor gate; if older, tell the user to update WSL on Windows.
KMAJ="$(echo "$KVER" | cut -d. -f1)"; KMIN="$(echo "$KVER" | cut -d. -f2)"
if [ "$KMAJ" -lt 5 ] || { [ "$KMAJ" -eq 5 ] && [ "$KMIN" -lt 10 ]; }; then
  echo "⚠️  Kernel looks older than 5.10.43.3 — run 'wsl --update' in Windows PowerShell."
fi

echo "── 0b. Check the GPU is visible to WSL ──"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-smi not found in WSL."
  echo "   On the WINDOWS host: install the NVIDIA CUDA-enabled WSL driver (Blackwell 570+ for a 5090)"
  echo "   from https://www.nvidia.com/download/index.aspx, then 'wsl --shutdown' and reopen WSL."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo "── 1. Create venv ──"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel

echo "── 2. Install Blackwell-capable torch (CUDA 12.8 wheels) ──"
# cu128 wheels ship torch >= 2.7 with sm_120 kernels for the 5090.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "── 3. Install the rest (torch omitted from this file on purpose) ──"
pip install -r "$SETUP_DIR/requirements-5090.txt"

echo "── 4. Verify GPU works ──"
python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("cuda build   :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available — check the Windows-side driver."
print("device       :", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability   :", f"sm_{cap[0]}{cap[1]}")
x = torch.randn(1024, 1024, device="cuda")
y = (x @ x).sum().item()   # forces a real kernel launch on the GPU
print("kernel launch: OK (matmul ran on GPU)")
PY

echo "── 5. Restore executable bit on shell scripts ──"
# tar/zip extraction can drop the +x bit; inference shells out to scripts/run_main.sh,
# so re-set it on every .sh to avoid 'Permission denied' during the run.
find . -name '*.sh' -not -path './venv/*' -exec chmod +x {} \;
echo "   done"

echo
echo "✅ Environment ready. To run the multi-seed robustness experiment:"
echo "   source venv/bin/activate"
echo "   ./scripts/pipelines/run_multiseed_robustness_experiment.sh"
echo
echo "Note: BERT / BERT-tiny download from HuggingFace on first run (needs internet)."
