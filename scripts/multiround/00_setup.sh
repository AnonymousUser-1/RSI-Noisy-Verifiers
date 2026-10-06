#!/usr/bin/env bash
# One-time environment: a Python 3.10-3.12 venv with CUDA PyTorch and the pinned libraries,
# a GPU check and the CPU test suite.  Usage: bash scripts/multiround/00_setup.sh [venv-dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV="${1:-$HOME/venvs/rsi}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
# The default PyPI torch wheel on Linux is a CUDA build.  If your driver needs a specific CUDA
# version, install torch first from https://download.pytorch.org/whl/cu1XX, then rerun this.
"$VENV/bin/python" -m pip install -r requirements-gpu.txt pytest
"$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "no CUDA device"; \
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), \
round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")'
"$VENV/bin/python" -m pytest tests -q

echo
echo "Done.  Use this interpreter:  export PY=$VENV/bin/python"
echo "Llama-3.2 is gated: request access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
echo "(and -1B-Instruct for llama3.2-1b), then log in once:  $VENV/bin/huggingface-cli login   (or export HF_TOKEN=...)"
