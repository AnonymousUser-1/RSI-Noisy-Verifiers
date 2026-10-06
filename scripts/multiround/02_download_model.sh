#!/usr/bin/env bash
# Download the base model at its pinned revision into the Hugging Face cache.
# The Llama-3.2 Instruct models are gated: approved access and `huggingface-cli login` (or HF_TOKEN) first.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

stage "download $MODEL_TAG"
"$PY" - "$RSI_BASE_PIN" <<'EOF'
import json, sys
from huggingface_hub import snapshot_download
pin = json.load(open(sys.argv[1]))
path = snapshot_download(pin["model"], revision=pin["revision"])
print("downloaded", pin["model"], "@", pin["revision"], "->", path)
EOF
