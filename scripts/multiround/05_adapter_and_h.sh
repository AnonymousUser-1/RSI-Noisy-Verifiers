#!/usr/bin/env bash
# The shared LoRA initialisation every arm starts from (rank 8, alpha 16, dropout 0, q/v, seed 0),
# and the multi-round run's reference gradient h at it (ITER_CONFIG; h is bound to the config hash).
# The one-step comparison computes its own h (experiments/one_step.sh).  Under a minute each.
# Re-runnable.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

if [ ! -f "$WORK/shared_adapter/shared_adapter.json" ]; then
  rm -rf "$WORK/shared_adapter"
  stage "shared adapter"
  "$PY" -u make_shared_adapter.py --config "$ITER_CONFIG" --out "$WORK/shared_adapter" --seed 0 \
    2>&1 | tee "$LOGS/05_adapter.log"
fi
if [ -f "$WORK/reference_iterative/h.json" ]; then
  echo "skip: h for iterative exists"
else
  rm -rf "$WORK/reference_iterative"
  stage "reference gradient (iterative)"
  "$PY" -u compute_reference_gradient.py --config "$ITER_CONFIG" --data "$DATA" \
    --shared-adapter "$WORK/shared_adapter" --out "$WORK/reference_iterative" --seed 0 2>&1 | tee "$LOGS/05_h_iterative.log"
fi
