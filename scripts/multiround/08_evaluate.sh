#!/usr/bin/env bash
# Pass@1 after every round of the unaudited multi-round run (06), then its figures.
# One greedy answer (temperature 0) per question of the splits in configs/matched_evaluation.json
# (eval_id: 2,000, in distribution; eval_ood: 1,000, larger graphs or expressions), for round 0 --
# the base model, evaluated once -- and for every block/arm/round 1-4: 41 checkpoints at 5 seed
# blocks, each loaded once for all splits.  On an H100 (graph, base model, 3,000 questions) 211 s per
# checkpoint for Llama-3.2-3B and 333 s for Qwen3-1.7B, with generation.repetition_stop ending loops
# early (3,544 s for Llama-3.2-3B without it).  Faster with a larger batch: point EVAL_CONFIG at
# a copy of configs/matched_evaluation.json with a larger batch_size (it changes the answers only by
# padding noise, and a batch too large for the GPU is halved automatically).
# Re-runnable: finished checkpoints are skipped.  Writes
#   $WORK/out_multiround/evaluation_greedy_eval_id/, evaluation_greedy_eval_ood/
#       per checkpoint: Pass@1, per difficulty, error types; every answer
#   $WORK/figures/${TASK}_${MODEL_TAG}_unaudited_SPLIT_pass1.png, _by_difficulty.png, _summary.json/.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

stage "evaluate (unaudited)"
# The splits named in EVAL_CONFIG, for the figures.
EVAL_SPLITS="$("$PY" -c 'import json, sys; sys.stdout.write(" ".join(json.load(open(sys.argv[1]))["splits"]))' "$EVAL_CONFIG")"
"$PY" -u evaluate_multiround.py --run "$WORK/out_multiround" --config "$EVAL_CONFIG" \
  2>&1 | tee -a "$LOGS/08_evaluate.log"
"$PY" -u plot_multiround.py --run "$WORK/out_multiround" --label unaudited --splits $EVAL_SPLITS \
  --out "$WORK/figures/${TASK}_${MODEL_TAG}_unaudited" 2>&1 | tee -a "$LOGS/08_evaluate.log"
