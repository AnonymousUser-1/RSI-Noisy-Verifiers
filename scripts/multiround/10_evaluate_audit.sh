#!/usr/bin/env bash
# Pass@1 after every round of the audited run (09), as 08 does for the unaudited one, then the
# figures comparing the two (the unaudited run is left out if 08 has not evaluated it yet).
# About as long as 08.  Re-runnable.  Writes
#   $WORK/out_audit/evaluation_greedy_eval_id/, evaluation_greedy_eval_ood/
#   $WORK/figures/${TASK}_${MODEL_TAG}_unaudited_vs_audited_SPLIT_pass1.png, _by_difficulty.png,
#       _summary.json/.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

stage "evaluate (audited)"
# The splits named in EVAL_CONFIG, for the figures.
EVAL_SPLITS="$("$PY" -c 'import json, sys; sys.stdout.write(" ".join(json.load(open(sys.argv[1]))["splits"]))' "$EVAL_CONFIG")"
"$PY" -u evaluate_multiround.py --run "$WORK/out_audit" --config "$EVAL_CONFIG" \
  2>&1 | tee -a "$LOGS/10_evaluate_audit.log"
evaluated=1
for split in $EVAL_SPLITS; do
  [ -f "$WORK/out_multiround/evaluation_greedy_$split/protocol.json" ] || evaluated=0
done
if [ "$evaluated" = 1 ]; then
  "$PY" -u plot_multiround.py --run "$WORK/out_multiround" --label unaudited --run "$WORK/out_audit" \
    --label "audited B=16" --splits $EVAL_SPLITS --out "$WORK/figures/${TASK}_${MODEL_TAG}_unaudited_vs_audited" \
    2>&1 | tee -a "$LOGS/10_evaluate_audit.log"
else
  "$PY" -u plot_multiround.py --run "$WORK/out_audit" --label "audited B=16" --splits $EVAL_SPLITS \
    --out "$WORK/figures/${TASK}_${MODEL_TAG}_audited" 2>&1 | tee -a "$LOGS/10_evaluate_audit.log"
fi
