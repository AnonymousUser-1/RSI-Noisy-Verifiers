#!/usr/bin/env bash
# Pilot check of this model on the dev split (pilot phase reads dev only), before any main pool:
# the dev prompts x generation.candidates answers with POOL_CONFIG's settings, then pilot_report.py.
# Go ahead when: no truncated answers, format errors <= 5%, accuracy 30-70%, and the J_E estimate
# for a round-1 pool >= the wrong answers K = 64 needs (64 x matching.error_fraction: 16 at 3:1; 1.25 x that for
# margin).  If not, stop: the prompt or temperature may need a pilot
# change for this model before the main run.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

mkdir -p "$WORK/pilot"
if [ ! -f "$WORK/pilot/dev_pool.jsonl.meta.json" ]; then
  rm -f "$WORK/pilot/dev_pool.jsonl"
  stage "pilot pool (dev)"
  "$PY" -u sample_candidates.py --config "$POOL_CONFIG" --data "$DATA" --out "$WORK/pilot/dev_pool.jsonl" \
    --seed 0 --split dev --study-id "$PILOT_STUDY" --phase pilot 2>&1 | tee "$LOGS/03_pilot_pool.log"
else
  echo "skip: pilot pool exists (pilot_report.py refuses it if the data, model or sampling changed since;" \
       "then rm -rf $WORK/pilot)"
fi
stage "pilot report"
"$PY" scripts/multiround/pilot_report.py --config "$POOL_CONFIG" --data "$DATA" \
  --pool "$WORK/pilot/dev_pool.jsonl" --split dev --out "$WORK/pilot/pilot_report.json"
