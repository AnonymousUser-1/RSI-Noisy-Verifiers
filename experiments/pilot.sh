#!/usr/bin/env bash
# Pilot check of an experiment's model and sampling on the dev split, before its main run:
# all 512 dev prompts x generation.candidates answers with config.json, then
# scripts/multiround/pilot_report.py.  Go ahead when the report's criteria pass: no truncated
# answers, format errors <= 5%, accuracy 30-70%, and the estimated J_E prompts per 2,048-prompt
# round-1 pool >= the wrong answers K = 64 needs (64 x matching.error_fraction: 16 at 3:1; 1.25 x that
# with margin).  Otherwise report it before the main run: round-1 matching
# may be infeasible (run.sh would stop with exit 2).  About 10-20 min.
#
#   bash experiments/pilot.sh EXPERIMENT
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/lib.sh
load_experiment "${1:-}"

stage_data
mkdir -p "$WORK/pilot"
if [ ! -f "$WORK/pilot/dev_pool.jsonl.meta.json" ]; then
  rm -f "$WORK/pilot/dev_pool.jsonl"
  stage "pilot pool (dev)"
  "$PY" -u sample_candidates.py --config "$CONFIG" --data "$DATA" --out "$WORK/pilot/dev_pool.jsonl" \
    --seed 0 --split dev --study-id "$STUDY-pilot" --phase pilot 2>&1 | tee "$LOGS/pilot_pool.log"
else
  echo "keep: pilot pool exists (pilot_report.py refuses it if config.json's sampling changed; then rm -rf $WORK/pilot)"
fi
stage "pilot report"
"$PY" scripts/multiround/pilot_report.py --config "$CONFIG" --data "$DATA" \
  --pool "$WORK/pilot/dev_pool.jsonl" --split dev --out "$WORK/pilot/pilot_report.json"
echo "Report: $WORK/pilot/pilot_report.json"
