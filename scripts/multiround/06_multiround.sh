#!/usr/bin/env bash
# The multi-round experiment: every seed block runs 4 rounds per arm (R, S).
#   round 1:  joint matched R/S subsets from the block's shared pool (K = 64 if every block allows,
#             else 32, else 16), trained from the shared adapter;
#   round 2+: each arm samples ITER_CONFIG generation.prompts_per_pool prompts of train_00t x
#             generation.candidates answers (512 x 4 in configs/matched_iterative.json) from its own
#             previous model and selects its own K = round 1's K examples with ITER_CONFIG's C:E
#             (matching.error_fraction; 48 + 16 at K = 64 and 3:1; R: any error, S: the task's target,
#             nonshortest for graph or ignore_parentheses for arithmetic).
# K/4 optimizer steps per round (16 at K = 64, 8 at 32, 4 at 16; 4 examples per step, one pass),
# lr 2e-4, continuing the arm's adapter.
# Exit 0 complete, 2 round-1 matching infeasible (stop and report), 3 complete with a stopped arm.
# Re-runnable: resumes from the last completed round.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

RESUME=""
if [ -f "$WORK/out_multiround/experiment.json" ]; then
  # Top-level key only (write_json indents by two).  A run of another study (a v01 run, made before
  # each block drew its own round-1 prompts) cannot be resumed: set it aside and start this one.
  recorded=$(grep -Eo '^  "study_id": "[^"]*"' "$WORK/out_multiround/experiment.json" | sed 's/.*: "//; s/"$//')
  if [ "$recorded" != "$STUDY" ]; then
    echo "06_multiround.sh: $WORK/out_multiround holds study '$recorded', not '$STUDY'." >&2
    echo "  Set it aside and rerun:  mv \"$WORK/out_multiround\" \"$WORK/out_multiround_$recorded\"" >&2
    exit 1
  fi
  RESUME="--resume"
fi
stage "multi-round $RESUME"
set +e
"$PY" -u run_iterative_experiment.py --config "$ITER_CONFIG" --data "$DATA" --shared-pool "$WORK/pools" \
  --out "$WORK/out_multiround" --seed 0 --study-id "$STUDY" --phase main \
  --shared-adapter "$WORK/shared_adapter" --reference-gradient "$WORK/reference_iterative" \
  $RESUME 2>&1 | tee -a "$LOGS/06_multiround.log"
code=${PIPESTATUS[0]}
set -e
echo "multi-round exit $code (0 complete, 2 round-1 infeasible, 3 complete with stopped arms, 1 failed)"
exit "$code"
