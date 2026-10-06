#!/usr/bin/env bash
# The audited multi-round run: 06 again with budgeted auditing.
# Same data, round-1 pools, shared adapter and seed blocks as 06; AUDIT_CONFIG adds, for each arm and
# round from round 1, 16 correctness checks on the selected examples (adaptive policy): an audited
# wrong answer is removed, the unaudited ones are weighted by their difficulty's estimated error rate.
# h is bound to the config, so the audit config gets its own (reference_audit).  As long as 06.
# Exit codes as 06.  Re-runnable: resumes from the last completed round.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

if [ ! -f "$WORK/reference_audit/h.json" ]; then
  rm -rf "$WORK/reference_audit"
  stage "reference gradient (audit config)"
  "$PY" -u compute_reference_gradient.py --config "$AUDIT_CONFIG" --data "$DATA" \
    --shared-adapter "$WORK/shared_adapter" --out "$WORK/reference_audit" --seed 0 2>&1 | tee "$LOGS/09_h_audit.log"
fi
RESUME=""
if [ -f "$WORK/out_audit/experiment.json" ]; then
  recorded=$(grep -Eo '^  "study_id": "[^"]*"' "$WORK/out_audit/experiment.json" | sed 's/.*: "//; s/"$//')
  if [ "$recorded" != "$AUDIT_STUDY" ]; then
    echo "09_audit.sh: $WORK/out_audit holds study '$recorded', not '$AUDIT_STUDY'." >&2
    echo "  Set it aside and rerun:  mv \"$WORK/out_audit\" \"$WORK/out_audit_$recorded\"" >&2
    exit 1
  fi
  RESUME="--resume"
fi
stage "audited multi-round $RESUME"
set +e
"$PY" -u run_iterative_experiment.py --config "$AUDIT_CONFIG" --data "$DATA" --shared-pool "$WORK/pools" \
  --out "$WORK/out_audit" --seed 0 --study-id "$AUDIT_STUDY" --phase main \
  --shared-adapter "$WORK/shared_adapter" --reference-gradient "$WORK/reference_audit" \
  $RESUME 2>&1 | tee -a "$LOGS/09_audit.log"
code=${PIPESTATUS[0]}
set -e
echo "audited multi-round exit $code (0 complete, 2 round-1 infeasible, 3 complete with stopped arms, 1 failed)"
exit "$code"
