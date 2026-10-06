#!/usr/bin/env bash
# Run one experiment end to end, or some of its stages.  See experiments/README.md.
#
#   bash experiments/run.sh EXPERIMENT [STAGE ...]
#
# EXPERIMENT: graph_unaudited, graph_audited, arithmetic_unaudited, arithmetic_audited (a folder in
# experiments/), or the path of a copy.  STAGE, in order: data model pools adapter train evaluate
# figures (default: all of them).  Every stage skips or resumes what is already done, so after an
# interruption run the same command again.
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/lib.sh
load_experiment "${1:-}"
shift
STAGES="${*:-data model pools adapter train evaluate figures}"
for s in $STAGES; do
  case "$s" in
    data|model|pools|adapter|train|evaluate|figures) "stage_$s" ;;
    *) echo "Unknown stage '$s' (data model pools adapter train evaluate figures)" >&2; exit 1 ;;
  esac
done
echo "DONE $EXP: $STAGES  (outputs in $WORK)"
