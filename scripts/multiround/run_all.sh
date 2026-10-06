#!/usr/bin/env bash
# Everything after setup and the pilot check, in order, for one model:
#   01 data -> 02 model download -> 04 round-1 pools -> 05 adapter and h -> 06 multi-round
#   -> 08 its Pass@1 evaluation and figures -> 09 the audited multi-round run -> 10 its evaluation
#   and the unaudited-vs-audited figures
#   (09 and 10 are skipped with RUN_AUDIT=0).  The one-step comparison is run from experiments/
#   (experiments/one_step.sh; ONE_STEP_EXPERIMENT_INSTRUCTION.md).
# Run 00_setup.sh once and 03_pilot_check.sh first for a model the study has not used yet.
# Re-run the same command after any interruption: finished stages are skipped and the
# multi-round runs and evaluations resume.  Example, detached on a Linux GPU box:
#   MODEL_TAG=qwen3-4b PY=$HOME/venvs/rsi/bin/python nohup bash scripts/multiround/run_all.sh \
#     > run_all_qwen3-4b.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
here=scripts/multiround
bash "$here/01_make_data.sh"
bash "$here/02_download_model.sh"
bash "$here/04_round1_pools.sh"
bash "$here/05_adapter_and_h.sh"
set +e
bash "$here/06_multiround.sh"
code=$?
set -e
# 0 complete, 3 complete with a stopped arm: both are evaluated.  2 (round 1 infeasible) and 1 stop here.
if [ "$code" != 0 ] && [ "$code" != 3 ]; then exit "$code"; fi
bash "$here/08_evaluate.sh"
if [ "${RUN_AUDIT:-1}" = 1 ]; then
  set +e
  bash "$here/09_audit.sh"
  acode=$?
  set -e
  if [ "$acode" != 0 ] && [ "$acode" != 3 ]; then exit "$acode"; fi
  bash "$here/10_evaluate_audit.sh"
fi
exit "$code"
