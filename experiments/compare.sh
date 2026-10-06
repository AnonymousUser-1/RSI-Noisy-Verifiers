#!/usr/bin/env bash
# Figures comparing evaluated experiments on one plot (e.g. unaudited vs audited of one task).
#
#   bash experiments/compare.sh graph_unaudited graph_audited
#   bash experiments/compare.sh arithmetic_unaudited arithmetic_audited
#
# Each experiment must have finished its evaluate stage on the same questions, decoding and base.
# Writes $RSI_ROOT/experiments/figures/A_vs_B_SPLIT_pass1.png, _by_difficulty.png, _summary.json/.csv
# (e.g. graph_unaudited_vs_graph_audited_eval_id_pass1.png)
# for each split in the first experiment's evaluation.json.
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/lib.sh
[ $# -ge 2 ] || { echo "usage: bash experiments/compare.sh EXPERIMENT EXPERIMENT [...]" >&2; exit 1; }

args=() names=() splits=""
for e in "$@"; do
  load_experiment "$e"
  args+=(--run "$WORK/out" --label "$LABEL")
  names+=("$EXP")
  [ -n "$splits" ] || splits="$(eval_splits "$EVAL_CONFIG")"
done
joined="$(printf "%s_vs_" "${names[@]}")"
out="$RSI_ROOT/experiments/figures/${joined%_vs_}"
mkdir -p "$(dirname "$out")"
stage "compare $*"
"$PY" -u plot_multiround.py "${args[@]}" --splits $splits --out "$out"
