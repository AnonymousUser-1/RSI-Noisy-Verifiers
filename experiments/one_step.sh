#!/usr/bin/env bash
# The one-step R/S/null comparison of an experiment, on the experiment's own round-1 pools and shared
# adapter.  See ONE_STEP_EXPERIMENT_INSTRUCTION.md.
#
#   bash experiments/one_step.sh EXPERIMENT [STAGE ...]
#
# EXPERIMENT: an unaudited experiment folder in experiments/, or the path of a copy.  STAGE, in order:
# reference train evaluate figures (default: all of them).
#
# The round-1 pools and the shared adapter are inputs: they are the experiment's own, produced by its
# multi-round stages (bash experiments/run.sh EXPERIMENT data model pools adapter) or imported
# (experiments/import_pools.py).  This script never draws or creates them, so the one-step and the
# multi-round comparison always start from the same answers and the same initial adapter.  The
# multi-round run's round-1 matching (out/matching/matched_subsets.json, or round1_reference/ after
# an import) is an input too: the acceptance checks compare the one-step R/S sets with it.
#
# The one-step config is the experiment's config.json with rounds 1 and the one-step learning rate
# (ONE_STEP_LR, default 5e-5): one AdamW step over all K examples per arm, in every block of SEEDS.
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/lib.sh
load_experiment "${1:-}"
shift
ONE_STEP_LR="${ONE_STEP_LR:-5e-5}"
OUT="$WORK/out_one_step"
ONE_CONFIG="$WORK/one_step_config.json"
# Cost records (rsi/cost.py) of h and of the evaluation; the arms' are in $OUT/cost_records, and a run
# set aside keeps its own (out_one_step_*/cost_records), so failed attempts are not lost.
COSTS="$WORK/one_step_cost_records"

case "$EXP" in
  *_audited*) echo "$EXP is an audited experiment; the one-step comparison runs with auditing off.  Run it" \
                   "on the unaudited experiment of the same model and task." >&2; exit 1 ;;
esac

# The multi-round round-1 matching the one-step matching must reproduce: the experiment's own multi-round
# run, or a copy of its matched_subsets.json placed in round1_reference/ (an import places it there).
round1_reference() {
  local f
  for f in "$WORK/out/matching/matched_subsets.json" "$WORK/round1_reference/matched_subsets.json"; do
    if [ -f "$f" ]; then echo "$f"; return; fi
  done
}

# config.json with rounds 1 and the one-step rate; every other field (base, sampling, LoRA, matching)
# is the experiment's, so the pools and the shared adapter check out against it.  Written once.
write_one_step_config() {
  "$PY" - "$CONFIG" "$ONE_CONFIG" "$ONE_STEP_LR" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
config["rounds"] = 1
config["training"] = {k: config["training"][k] for k in ("lora_rank", "lora_alpha", "lora_dropout", "target_modules")}
config["training"]["learning_rate"] = float(sys.argv[3])
if config["audit"] != {"policy": "none", "budget": 0, "weighting": True}:
    sys.exit("%s: auditing is on; the one-step comparison runs with auditing off" % sys.argv[1])
text = json.dumps(config, indent=2) + "\n"
try:
    if open(sys.argv[2]).read() != text:
        sys.exit("%s differs from what config.json and ONE_STEP_LR give now.  Set aside %s, out_one_step/ and "
                 "reference_one_step/ in the same folder to start again." % (sys.argv[2], sys.argv[2]))
except FileNotFoundError:
    open(sys.argv[2], "w").write(text)
EOF
}

# The inputs the one-step run reads and never creates: the data, every block's round-1 pool and the
# shared adapter.
require_inputs() {
  require_import
  local s block missing=()
  [ -f "$DATA/manifest.json" ] || missing+=("$DATA/manifest.json")
  for s in $SEEDS; do
    block=$(block_name "$s")
    [ -f "$WORK/pools/$block.jsonl" ] && [ -f "$WORK/pools/$block.jsonl.meta.json" ] || missing+=("$WORK/pools/$block.jsonl")
  done
  [ -f "$WORK/shared_adapter/shared_adapter.json" ] || missing+=("$WORK/shared_adapter/")
  [ -n "$(round1_reference)" ] || missing+=("$WORK/out/matching/matched_subsets.json")
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "$EXP: the one-step run reads the experiment's round-1 pools, shared adapter and round-1 matching" >&2
    echo "and does not create them.  Missing: ${missing[*]}" >&2
    echo "Produce them with the experiment's own stages (bash experiments/run.sh $EXP; the train stage" >&2
    echo "writes the round-1 matching before its first update), or import them (experiments/import_pools.py)." >&2
    echo "See ONE_STEP_EXPERIMENT_INSTRUCTION.md, section 3." >&2
    exit 1
  fi
}

# h at the shared adapter for the one-step config (h is bound to the config hash, so the multi-round
# reference/ does not serve).
stage_one_step_reference() {
  require_inputs
  if [ -f "$WORK/reference_one_step/h.json" ]; then echo "keep: one-step h exists"; return; fi
  rm -rf "$WORK/reference_one_step"
  stage "one-step reference gradient h"
  "$PY" -u compute_reference_gradient.py --config "$ONE_CONFIG" --data "$DATA" \
    --shared-adapter "$WORK/shared_adapter" --out "$WORK/reference_one_step" --seed 0 --cost-records "$COSTS" \
    2>&1 | tee "$LOGS/one_step_reference.log"
}

# The acceptance checks of experiments/one_step_check.py on out_one_step/, recorded in
# out_one_step/checks.json.  Every stage that uses the run calls this, also on a run finished earlier: a
# run that fails a check is never reused, evaluated or plotted.
check_one_step() {
  if [ ! -f "$OUT/experiment.json" ] || ! grep -Eq '^  "status": "complete"' "$OUT/experiment.json"; then
    echo "$EXP: $OUT holds no complete one-step run; run the train stage first" >&2
    exit 1
  fi
  local reference
  reference=$(round1_reference)
  if ! "$PY" experiments/one_step_check.py --run "$OUT" ${reference:+--reference "$reference"} \
      --record "$OUT/checks.json" 2>&1 | tee "$LOGS/one_step_check.log"; then
    echo "$EXP: the one-step run in $OUT fails its checks (above; recorded in $OUT/checks.json).  It is" \
         "not evaluated.  Set it aside (mv $OUT ${OUT}_failed) and train again, or report the failure." >&2
    exit 1
  fi
}

# The same checks and, in addition, a complete evaluation: every split of evaluation.json on all 11
# checkpoints (the base, and R and S of each block), on all its questions and the arms' current
# adapters.  Nothing is plotted or tabulated from an incomplete evaluation.
check_one_step_evaluated() {
  check_one_step
  local reference
  reference=$(round1_reference)
  if ! "$PY" experiments/one_step_check.py --run "$OUT" ${reference:+--reference "$reference"} \
      --evaluation "$EVAL_CONFIG" --record "$OUT/checks.json" 2>&1 | tee "$LOGS/one_step_check.log"; then
    echo "$EXP: the evaluation of $OUT is not complete (above; recorded in $OUT/checks.json).  Run the" \
         "evaluate stage again; it resumes and evaluates only what is missing." >&2
    exit 1
  fi
}

# One step per arm: run_matched_experiment.py on every block's pool (copied, since it reads every
# *.jsonl in the folder).  It does not resume: a complete out_one_step/ is kept and checked again, an
# incomplete one stops.
stage_one_step_train() {
  if [ -f "$OUT/experiment.json" ]; then
    if grep -Eq '^  "status": "complete"' "$OUT/experiment.json"; then
      echo "keep: $OUT is complete; checking it again"
      check_one_step
      return
    fi
    echo "$OUT holds a run that did not complete; set it aside (mv $OUT ${OUT}_failed) and run again" >&2
    exit 1
  fi
  require_inputs
  local s block
  for s in $SEEDS; do
    block=$(block_name "$s")
    "$PY" experiments/pool_check.py --config "$CONFIG" --data "$DATA" --pool "$WORK/pools/$block.jsonl" --seed "$s"
  done
  if [ ! -f "$WORK/reference_one_step/h.json" ]; then
    echo "$EXP: no one-step h in $WORK/reference_one_step; run the reference stage first" >&2
    exit 1
  fi
  rm -rf "$WORK/pools_one_step"
  mkdir -p "$WORK/pools_one_step"
  for s in $SEEDS; do
    block=$(block_name "$s")
    cp "$WORK/pools/$block.jsonl" "$WORK/pools/$block.jsonl.meta.json" "$WORK/pools_one_step/"
  done
  stage "one-step train"
  set +e
  "$PY" -u run_matched_experiment.py --config "$ONE_CONFIG" --data "$DATA" --shared-pool "$WORK/pools_one_step" \
    --out "$OUT" --seed 0 --study-id "$STUDY-one-step" --phase main \
    --shared-adapter "$WORK/shared_adapter" --reference-gradient "$WORK/reference_one_step" \
    2>&1 | tee "$LOGS/one_step_train.log"
  local code=${PIPESTATUS[0]}
  set -e
  echo "$EXP: one-step train exit $code (0 complete, 2 round-1 matching infeasible, 1 failed)"
  [ "$code" = 0 ] || exit "$code"
  check_one_step
}

# Greedy Pass@1 of the base (round 0, which the null arm is) and of every updated arm, under the
# experiment's evaluation.json, then the completeness check.  Re-runnable: finished checkpoints are
# skipped.
stage_one_step_evaluate() {
  check_one_step
  stage "one-step evaluate"
  "$PY" -u evaluate_multiround.py --run "$OUT" --config "$EVAL_CONFIG" --cost-records "$COSTS" \
    2>&1 | tee -a "$LOGS/one_step_evaluate.log"
  check_one_step_evaluated
}

# From a complete, checked evaluation only: Pass@1 of R and S against the base and S - R
# (plot_multiround.py, as the multi-round figures); the update figures (h^T Delta theta, |Delta theta|,
# per-example gradients, error change against null; PDF and 300 dpi PNG); the per-block table, the
# paper's one-step results table, and the cost table.  Any failing command stops the script (pipefail).
stage_one_step_figures() {
  check_one_step_evaluated
  stage "one-step figures"
  local prefix="$WORK/figures/${EXP}_one_step" log="$LOGS/one_step_figures.log" reference
  reference=$(round1_reference)
  "$PY" -u plot_multiround.py --run "$OUT" --label "one-step" --splits $(eval_splits "$EVAL_CONFIG") \
    --out "$prefix" 2>&1 | tee -a "$log"
  "$PY" -u experiments/one_step_figures.py --run "$OUT" --label "$EXP" --out "$prefix" 2>&1 | tee -a "$log"
  "$PY" experiments/one_step_check.py --run "$OUT" ${reference:+--reference "$reference"} \
    --evaluation "$EVAL_CONFIG" --table "${prefix}_table.csv" --table15 "${prefix}_table15.csv" 2>&1 | tee -a "$log"
  "$PY" experiments/one_step_cost.py --records "$COSTS" $(ls -d "$WORK"/out_one_step*/cost_records 2>/dev/null) \
    --pools "$WORK/pools_one_step" --out "${prefix}_cost.csv" 2>&1 | tee -a "$log"
}

STAGES="${*:-reference train evaluate figures}"
write_one_step_config
for s in $STAGES; do
  case "$s" in
    reference) stage_one_step_reference ;;
    train) stage_one_step_train ;;
    evaluate) stage_one_step_evaluate ;;
    figures) stage_one_step_figures ;;
    *) echo "Unknown stage '$s' (reference train evaluate figures)" >&2; exit 1 ;;
  esac
done
# Reached only when every stage succeeded: set -euo pipefail stops the script at the first failure.
echo "DONE $EXP one-step, every stage succeeded: $STAGES  (outputs in $OUT)"
