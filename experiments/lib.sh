# Sourced by run.sh, pilot.sh and compare.sh with an experiment: reads that experiment's
# settings.sh, sets the paths and defines one function per stage.  It holds no hyperparameter:
# every value a run uses is in the experiment's config.json, evaluation.json and settings.sh.
#
# Machine settings, from the environment (the same for every experiment):
#   PY                    the Python of the venv (default: python)
#   RSI_ROOT              where data and outputs go (default: $HOME/rsi-work)
#   CUDA_VISIBLE_DEVICES  which GPU (default: 0)

load_experiment() {
  local arg="${1:?usage: bash experiments/run.sh EXPERIMENT [STAGE ...]  (EXPERIMENT: a folder in experiments/)}"
  if [ -d "$arg" ]; then EXP_DIR="${arg%/}"; else EXP_DIR="experiments/$arg"; fi
  if [ ! -f "$EXP_DIR/settings.sh" ]; then
    echo "No experiment at $EXP_DIR (it needs settings.sh, config.json and evaluation.json)." >&2; exit 1
  fi
  EXP="$(basename "$EXP_DIR")"
  TASK="" STUDY="" PIN_FILE="" SEEDS="" SHARE_POOLS_WITH="" LABEL="" DATA_NAME="" POOLS_IMPORTED="" POOLS_IMPORT_ALLOWED=""
  # shellcheck disable=SC1090
  source "$EXP_DIR/settings.sh"
  case "$TASK" in
    graph|arithmetic) ;;
    *) echo "$EXP_DIR/settings.sh: TASK must be graph or arithmetic, not '$TASK'" >&2; exit 1 ;;
  esac
  for name in STUDY PIN_FILE SEEDS; do
    if [ -z "${!name}" ]; then echo "$EXP_DIR/settings.sh does not set $name" >&2; exit 1; fi
  done
  LABEL="${LABEL:-$EXP}"
  CONFIG="$EXP_DIR/config.json"
  EVAL_CONFIG="$EXP_DIR/evaluation.json"
  for f in "$CONFIG" "$EVAL_CONFIG" "$PIN_FILE"; do
    [ -f "$f" ] || { echo "Missing $f" >&2; exit 1; }
  done

  PY="${PY:-python}"
  RSI_ROOT="${RSI_ROOT:-$HOME/rsi-work}"
  # DATA_NAME (optional): another data folder under $RSI_ROOT/data, for imported pools bound to the
  # exact bytes of the data they were drawn from (experiments/import_pools.py).
  DATA="$RSI_ROOT/data/${DATA_NAME:-$TASK}"
  WORK="$RSI_ROOT/experiments/$EXP"
  # Importing is an explicit alternative to generating a fresh run. Keep the original-input
  # replay restriction only when an import receipt exists, not for a model's fresh runs.
  if [ "$POOLS_IMPORT_ALLOWED" = yes ] && [ -f "$WORK/pools/IMPORTED.json" ]; then
    POOLS_IMPORTED=yes
  fi
  LOGS="$WORK/logs"
  mkdir -p "$LOGS"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
  # Every entry checks the base against this pin.  (pwd -W gives a Windows path under Git Bash.)
  export RSI_BASE_PIN="$(pwd -W 2>/dev/null || pwd)/$PIN_FILE"
}

stage() { echo "STAGE $EXP: $1  $(date '+%Y-%m-%d %H:%M:%S')"; }

block_name() { printf "b%02d" "$1"; }

# POOLS_IMPORTED=yes (settings.sh): the round-1 pools, the shared adapter and the data are imported
# with experiments/import_pools.py, never drawn or generated here.
require_import() {
  if [ "$POOLS_IMPORTED" = yes ] && [ ! -f "$WORK/pools/IMPORTED.json" ]; then
    echo "$EXP: its round-1 pools, shared adapter and data are imported, not drawn.  Run first:" >&2
    echo "  $PY experiments/import_pools.py $EXP SOURCE_DIR" >&2
    exit 1
  fi
}

# The study's questions for TASK (seed 2027; the same for every experiment and model), with the
# fingerprint check of scripts/multiround/01_make_data.sh.  Skipped if they exist.
stage_data() {
  require_import
  stage "data ($TASK)"
  TASK="$TASK" DATA="$DATA" WORK="$WORK" PY="$PY" bash scripts/multiround/01_make_data.sh
}

# The pinned base model, into the Hugging Face cache.
stage_model() {
  stage "model"
  "$PY" - "$RSI_BASE_PIN" <<'EOF'
import json, sys
from huggingface_hub import snapshot_download
pin = json.load(open(sys.argv[1]))
print("downloaded", pin["model"], "@", pin["revision"], "->", snapshot_download(pin["model"], revision=pin["revision"]))
EOF
}

# One round-1 pool per seed block: all of train_001 (generation.prompts_per_pool) x
# generation.candidates answers, sampled with config.json.  A block's pool is copied from
# SHARE_POOLS_WITH when that experiment drew it with exactly this base, data and sampling.
stage_pools() {
  require_import
  mkdir -p "$WORK/pools"
  local s block pool other
  for s in $SEEDS; do
    block=$(block_name "$s"); pool="$WORK/pools/$block.jsonl"
    if [ -f "$pool.meta.json" ]; then echo "keep: pool $block exists"; continue; fi
    other="$RSI_ROOT/experiments/$SHARE_POOLS_WITH/pools/$block.jsonl"
    if [ -n "$SHARE_POOLS_WITH" ] && [ -f "$other.meta.json" ] && \
        "$PY" experiments/pool_check.py --config "$CONFIG" --data "$DATA" --pool "$other" --seed "$s" > /dev/null; then
      cp "$other" "$pool"
      cp "$other.meta.json" "$pool.meta.json"
      echo "reuse: pool $block from $SHARE_POOLS_WITH (same base, data and sampling)"
      continue
    fi
    rm -f "$pool"
    stage "pool $block"
    "$PY" -u sample_candidates.py --config "$CONFIG" --data "$DATA" --out "$pool" \
      --seed "$s" --split train_001 --study-id "$STUDY" --phase main 2>&1 | tee "$LOGS/pool_$block.log"
  done
}

# The shared LoRA initialisation both arms start from (seed 0) and the reference gradient h at it.
# Both are bound to config.json: after editing it, delete $WORK/shared_adapter and $WORK/reference --
# except after raising only rounds, which extends the run and needs both as they are.
stage_adapter() {
  require_import
  if [ ! -f "$WORK/shared_adapter/shared_adapter.json" ]; then
    rm -rf "$WORK/shared_adapter"
    stage "shared adapter"
    "$PY" -u make_shared_adapter.py --config "$CONFIG" --out "$WORK/shared_adapter" --seed 0 2>&1 | tee "$LOGS/adapter.log"
  else
    echo "keep: shared adapter exists"
  fi
  if [ ! -f "$WORK/reference/h.json" ]; then
    rm -rf "$WORK/reference"
    stage "reference gradient h"
    "$PY" -u compute_reference_gradient.py --config "$CONFIG" --data "$DATA" \
      --shared-adapter "$WORK/shared_adapter" --out "$WORK/reference" --seed 0 2>&1 | tee "$LOGS/reference.log"
  else
    echo "keep: h exists"
  fi
}

# The multi-round run (run_iterative_experiment.py): round 1 matched R/S from the shared pools, then
# each arm samples its own pools from its own model; training and auditing as config.json says.
# Exit 0 complete, 3 complete with a stopped arm (both continue to evaluation); 2 round-1 matching
# infeasible, 1 failed (both stop).  Re-runnable: resumes from the last completed round.  When
# config.json's rounds is larger than the run's in out/, the complete run is extended to it
# (--extend-rounds): its rounds are kept and the new ones follow.
stage_train() {
  if [ "$POOLS_IMPORTED" = yes ]; then
    echo "$EXP: its round-1 pools are imported, and so is the multi-round run's round-1 matching on them" >&2
    echo "($WORK/pools/IMPORTED.json).  A multi-round run is not started from this folder; it serves" >&2
    echo "experiments/one_step.sh." >&2
    exit 1
  fi
  local s block bad=0
  for s in $SEEDS; do
    block=$(block_name "$s")
    if ! "$PY" experiments/pool_check.py --config "$CONFIG" --data "$DATA" --pool "$WORK/pools/$block.jsonl" --seed "$s"; then
      bad=1
    fi
  done
  if [ "$bad" = 1 ]; then
    echo "$EXP: a round-1 pool does not match config.json (edited since it was drawn?).  Delete $WORK/pools" \
         "and run the pools stage again." >&2
    exit 1
  fi
  local resume=""
  if [ -f "$WORK/out/experiment.json" ]; then
    local recorded
    recorded=$(grep -Eo '^  "study_id": "[^"]*"' "$WORK/out/experiment.json" | sed 's/.*: "//; s/"$//')
    if [ "$recorded" != "$STUDY" ]; then
      echo "$WORK/out holds study '$recorded', not '$STUDY'.  Set it aside to start this one:" >&2
      echo "  mv \"$WORK/out\" \"$WORK/out_$recorded\"" >&2
      exit 1
    fi
    resume="--resume"
    local ran want
    ran=$(grep -Eo '^  "rounds": [0-9]+' "$WORK/out/experiment.json" | grep -Eo '[0-9]+$' || true)
    want=$("$PY" -c 'import json, sys; sys.stdout.write(str(json.load(open(sys.argv[1]))["rounds"]))' "$CONFIG")
    if [ -n "$ran" ] && [ "$want" -gt "$ran" ]; then
      echo "$EXP: out/ holds a $ran-round run and config.json has rounds $want: extending it with rounds" \
           "$((ran + 1)) to $want"
      resume="--resume --extend-rounds"
    fi
  fi
  stage "train $resume"
  set +e
  "$PY" -u run_iterative_experiment.py --config "$CONFIG" --data "$DATA" --shared-pool "$WORK/pools" \
    --out "$WORK/out" --seed 0 --study-id "$STUDY" --phase main \
    --shared-adapter "$WORK/shared_adapter" --reference-gradient "$WORK/reference" \
    $resume 2>&1 | tee -a "$LOGS/train.log"
  local code=${PIPESTATUS[0]}
  set -e
  echo "$EXP: train exit $code (0 complete, 3 complete with a stopped arm, 2 round-1 infeasible, 1 failed)"
  if [ "$code" != 0 ] && [ "$code" != 3 ]; then exit "$code"; fi
}

# Greedy Pass@1 of round 0 (the base) and every block/arm/round, on every question of the splits
# in evaluation.json.  Re-runnable: finished checkpoints are skipped.
stage_evaluate() {
  stage "evaluate"
  "$PY" -u evaluate_multiround.py --run "$WORK/out" --config "$EVAL_CONFIG" 2>&1 | tee -a "$LOGS/evaluate.log"
}

eval_splits() {
  "$PY" -c 'import json, sys; sys.stdout.write(" ".join(json.load(open(sys.argv[1]))["splits"]))' "$1"
}

# Pass@1 vs round for R and S, S - R, the target-error rate, per difficulty: one set per split.
stage_figures() {
  stage "figures"
  "$PY" -u plot_multiround.py --run "$WORK/out" --label "$LABEL" --splits $(eval_splits "$EVAL_CONFIG") \
    --out "$WORK/figures/$EXP" 2>&1 | tee -a "$LOGS/figures.log"
}
