# Settings shared by every script in scripts/multiround/.  Edit here (or export the
# variables before running a script); the scripts source this file.  Run every script
# from anywhere: each one changes to the repository root first.
#
#   MODEL_TAG=qwen3-4b bash scripts/multiround/run_all.sh
#   TASK=arithmetic MODEL_TAG=qwen3-4b bash scripts/multiround/run_all.sh
#
# Keep STUDY the same for every rerun of one study: --resume and the pool records use it.

# Which task: graph (S targets nonshortest paths) | arithmetic (S targets ignore_parentheses)
TASK="${TASK:-graph}"
case "$TASK" in
  graph|arithmetic) ;;
  *) echo "config.sh: unknown TASK '$TASK' (graph | arithmetic)" >&2; exit 1 ;;
esac

# Which base model: qwen3-1.7b (the main study) | qwen3-4b | llama3.2-3b | llama3.2-1b
MODEL_TAG="${MODEL_TAG:-qwen3-1.7b}"

case "$MODEL_TAG" in
  qwen3-1.7b)  SUFFIX="";             PIN_FILE="configs/pins/base_pin.json" ;;
  qwen3-4b)    SUFFIX="_qwen3-4b";    PIN_FILE="configs/pins/base_pin_qwen3-4b.json" ;;
  llama3.2-3b) SUFFIX="_llama3.2-3b"; PIN_FILE="configs/pins/base_pin_llama3.2-3b.json" ;;
  llama3.2-1b) SUFFIX="_llama3.2-1b"; PIN_FILE="configs/pins/base_pin_llama3.2-1b.json" ;;
  *) echo "config.sh: unknown MODEL_TAG '$MODEL_TAG' (qwen3-1.7b | qwen3-4b | llama3.2-3b | llama3.2-1b)" >&2; exit 1 ;;
esac

# Configs: pool sampling + one-step arms, and the multi-round training (4 rounds, K/4 steps each -- 16 at
# K = 64 --, lr 2e-4).
POOL_CONFIG="configs/matched_pool${SUFFIX}.json"
ITER_CONFIG="configs/matched_iterative${SUFFIX}.json"
# The audited multi-round run (09): ITER_CONFIG plus budgeted auditing, adaptive, 16 correctness
# checks per arm per round from round 1.
AUDIT_CONFIG="configs/matched_iterative_audit${SUFFIX}.json"
# Every entry checks the base against this pin; it is recorded in every run.
# (pwd -W gives a Windows path under Git Bash, which Windows Python can open; Linux has no -W.)
REPO_ROOT="$(pwd -W 2>/dev/null || pwd)"
export RSI_BASE_PIN="${REPO_ROOT}/${PIN_FILE}"

# Where things go.  DATA is shared by all models (same questions); WORK is per model and task
# (graph keeps the plain model name it had before arithmetic was added, so running studies resume).
RSI_ROOT="${RSI_ROOT:-$HOME/rsi-work}"
DATA="${DATA:-$RSI_ROOT/data/$TASK}"
# An exported DATA must hold this TASK's questions: the entries take S's target from the data, so
# graph data under TASK=arithmetic would silently run graph under the arithmetic study's name.
if [ -f "$DATA/manifest.json" ] && ! grep -Eq "^  \"task\": \"$TASK\"" "$DATA/manifest.json"; then
  echo "config.sh: $DATA does not hold $TASK data; set TASK to match it or unset DATA" >&2; exit 1
fi
if [ "$TASK" = graph ]; then DEFAULT_WORK="$RSI_ROOT/$MODEL_TAG"; else DEFAULT_WORK="$RSI_ROOT/$MODEL_TAG-$TASK"; fi
WORK="${WORK:-$DEFAULT_WORK}"
LOGS="$WORK/logs"

# Study identity (RUN_COST_SPEC 1).  Change the version suffix for a fresh study; never mid-study.
# v02: round-1 matching draws each seed block's own prompts (#17, 2026-10-02); a v01 multi-round run
# used the old matching, where the blocks shared most prompts, and is not used.
STUDY="${STUDY:-md-main-${TASK}-iter4-${MODEL_TAG}-v02}"
PILOT_STUDY="${PILOT_STUDY:-md-pilot-${TASK}-${MODEL_TAG}-v01}"
AUDIT_STUDY="${AUDIT_STUDY:-md-audit-${TASK}-iter4-b16-${MODEL_TAG}-v01}"

# Evaluation (08, 10): greedy Pass@1 on every question of the splits in EVAL_CONFIG, with its batch
# size (a batch too large for the GPU is halved automatically).  It is not part of a training study:
# on a large GPU, point EVAL_CONFIG at a copy with a larger batch_size (configs/README.md).
EVAL_CONFIG="${EVAL_CONFIG:-configs/matched_evaluation.json}"

# Settings that moved into the config files: stop rather than ignore them.
for retired in EVAL_SPLITS EVAL_BATCH LATER_PROMPTS LATER_SAMPLES; do
  if [ -n "${!retired:-}" ]; then
    case "$retired" in
      EVAL_*) where="\"splits\" / \"batch_size\" in EVAL_CONFIG ($EVAL_CONFIG)" ;;
      *) where="generation.prompts_per_pool / generation.candidates in ITER_CONFIG ($ITER_CONFIG)" ;;
    esac
    echo "config.sh: $retired is no longer read; the setting is $where.  Unset $retired." >&2; exit 1
  fi
done

# Seed blocks (one round-1 pool each: b00, b01, ...).  Every sampling and training setting is in
# POOL_CONFIG / ITER_CONFIG (configs/README.md); nothing here overrides them.
SEEDS="${SEEDS:-0 1 2 3 4}"

# Python (a 3.10-3.12 environment with requirements-gpu.txt) and the GPU to use.
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1

mkdir -p "$LOGS"
stage() { echo "STAGE $1 $(date '+%Y-%m-%d %H:%M:%S')"; }
