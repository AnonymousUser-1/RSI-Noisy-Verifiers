#!/usr/bin/env bash
# Round-1 pools: one shared pool per seed block (b00, b01, ...), drawn from the base model on
# train_001 with POOL_CONFIG's settings (configs/README.md; Qwen3-1.7B: 2,048 prompts x 8 answers,
# temperature 1.3).  Both arms of a block are built from its pool.  ~38 min per pool for Qwen3-1.7B
# on an RTX 4070 Laptop at the former 256-token cap; an answer that loops runs to max_new_tokens
# (2,048) and holds its batch until then.
# Re-runnable: a pool with its .meta.json is kept if POOL_CONFIG's generation settings are still the
# ones it was drawn with (else this stops: blocks drawn with other settings would be mixed); a
# half-written pool is drawn again.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

mkdir -p "$WORK/pools"
for s in $SEEDS; do
  block=$(printf "b%02d" "$s")
  meta="$WORK/pools/$block.jsonl.meta.json"
  if [ -f "$meta" ]; then
    "$PY" - "$POOL_CONFIG" "$meta" <<'EOF' || exit 1
import json, sys
sys.path.insert(0, ".")
from rsi.common import load_config
config, meta = sys.argv[1], sys.argv[2]
want, have = load_config(config)["generation"], json.load(open(meta, encoding="utf-8")).get("generation")
if have != want:
    sys.exit("04_round1_pools.sh: %s was drawn with generation %s, but %s now says %s.  Move $WORK/pools "
             "aside to draw every block with the new settings, or restore the config." % (
                 meta, json.dumps(have, sort_keys=True), config, json.dumps(want, sort_keys=True)))
EOF
    echo "skip: pool $block exists, drawn with $POOL_CONFIG's generation settings"; continue
  fi
  rm -f "$WORK/pools/$block.jsonl"
  stage "pool $block"
  "$PY" -u sample_candidates.py --config "$POOL_CONFIG" --data "$DATA" --out "$WORK/pools/$block.jsonl" \
    --seed "$s" --split train_001 --study-id "$STUDY" --phase main 2>&1 | tee "$LOGS/04_pool_$block.log"
done
