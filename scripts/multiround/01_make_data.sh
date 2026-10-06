#!/usr/bin/env bash
# The dataset for $TASK (graph or arithmetic): seed 2027, 8 x 2,048 training prompts, dev 512,
# calibration 500, eval_id 2,000, eval_ood 1,000, gradient_reference 64, with the study's prompts
# (rsi/tasks.py: graph_prompt, arithmetic_prompt).  Same questions and ids as
# data/snapshots/generator-381f340-seed2027 (data-snapshot-20261002); only the prompt text
# differs.  Shared by every model.  Skipped if $DATA already exists.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/multiround/config.sh

if [ -f "$DATA/manifest.json" ]; then
  echo "skip: $DATA exists"
elif [ -d "$DATA" ] && [ -n "$(ls -A "$DATA")" ]; then
  echo "01_make_data.sh: $DATA has files but no manifest.json (an interrupted run?); delete it and rerun" >&2
  exit 1
else
  stage "data ($TASK)"
  "$PY" generate_data.py --task "$TASK" --seed 2027 --out "$DATA"
fi

# File hashes depend on the platform's line ending, so compare these OS-independent fingerprints
# with everyone else's.
"$PY" - "$DATA" "$TASK" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from rsi.common import digest
root, task = Path(sys.argv[1]), sys.argv[2]
EXPECTED = {
    "graph": ("8057e1acc8ac4266e4c38f0dc2d04707a02f76cd1c52b06f3518b343b791a43f",
              "78894a850567edab038c8a2abdaec3377ab89535b6a4155378be264bc85592dd"),
    "arithmetic": ("255036c428848baa26512a5f6f13de51fe4de5d6c78e5fa703f1f013b3f92161",
                   "7f935f392ea466901be8ad92901d8aacd50145a9e5738d92ff19abedf5a39c2f"),
}
manifest = json.load(open(root / "manifest.json"))
if manifest["task"] != task:
    sys.exit("%s holds %s data, not %s" % (root, manifest["task"], task))
files = sorted(root.glob("*.jsonl"))
ids = digest({p.stem: sorted(json.loads(l)["id"] for l in open(p)) for p in files})
prompts = digest({p.stem: digest([json.loads(l)["prompt"] for l in open(p)]) for p in files})
want_ids, want_prompts = EXPECTED[task]
print("ids fingerprint     ", ids, "OK" if ids == want_ids else "DIFFERS")
print("prompts fingerprint ", prompts, "OK" if prompts == want_prompts else "DIFFERS")
if ids != want_ids or prompts != want_prompts:
    sys.exit("The data is not the study's data: check the branch (rsi/tasks.py prompts) and the seed.")
EOF
