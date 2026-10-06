# Optional original experiment inputs

Every model's fresh experiments generate their own data and candidate pools. The original
Qwen3-4B round-1 inputs are supplied separately as `RSI-Qwen3-4B-Graph-paired-inputs-20261004.zip`
for inspecting the recorded experiment or replaying its one-step comparison. Downloading this
archive is not a prerequisite for a fresh run.

[Download the optional archive](https://github.com/AnonymousUser-1/RSI-Noisy-Verifiers/releases/download/qwen3-4b-round1-inputs-v1/RSI-Qwen3-4B-Graph-paired-inputs-20261004.zip)
and its [SHA256SUMS.txt](https://github.com/AnonymousUser-1/RSI-Noisy-Verifiers/releases/download/qwen3-4b-round1-inputs-v1/SHA256SUMS.txt).
The archive SHA-256 is
`7981687e4dfa3b02aa5b28dabdfcffa7300e8e9775b995e647611e66febc7e3e`.

Extract the archive outside the code checkout. Its top-level directory contains `graph/`,
`pools/`, `shared_adapter/`, `matching/`, two reference-gradient directories and their manifests.
Set `INPUTS` to that extracted directory. The archive keeps all original published bytes and hashes.

```bash
export INPUTS=/path/to/RSI-Qwen3-4B-Graph-paired-inputs-20261004
(cd "$INPUTS" && python verify_inputs.py)
```

## Check the original paper records

From `manuscript/`, the following use original task files or candidate answers:

```bash
python verify_qwen4b_results.py --code .. --inputs "$INPUTS"
python summarize_qwen4b_tables.py --code .. --inputs "$INPUTS" --check
```

The first checks saved table values and source hashes. The second also rejudges the 81,920
original round-1 candidates. Neither command trains a model or modifies the saved results.
`summarize_qwen4b_results.py --code .. --inputs "$INPUTS"` regenerates the Qwen3-4B summary and
figures, including rejudging the 123,000 saved audited answers; run regeneration in a working copy.
Missing archived files remain an explicit incomplete check, never a silent pass.

## Replay the original one-step comparison

Use a fresh output directory, separate from any experiment you generated yourself:

```bash
export RSI_ROOT=/path/to/original-input-replay
python experiments/import_pools.py graph_unaudited_qwen3-4b "$INPUTS"
bash experiments/one_step.sh graph_unaudited_qwen3-4b
```

Import validates the model, sampling configuration, seeds, data, adapter settings and file hashes,
and copies the original first-round matching. The import receipt reserves this experiment folder
for one-step replay. Start a fresh multi-round run with another `RSI_ROOT` instead.

Only Qwen3-4B's original inputs are supplied. The other models' saved run records include input
metadata, hashes and matched selections, but not the full candidate answers or initial adapter
weights. This difference affects original-sample inspection, not the availability of their fresh
generation workflows. The archive does not contain the adapters of every trained round.
