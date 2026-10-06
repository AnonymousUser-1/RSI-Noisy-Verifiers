# Arithmetic R/S intervention

This is a separate arithmetic extension, requested after the graph protocol was
frozen. It does not change the graph main line, run optimizer updates, or report
a completed LLM experiment.

`rsi/arithmetic_intervention.py` reads saved model-generated candidate pools and
constructs both training arms jointly:

- **R:** uniform random error selection within each shared prompt/token bucket.
- **S:** select arithmetic `ignore_parentheses` errors within that same bucket.
  The existing exact judge defines this as a wrong answer equal to evaluating
  the expression after removing parentheses. If removing parentheses leaves the
  correct value unchanged, it is not an error and cannot enter the error quota.

The graph matcher still defaults to `nonshortest`. The arithmetic entry passes
its target explicitly; there is no fallback to graph, sign, or arbitrary errors
when the target is scarce.

## Inputs and sampling

Use a dataset root containing `manifest.json` and the original question splits,
not the `answers/arithmetic` reference sidecars. Only `dev` (pilot) or
`train_NNN` (main) can enter this intervention. Calibration, evaluation and
gradient-reference splits are refused.

First use `sample_candidates.py` to save one pool per seed block. For example,
from the repository root on the GPU server:

```bash
python sample_candidates.py --config configs/matched_pool.json \
  --data data/snapshots/generator-381f340-seed2027/arithmetic \
  --out results/arithmetic/pools/b00.jsonl --seed 0 --backend hf \
  --split train_001 --study-id arithmetic-rs --phase main
```

Repeat for `b01.jsonl` with seed 1 and `b02.jsonl` with seed 2, using the same
configuration, split and identity. Keep every `<block>.jsonl.meta.json` sidecar.
The configuration's model revision must resolve to the repository's frozen base
pin; explicitly put the immutable revision from `matched-dynamics/base_pin.json`
in your configuration rather than relying on a moving Hub `main`.

The data snapshot is published on `data-snapshot-20261002`, not bundled
into `t006-integration`. Download it separately if that example path is absent;
record the downloaded commit and preserve its manifest hashes.

## Construct both arms

```bash
python -m rsi.arithmetic_intervention \
  --data data/snapshots/generator-381f340-seed2027/arithmetic \
  --pools results/arithmetic/pools --split train_001 \
  --study-id arithmetic-rs --phase main --seed 0 \
  --out results/arithmetic/matched
```

HF is the default backend. This command needs Transformers and the pinned
tokenizer, but does not load model weights or require a GPU. It uses the same
chat-template and response-only encoding as training, recounting response
tokens **including the real EOS** and excluding prompt/padding. It never uses
raw generation `completion_tokens` as a substitute for supervised token counts
and never truncates a training example to force a match.

It verifies dataset/pool hashes, task/sample IDs, full candidate coverage,
study/phase/split identity, and common model/revision/generation settings before
selection. Candidate IDs can repeat across seed blocks; their responses and
token counts remain block-local. These checks bind saved inputs, but cannot
authenticate how an arbitrary externally authored response was produced; only
use genuine model pools from the recorded sampling workflow for research.

Both arms have the same prompts, same correct-response IDs, identical per-prompt
supervised token counts, one response per prompt, C:E=3:1 and weight 1. R may
select the same error as S; that coincidence is kept. The common size follows
64 -> 32 -> 16 and must work for **every** input block. Missing target errors or
missing equal-length target/non-target buckets are genuine feasibility failures.
Do not change seeds, manufacture answers, or relax matching after seeing them.

## Outputs and limits

- `intervention.json`: task/target identity, hashes, tokenization, original-pool
  error counts, policies, output hashes, code hash/command and `trained: false`.
- `ladder.json`: every size attempt and per-block feasibility certificate.
- `matched_subsets.json`: both arms' candidate IDs and constraint/rate audits,
  with the common original pool as the denominator.
- `<block>/R/training.jsonl` and `<block>/S/training.jsonl`: original prompt/model
  response pairs, uniform weights, IDs and supervised token counts. No oracle
  answer or correctness/error label enters the training rows.

Exit status is 0 on feasible preparation, 2 when the whole ladder is infeasible,
and 1 for invalid inputs. An infeasible run writes its certificate but no
training files. Existing output directories containing results are refused.

`--backend mock` accepts only mock pools and labels the manifest and matching
artifacts `DEMO_ONLY`. Its whitespace count plus synthetic EOS is only for CPU
plumbing tests, never research token matching. The existing mock generator may
not generate any `ignore_parentheses` errors; that should produce an infeasible
certificate, not fake evidence of an arithmetic contrast.

This script prepares one shared-pool intervention, not eight on-policy rounds.
The graph orchestration/HF integration limitations remain unchanged. Exporting
arithmetic training files does not make the integrated real-model study complete.
Choose a separate arithmetic shared initialization and downstream training/
evaluation protocol before launching a formal replication. For a one-step
replication, both arms must start from that same initialization and take the same
single optimizer update; no such GPU update is performed here.

## CPU checks

```bash
python -m unittest discover -s tests -p 'test_arithmetic_intervention.py' -v
```

The tests use hand-built, explicitly test-only candidate fixtures and a stub
tokenizer. They check arithmetic R/S behavior, graph-default regression,
matching/ladder constraints, response/EOS counting, block-local IDs, split
isolation, provenance refusals, CLI output and infeasibility. They are not
measurements of a trained language model.
