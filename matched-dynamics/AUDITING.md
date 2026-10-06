# Optional budgeted auditing for the R/S runners

`run_matched_experiment.py` and `run_iterative_experiment.py` apply the existing
`rsi/auditing.py` policies **after R/S selection and before optimization** when
the config enables auditing. This is a separate extension, not a modification
of the frozen unaudited main protocol. Existing configs remain `none`, budget 0.
This feature is CPU/unit-tested; it is not a completed GPU experiment or a
scientific result. Graph and arithmetic use their respective exact checkers;
the same integration also handles imported tasks supported by the runners.

## Activation

Add this section to a copy of the appropriate model's config:

```json
"audit": {"policy": "adaptive", "budget": 16, "weighting": true}
```

Policies are `uniform`, `balanced` and `adaptive`. A budget is a nonnegative
integer count of **correctness queries per arm per round**, capped at the
pre-audit selected subset's size. Set `none` with budget 0 for the control.
`weighting: false` removes audited errors without reweighting other examples.
The supplied Qwen3-1.7B examples are `configs/matched_pool_audit.json` (one step)
and `configs/matched_iterative_audit.json` (four rounds), both with budget 16.
A budget of 64 audits every selected example when K <= 64; it is effectively
full trusted filtering, not a scarce-budget condition. Freeze budgets and
policies before confirmatory runs; compare B=0 and multiple positive budgets.

The allocator sees only candidate IDs, task IDs, responses and weights, plus
task strata and the **same arm's previous-round audit counts**. It never sees
current hidden correctness/error labels. `audit_pool` calls `judge` exactly
once for each audited candidate, removes audited incorrect responses, and
optionally assigns the existing posterior reliability weights to unaudited
responses. It reveals no canonical answer and does not replace responses.
Adaptive priorities use the immediately preceding round's stratum counts;
they do not pool histories across arms or use current-round labels to allocate.
Paired arms share the audit RNG seed within a block; different selections and
adaptive histories can still result in different audited examples/allocations.

## Running

Keep the baseline's data, pinned base, shared adapter and round-1 pools. These
are separate outputs, and **h must be bound to the audit-enabled training
config**. Computing h still uses only the held-out `gradient_reference` split,
with uniform reference weights and no audit queries. For the four-round example:

```bash
CUDA_VISIBLE_DEVICES=0 python compute_reference_gradient.py \
  --config configs/matched_iterative_audit.json --data DATA \
  --shared-adapter SHARED_ADAPTER --out REFERENCE_AUDIT --seed 0

CUDA_VISIBLE_DEVICES=0 python run_iterative_experiment.py \
  --config configs/matched_iterative_audit.json --data DATA --shared-pool POOLS \
  --shared-adapter SHARED_ADAPTER --reference-gradient REFERENCE_AUDIT \
  --out RUN_AUDIT --study-id md-audit-iter4-v01 --phase main \
  --seed 0
```

Replace the capitalized paths with actual paths. DATA must contain all requested
training rounds; POOLS is the baseline's block-keyed round-1 pool directory.
For a one-step run, compute h with `configs/matched_pool_audit.json`, then use
`run_matched_experiment.py --backend hf` with that same config and its reference
directory. Do not mix the one-step and iterative config hashes. The standalone
`one_step.py` CLI retains its strict unaudited comparison; activate this
extension through the newer runners. Other model pins work with copies of
their corresponding configs; these example files do not change any base pin.

## Matching and training

All original matching/selection constraints apply **before auditing**. The
accepted correct/error counts, task coverage, token-length pairing, weights
and rates can change after deletion/reweighting; post-audit equality is not
promised. Keep the B=0 comparison as the matched-rate isolation study and
describe B>0 as an auditing intervention. Do not claim that pre-audit matching
certifies equal post-audit training distributions or downstream learning.

The one-step runner accumulates `sum_i w_i grad(loss_i) / sum_i w_i`, then takes
one AdamW step. Its frozen null remains unaudited and takes zero steps. The
iterative runner normalizes weights within each effective batch, takes
`ceil(positive_weight_examples / effective_batch_size) * epochs` steps and
records the actual count. Each round starts with fresh optimizer state.
If no examples have positive weights, either runner takes zero steps; an
unchanged adapter is saved so iterative generation can continue next round.

## Outputs and trusted-label cost

Each audited arm-round writes:

- `pre_audit.jsonl`: selected responses before the auditing intervention.
- `audit.json`: policy, requested/actual budget, queried IDs, correctness-only
  labels, inclusion probabilities, per-stratum history and pre/post statistics.
- `training.jsonl`: post-audit responses/weights, with no correctness or error
  taxonomy fields in the saved training file. Optimization uses prompts,
  responses and weights; the one-step recorder still uses experimental truth
  labels internally to measure correct/error gradient contributions.
- `complete.json`: actual `audit_queries`, `cumulative_audit_queries`,
  `audit_groups`, selected/retained example counts and training statistics.
  Iterative rounds also record `expected_optimizer_steps` after auditing.
- `finished.json`: the arm's cumulative audit query count on completion.

TPR/FPR use the **original candidate pool's** N+/N- denominators, not the
retained subset's own denominators. The report additionally distinguishes
retained rows from positive-weight rows and gives weight sum/effective sample
size. Labels in the controlled selection machinery are experimental oracle
information: their cost is **not** included in `audit_queries`. This budget
counts only the additional online auditing intervention, not all trusted
labels used to construct, diagnose or evaluate the controlled experiment.
At budget B and T rounds, two completed arms cost at most `2 * T * min(B, K)`
queries per block. Infeasible/stopped rounds spend no audit budget.

## Resume and evaluation

The iterative runner saves completed audits under
`BLOCK/ARM/audit_ledger/round_00t.json`, outside the retryable training-round
directory. Later-round generated pools are also snapshotted in
`round_00t.pool.json`, bound to the config, prompt/task data, sampling seed and
previous adapter files. GPU sampled decoding can vary on retry, so the runner
reuses that saved pool rather than resampling an already-audited selection.
A training interruption after the audit ledger is saved reuses those
results, rather than re-querying or double-charging them. Hashes bind the
selection, tasks, audit config, seed, round, original-pool denominators and
prior arm history; changed inputs or payload corruption fail closed. Completed
rounds are skipped as before. An interruption **during the query batch before
the ledger is saved** can still repeat local checker calls on retry; this is
not an exactly-once guarantee for a remote paid oracle. Preserve ledger files.
Existing unaudited outputs cannot be converted in place: use a fresh output
directory and study ID for an auditing condition.

ID/OOD evaluation remains a separate `evaluate.py --run BLOCK/ARM ...` step,
as in the baseline. Neither runner automatically evaluates checkpoints. Freeze
the same held-out splits and decoding for all conditions. The strict frozen
one-step paired-matching protocol validates the original unaudited subsets;
it does not certify an audited training subset. Analyze audited conditions
using their post-audit records and state this protocol extension explicitly.

## Regression checks

```bash
python -m pytest -q tests/test_budgeted_auditing.py tests/test_runner_auditing.py \
  tests/test_one_step_accumulation.py
python -m pytest -q
```

The integration tests use real local checker calls on fixture responses and
stub GPU generation/training. CPU toy LoRA tests execute the real one-step
weighted gradient/update, composition checks and zero-signal diagnostics.
