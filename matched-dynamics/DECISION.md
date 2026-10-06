# First public version: scope decision and disposition

Decision owner: research/engineering/delivery lead for this task. Date: 2026-09-28.
This file records what the first reviewable version contains, what it deliberately excludes, and
how each known defect was handled. It is a decision record, not an acceptance.

## What the first version is

A **reviewable checkpoint**, not a finished study:

1. `matched-dynamics/PLAN.md` - the research claim, the analytic core with frozen numbers, the
   joint matched-subset construction, the measurement set, the isolation rules, and the readout
   limits.
2. `matched-dynamics/HANDOFF.md` - the three human collaborators' ownership, inputs, outputs, and
   hand-off order.
3. `matched-dynamics/KNOWN_GAPS.md` - every defect and every unrun piece, including the three
   concrete module defects and the fact that no GPU work has happened.
4. `rsi/matching.py`, `rsi/logistic.py` - the new candidate modules, committed **as unaudited
   candidates**.

## Scope excluded from this version

- The full HF/PEFT generation-training-evaluation chain for the new main line. It does not exist
  yet; it is the next engineering task, not a claim about this version.
- Any GPU run, any throughput or memory number, any power statement.
- The legacy eight-round matrix, the audit budget matrix, and the 4B replication.
- Merging anything into `main`, and any change to repository visibility.

Reasons: the user asked for a first upload to review, the deadline is 2026-10-06, and an
unreviewed full pipeline would be a larger unverifiable claim than a flagged checkpoint.

## Disposition of the three module defects

The hand-off checkpoint flagged three local-check problems. Disposition for this version:

- `finite_pool_trajectory`: **removed.** It was outside the frozen contract and simulated a
  different process. Nothing in `PLAN.md` cited it. Deleted together with its helpers `_sigma_np`
  and `sigma_np`.
- `fixed_128_fixture` group test: **kept on the default path only.** The default `L = 10` output was
  recomputed for this version and is correct (both branches 48 correct + 32 error, purity .6,
  TPR .75, FPR .5). The `abs(x) == L` group test breaks at `L = 1`, so no `L = 1` claim may be made
  from the fixture until it uses explicit group labels.
- `s_gradient_numerator`: **fixed, and the earlier severity call was an understatement.** The
  hand-off record said the hardwired coefficients were those of `L = 10` and that only a non-default
  `L` was at risk. That is not what the defect is. The hardwired form is
  `2.5 q (.5 - p + .5 q) - .375 p (1 - p)`, which is exact on the `theta <= 0` branch at `L = 10`
  and wrong off it **at the default `L` too**: at `L = 10`, `theta = 0.31` it returns `0.869094`
  against a true `0.014833`, and at `theta = 1.0` it returns `0.598537` against `-0.024528` -- a
  sign inversion on the structured update, on the frozen contract's own parameter setting. The
  cause is the piecewise structure of `r_1`/`r_L` (for `theta >= 0` the budget saturates the
  large-magnitude group and the remainder carries a `(1-q)/(1-p)` ratio), not a dimensionality
  constant. A replacement "general closed form" proposed during this pass is also wrong for the
  same reason and was **not** installed. The function now derives the numerator from
  `acceptance_rates` and is regression-tested against `stopped_gradient * acceptance_rate` over
  `theta` in `[-4, 4]` and `L` in `{1, 2, 4, 10, 25, 100}` (max deviation `1.1e-16`). Contract
  values are unchanged: `theta <= 0` at `L = 10` reproduces the old numbers exactly.

The default-parameter results in `PLAN.md` do **not** depend on any of the three. The general path
(`stopped_gradient`, `acceptance_rates`, `flow_readouts`, `find_balance`) was recomputed for this
version and matches the frozen numbers, including `L = 1` (both `-0.1`) and `L = 2`
(`-0.15` / `-0.05`).

## What comes immediately after this version

1. Engineering fix task (see the dispatch record): remove the unrequested simulation, give the
   fixture explicit group labels, parameterise or domain-guard the numerator, add the failing-then-
   passing tests, and record the CPU environment with the result.
2. Engineering implementation of the minimum chain: shared pool forwarding, shared adapter, joint
   matching CLI path, one-step entry, gradient and `Delta theta` diagnostics, denominator fix in
   the analysis readout.
3. A fresh-context independent review of the frozen candidate after that work stabilises, not of
   this checkpoint.

## Standing limits

No experiment, no publication, and no part of the language-model study is complete. The older
manuscript draft in the wider project is not aligned with this main line, its experiment chapter is
empty, and no reviewer has examined this construction or the core logistic argument.

## Pilot change 2026-10-01: graph prompt and pool sampling temperature

Owner: Role B. Pilot evidence only, on pilot data (generator seed 2027, small splits); the
pilot data and pools are not confirmatory data. Machine: RTX 4070 Laptop GPU (8 GiB), Qwen3-1.7B at
the frozen pin, bf16.

1. **Graph prompt** (`rsi/tasks.py:graph_prompt`). With the graph as a JSON edge dump, 256/256
   sampled answers (64 prompts x 4) explained a BFS instead of replying with the list, so every
   answer was a `format` error and no pool could be matched. Of eight variants tried, a neighbour
   list plus one worked example gave 51% correct, 4/256 `format`, 37 `nonshortest`, ~24-token
   replies. The prompt uses no rng: instances and ids are unchanged, only the `prompt` field and the
   dataset file hashes change (`tests/test_gradient_reference_split.py` checks this against the
   015cf5e bytes). **Data generated before this change, including the
   `data-snapshot-20261002` snapshot, carries the old prompt and must be regenerated.**
2. **Pool sampling** (`configs/matched_pool.json`): temperature 1.3, top-p 1.0, top-k off, 8
   samples per prompt. At the repository default (0.7 / 0.8 / 20) a prompt's samples were nearly
   always one path (~1.1 distinct answers per prompt), so almost no task had a `nonshortest` and
   another error at one token length (J_E), and matching was infeasible at every K even on 256 x 8.
   On 128 prompts x 8: T = 0.7 -> 1 J_E task, T = 1.0 -> 2, T = 1.3 -> 4 (K = 16 feasible), with
   accuracy ~50% throughout. DEFAULTS are unchanged, so evaluation decoding is not affected.

Open: pool size per block for K = 64 is unmeasured (estimate ~16 J_E tasks needs ~512 prompts at
T = 1.3); the J_E rule (target and non-target error at one length) is the binding constraint and is
worth a design review; matching uses generation-time `completion_tokens` while training re-encodes
the response (8/2048 differ), which must be reconciled before an hf matched run.

## Multi-round extension 2026-10-01 (addition to the one-step comparison)

Decided by the team: the one-step R/S comparison stays the primary
readout; a multi-round extension is added, graph first, arithmetic after it works.

- 4 rounds first (8 if time allows), one training split per round (`train_00t`); 5 seeds now,
  10 if compute allows.
- Each round continues from the arm's previous adapter.
- Round 1 is the joint matched construction over the shared round-1 pool.  From round 2 each
  arm samples from its own model and selects its own K: R and S pools differ, and only the
  counts and rates are matched (K, C:E = 3:1, one example per prompt, common prompt set).
- Several optimizer steps per round: 16 (K = 64, 4 examples per step, one pass, fresh AdamW,
  lr 2e-4, deterministic kernels), `configs/matched_iterative.json`.  lr: 2e-4 as in arXiv
  2602.10014 (LoRA alpha/r = 2 as ours, 1 epoch per round), the closest published setup; the
  one-step study keeps 5e-5, since its first AdamW step has a fixed size and its readout is
  the direction h^T Delta theta.
- Later rounds sample the first 512 prompts of `train_00t`, 4 answers each, temperature 1.3.

Implementation: `run_iterative_experiment.py`, `rsi/iterative.py`.  GPU smoke test on seed-1
pilot data (1 block, 4 rounds, 128 x 4 later pools, then at lr 5e-5): complete in 18 minutes;
S had 20-23 prompts with a nonshortest error per round against E = 16, so 512 prompts is kept
as the later-round size.  The lr was raised to 2e-4 after that test, before any run on the
seed-2027 data, and is checked by a second smoke test on the same pilot data.

Evaluation (Role C): Pass@1 on eval_id / eval_ood after rounds 0-4 per arm and seed, as in
arXiv 2602.10014.  evaluate.py samples with the run's own generation settings, which here are
the pool settings (temperature 1.3); the evaluation decoding has to be fixed separately.

## Round-1 matching draws its prompts per block 2026-10-02

Decided by Role C; proposed for the team's review in the follow-up pull request
to #14.

Problem.  `rsi/matching.py` fills the quotas with the most constrained tasks first and broke
ties by task id.  Every task of an 8-answer pool has 8 candidates, so the tie-break decided:
the correct quota took the lowest task ids that have a correct answer.  The seed blocks' pools
cover nearly the same such prompts, so the blocks trained round 1 on largely the same correct
examples, and the matching seed did not change them.  On the H100 Qwen3-1.7B graph pools
(3 blocks x 384 prompts x 8), all 12 correct (prompt, answer) rows at K = 16 were the same in
every block, and a C = 48 quota filled the same way would have shared 46 of its 48 prompts.
On the Llama-3.2-3B graph pools, which reach K = 64, 24 of the 48 correct prompts were the
same in all three blocks.  The blocks differed mostly in their K/4 error examples (and the
one-step run trains every block with one seed), so they were not independent replicates of
round 1, and an interval over blocks was too narrow.

Decision.  Equally constrained tasks are taken in a random order drawn from --seed and the
block's own pool (`rsi.matching.pool_key`: its candidate ids and responses).  The role coin
flip for a task that can take either role, and the choice of a candidate inside a bucket, are
keyed the same way, so no two blocks share a random stream.  Unchanged: most constrained first
(tasks with one legal role, correct-only or error-only, before flexible ones), every hard
constraint, the K ladder and 3:1.  A block's matching depends on its pool and --seed only, not
on its name or row order.  The blocks now draw independently and overlap only by chance.  On
the Qwen pools no correct prompt is in all three blocks at K = 16 (12 before), and a C = 48
quota would share 3 (46 before; chance about 2).  The Llama-3.2-3B pools still reach K = 64,
with 1 of 48 correct prompts in all three blocks (24 before; chance about 3.5) and no identical
correct (prompt, answer) row (17 before).  Where a role has few candidates the overlap stays
high by necessity: the Qwen pools have 4-5 error-only prompts per block for E = 4, so their
error prompts largely coincide.

Effect.  The same pools now give other round-1 subsets than before.  A study whose round 1
already ran finishes on the code it started with (the resume identity binds the source hash);
results from before and after this change are not pooled.  Rounds 2+ already drew their prompts
uniformly per block, arm and round (`rsi/iterative.py`) and are unchanged.

## Answer length: a 2,048-token cap, and cut answers are never trained on 2026-10-02

Decided by Role C; proposed for the team's review in the pull request that makes
every setting explicit in the configs (`configs/README.md`).

Problem.  The matched configs capped answers at 256 new tokens, and an answer still running at
the cap was kept as if it had finished: it could be matched, selected and trained on, and
evaluation judged it without saying so.

Decision.  `max_new_tokens` 2,048 and `max_sequence_length` 4,096 in every matched config.  Every
generated row records `truncated` (no stop token within `max_new_tokens`).  A cut answer is judged
and kept in its pool, but never matched, selected or trained on; the pilot report and both
evaluators count it.  On hf a pool without the field is refused.  The cap cuts no real answer:
on the Qwen3-1.7B graph pools (H100, 3 x 3,072 answers at T = 1.3) the p99 is 54 tokens, and the
14 answers at the old 256 cap were all repetition loops; Llama-3.2 in the next entry.  A larger
cap only lets a loop run longer, and one looping answer holds its generate batch until the cap.

Effect.  Every matched config's hash changes, and pools drawn before this change are refused on
hf.

## Llama-3.2 pool sampling 2026-10-02

Decided by Role C; proposed for the team's review in the pull request that makes
every setting explicit in the configs (`configs/README.md`).

Problem.  The Llama configs carried Qwen's pool sampling (temperature 1.3, top-p 1.0, top-k
off).  Llama-3.2 degenerates under it: in the earlier GPU checks (at the former 256-token cap),
Llama-3.2-1B gave no correct graph answer and drifted into multilingual noise, and
Llama-3.2-3B was about 10% correct (Qwen3-1.7B: about 52%).

Sweep.  The study's graph data (seed 2027), the `03_pilot_check.sh` procedure on `dev`
(`sample_candidates.py` seed 0, then `pilot_report.py`), at most 2,048 new tokens, H100, this
change's code.  Llama-3.2-3B on all 512 prompts x 8, Llama-3.2-1B on the first 128 x 8.  "J_E
est." is the pilot's estimate of J_E prompts in a 2,048-prompt round-1 pool; K = 64 needs 16,
and the pilot asks for 20.

| Setting | T / top-p / top-k | 3B correct | format errors | cut at 2,048 | distinct answers per prompt | prompts with a correct answer | J_E est. | longest finished answer (tokens) |
|---|---|---|---|---|---|---|---|---|
| A | 0.6 / 0.9 / off | 24.7% | 1.2% | 51 / 4,096 | 2.82 | 196 / 512 | 196 | 121 |
| B | 0.8 / 0.95 / off | 22.4% | 1.0% | 38 / 4,096 | 3.79 | 210 / 512 | 324 | 213 |
| C | 1.0 / 0.9 / off | 22.4% | 0.9% | 27 / 4,096 | 4.39 | 235 / 512 | 348 | 377 |
| D | 1.0 / 1.0 / 50 | 19.4% | 3.1% | 1 / 4,096 | 5.28 | 230 / 512 | 376 | 1,361 |

| Setting | 1B correct | format errors | cut at 2,048 | prompts with a correct answer |
|---|---|---|---|---|
| A | 0.0% | 22.6% | 68 / 1,024 | 0 / 128 |
| B | 0.1% | 32.6% | 84 / 1,024 | 1 / 128 |
| C | 0.0% | 36.7% | 90 / 1,024 | 0 / 128 |
| D | 0.1% | 42.1% | 13 / 1,024 | 1 / 128 |

The cut answers are runaways, not long answers: 111 of the 117 cut 3B answers end in an exact
repetition (`"v846", "v846", ...`), and the others read are counting runs (`"v487", "v488",
...`) or endless vertex lists, as are the 1B ones.  No finished 3B answer comes near 2,048, so
a larger cap would only lengthen the loops.  Qwen3-1.7B loops too, more rarely: on the H100
graph pools at the former 256-token cap, 2-7 of 3,072 answers per block ran to the cap, all
repetitions, and almost no finished answer passed 128 tokens.  A cut
answer is judged but never matched or trained on (`truncated`, `rsi/experiment.py` ROW_FIELDS).

Decision.  Llama-3.2, both sizes, samples at temperature 0.6, top-p 0.9, top-k off (A), in the
round-1 pool config and in the later rounds (`configs/matched_{pool,iterative}_llama3.2-*.json`).
Accuracy is the pilot criterion that every setting misses, and A comes closest; A is Meta's own
`generation_config.json` for both models, not a value fitted to this sweep; and K = 64 stays
feasible with the pilot's margin.  The cost: A's samples are the least diverse (2.82 distinct
answers per prompt) and loop the most (1.2% of answers).  C is the alternative: 2.3 points
less accurate, half the loops, 1.6 more distinct answers per prompt.  The Qwen configs are
unchanged.

Open.  Llama-3.2-3B stays below the 30-70% accuracy criterion at every setting tried, so on
graph it fails the pilot check; that is a prompt or task question, not a sampling one, and a
Llama study needs the team's decision on it first.  Llama-3.2-1B is not usable on graph with
this prompt at any setting tried (at most 0.1% correct).  Its configs carry the 3B sampling so
the two sizes decode alike.

Effect.  The Llama configs' hashes change.  No Llama study has run, only pilots, so nothing is
re-run.

## Multi-round pools: 2,048 prompts x 8 answers in every round 2026-10-02

Decided by Role B with the team.

Decision.  Every round of the multi-round experiments samples all 2,048 prompts of its split x 8
answers, the round-1 pool's size, instead of 512 x 4 in rounds 2 to 4.  The four Qwen3-1.7B
experiments (graph and arithmetic, each unaudited and with budgeted auditing, adaptive / 16) run
from `experiments/`, each with its own `config.json`, `evaluation.json` and `settings.sh`, so a
teammate can change one experiment's settings without touching the others (`experiments/README.md`).
One config per experiment serves its round-1 pools, shared adapter, h and training, since the
round-1 and later sampling settings are now the same.  An audited experiment reuses its unaudited
sibling's round-1 pools when both were drawn with identical base, data and sampling
(`experiments/pool_check.py`), so the comparison starts from the same answers.

Effect.  A new study for each experiment (`md-<task>-<audited|unaudited>-qwen3-1.7b-2048x8-v01`);
`configs/matched_iterative*.json` and `scripts/multiround/` are unchanged.  Sampling per experiment
grows about fourfold (about 574,000 answers instead of about 143,000).

## Graph may start below the 30% accuracy band 2026-10-02

Decided by Role C.  Llama-3.2-3B's graph pilot (`03_pilot_check.sh`, `dev` 512 x 8,
T 0.6 / 0.9) is 24.5% correct, below the pilot's 30-70% criterion; on the other criteria that bear
on validity it passes (format errors 1.4%, J_E estimate 188 per 2,048 prompts; its 57 cut answers
are excluded by design).  Starting graph training from a somewhat lower accuracy is acceptable:
round-1 matching reaches K = 64, and a small end-to-end run of stages 04-10 (3 blocks x 192
prompts, 2 rounds) completed every stage with round 1 at K = 64.  Llama-3.2-1B (about 0% correct)
still cannot run: round-1 matching is infeasible at every K.

## The C:E ratio and the per-prompt token tolerance are config fields 2026-10-02

Decided by Role C.

Change.  `matching.error_fraction` (each arm's K holds E = K x error_fraction wrong answers and
C = K - E correct ones; 0.25 is 3:1) and `matching.token_tolerance` (in round-1 matching, R's and
S's wrong answers on one prompt may differ by at most this many supervised tokens; 0: the same
count; `null`: no limit) are config fields.  Round-1 matching (one-step and multi-round), the
later-round selection (the ratio), the check of the rows each arm trains on, and the pilot's J_E
estimate and K = 64 criteria read them, and one-step evaluation (`rsi/paired_evaluation.py`)
checks a run against its own values.  With a nonzero tolerance S draws uniformly among the target
errors that have a non-target error within the tolerance (always the shortest, as at tolerance 0,
would make S's errors systematically shorter than R's), and R draws uniformly among the prompt's
errors within the tolerance of S's (S's own included, as before).
With the defaults, 0.25 and 0, the selection is the frozen one (PLAN.md 3.4) draw for draw: checked
against main on 2,500 random pool sets (1,634 feasible) and 15,000 later-round selections, all
identical.  Every matched config and the graph experiments state 0.25 and 0; the two arithmetic
experiments state 0.25 and `null`.

Why.  With tolerance 0 a prompt takes an error role only if it holds S's target error and a
non-target error of exactly the same length.  Answers that show their steps (arithmetic since #23) are 100-500
tokens long and rarely share a length.  What remains with a tolerance: R's and S's error lengths
may differ, and length can go with the error type; every example weighs the same in the loss
(the per-example token mean), but its gradient need not.

What it buys, on the pilot pools (`dev`; prompts that can take an error role, per 2,048):

| Pool | tolerance 0 | 5 | 20 | no limit | largest K at 3:1 / 7:1, no limit |
|---|---|---|---|---|---|
| graph, Qwen3-1.7B | 44 | 64 | 160 | 160 | 64 / 64 |
| graph, Qwen3-4B | 20 | 36 | 96 | 104 | 64 / 64 |
| graph, Llama-3.2-3B | 188 | 324 | 364 | 368 | 64 / 64 |
| arithmetic, Qwen3-1.7B | 0 | 0 | 0 | 0 | none / none |
| arithmetic, Qwen3-4B | 0 | 0 | 0 | 0 | none / none |
| arithmetic, Llama-3.2-3B | 0 | 4 | 4 | 8 | 32 / 64 |
| arithmetic, Llama-3.2-1B | 4 | 4 | 8 | 12 | 32 / 64 |

Open.  The tolerance does not rescue the Qwen arithmetic experiments: their answers almost never
hold S's target error (`ignore_parentheses`: 1 and 0 of 4,096), so S's quota cannot be filled at
any tolerance or ratio; that needs another S target.  The Llama arithmetic numbers rest on 1-3
prompts of 512 and are rough: Llama-3.2-3B's 8 per 2,048 is exactly the 8 that K = 64 needs at
7:1 and the 8 that K = 32 needs at 3:1, with no margin.

## Arithmetic runs on Llama-3.2 for now 2026-10-03

Decided by Role C.  With the reasoning prompt (#23), Qwen3-1.7B and Qwen3-4B almost
never make S's target error (`ignore_parentheses`: 1 and 0 of 4,096 pilot answers), so round-1
matching is infeasible at every K, ratio and token tolerance (the C:E and tolerance entry above).
Training cannot start with an incomplete quota (round 1 stops before training, nothing is relaxed),
and starting it anyway would not test anything: with no target errors S's training set is R's.
So the arithmetic experiments run on Llama-3.2-3B and Llama-3.2-1B for now
(`experiments/arithmetic_{unaudited,audited}_llama3.2-{3b,1b}`; their J_E estimates at no token
tolerance are 8 and 12 per 2,048 prompts, rough).  The Qwen arithmetic experiments stay in
`experiments/` for a later S target the models produce.  The Llama-3.2-3B graph experiments
(`experiments/graph_{unaudited,audited}_llama3.2-3b`) run beside the Qwen3-1.7B ones (graph may
start below 30%, the entry above).  Each Llama experiment is its Qwen sibling with the Llama base,
its pin and Llama's sampling (T 0.6, top-p 0.9); everything else is the same.

## Synchronize the verified L=1 fixture correction 2026-10-04

The local correction at `628b10301fa1b5e0295eb1d2fb05c4845c3aafc9` was not included when
`theory_checks.py` was uploaded. As a result, the repository's numerical checker rejected its
own L=1 fixture. This branch ports the same implementation and regression tests onto main
`7295a6bdc8b1869739cff909577421d01a56a926`, with no change to the population formulas or LLM
training. The two group identities are explicit even when their numerical magnitudes coincide.
The L=2 description now agrees with the gradients: both initial updates improve sampling accuracy.
Numerical root finding is described as a bracketed calculation, not a uniqueness proof.

The current reproduction commands and source-bound verification record are in
[THEORY_STATUS.md](THEORY_STATUS.md). This synchronization does not incorporate the separate
one-step experiment branch, publish figure bundles, or replace the author's manuscript source.
## The one-step comparison runs on every experiment's pools 2026-10-04

Proposed by the repository maintainer for the team's review in the pull request that adds
`experiments/one_step.sh`.

Problem.  The one-step R/S comparison is the primary readout (the multi-round entry above, 2026-10-01,
and the one-step protocol of the paper's section 4.4).  `EXPERIMENT_PLAN.md` and `MULTIROUND_EXPERIMENT_INSTRUCTION.md`
listed it as an optional last step, and `experiments/` had no stage for it, so no current experiment
ran it.  The earlier Qwen3-4B one-step run predates the per-block matching of 2026-10-02: its three
blocks share 56 to 58 of their 64 prompts, the defect for which the Qwen3-1.7B multi-round v01 was
set aside.  It is not used.

Decision.
- Every unaudited experiment runs the one-step comparison on its own round-1 pools and shared
  adapter, at its five seed blocks (`experiments/one_step.sh`; `run_matched_experiment.py` now accepts
  b00 to b04 on hf as well as b00 to b02).  The pools, the shared adapter and the multi-round run's
  round-1 matching are inputs; the one-step entry never draws or creates them, because pools cannot
  be redrawn identically on another GPU and a one-step run on other pools would not be paired with
  the multi-round run.  The round-1 matching is the same call on the same pools with the same seed,
  so R's and S's training sets equal the multi-round round 1's.
- The one-step config is the experiment's `config.json` with `rounds` 1 and the learning rate 5e-5,
  as decided above for the one-step study.
- A one-step run is accepted only if it passes `experiments/one_step_check.py`: one step in R and S,
  none in null and null unchanged, the gradient composition in every arm, no two blocks sharing half
  their prompts, and R/S sets equal to the multi-round round-1 matching (failed when that matching is
  not available).  The checks run after training and again before every reuse, evaluation or figure;
  a run that fails one is not evaluated.
- Evaluation is the multi-round one: greedy Pass@1 on all of eval_id and eval_ood.  The null arm takes
  no step and is the base, so its Pass@1 is round 0's.  `evaluate_one_step.py` (sampled, several draws
  per question) stays available, but it needs per-block training seeds and a handoff this entry does
  not write.
- Qwen3-4B graph (`experiments/graph_unaudited_qwen3-4b`) uses the Qwen3-4B round-1 pools on which a
  multi-round run already exists.  They were drawn before `repetition_stop` existed, so they are
  imported as they are (`experiments/import_pools.py`), with that run's shared adapter and round-1
  matching: their metas are not rewritten, the experiment's config states `repetition_stop` null as
  drawn, and its `evaluation.json` states the current loop stop for the greedy answers.  Its one-step
  results are reported as first pools drawn without the loop stop, with current matching, training
  and evaluation.  Drawing new Qwen3-4B pools would make another study, not one paired with that
  run.  The experiment serves the one-step comparison only: it has no audited sibling, and `run.sh`
  does not start a multi-round run from it.
- The one-step comparison has one entry, `experiments/one_step.sh`.  `scripts/multiround/07_one_step.sh`
  (three blocks, no acceptance checks) and `RUN_ONE_STEP` in `run_all.sh` are removed, and `05` no
  longer computes a one-step h.
- `diagnostics.json` records each example's gradient norm and loss (`per_sample`), which the paper
  reports, besides the sums it already held.
- Five blocks.  The one-step comparison runs and is reported on b00 to b04, not the three blocks
  (b00 to b02) of PLAN.md and of the manuscript's one-step protocol: the five are the multi-round
  runs' blocks, so each one-step block is paired with the multi-round block that uses the same
  round-1 pool, and with the 95% t interval over blocks, five blocks give about half the half-width
  of three (t(0.975, 4)/sqrt 5 = 1.24 sd against t(0.975, 2)/sqrt 3 = 2.48 sd).  Decided before any
  one-step result on these pools exists; the manuscript's three-block statements are to be changed
  to five.  The maintainer approved it on 2026-10-04.
- Nothing is drawn from an incomplete evaluation.  The figures and tables are written only when
  every split of `evaluation.json` has been evaluated on all 11 checkpoints (the base, and R and S of
  every block), on all its questions and on the adapters the arms hold now
  (`experiments/one_step_check.py`, check 5); a failure names the missing checkpoints.  The script
  prints DONE only when every stage succeeded.
- The paper's one-step results table (task, model, block, arm, ID and OOD error, the ID error
  change against null, h^T Delta theta, |Delta theta|_2) is written by
  `one_step_check.py --table15`; the error is 1 - greedy Pass@1.  `experiments/one_step_figures.py`
  draws h^T Delta theta, |Delta theta|_2, the per-example gradient contributions and the error change
  against null, as vector PDF with a 300 dpi PNG and the plotted numbers.  `plot_multiround.py` is
  unchanged: the one-step Pass@1 figures are drawn by it, as the multi-round figures are.
- Cost records (`rsi/cost.py`): one per stage attempt of `compute_reference_gradient.py`,
  `run_matched_experiment.py` (per arm) and `evaluate_multiround.py` (per checkpoint and split),
  failed attempts included, with monotonic wall time (GPU work settled at both ends), allocated GPU
  hours, generated responses and output tokens, and memory peaks; a measurement not taken is null,
  never 0.  `experiments/one_step_cost.py` counts each cost_id once and writes the paper's cost table
  rows (shared preparation and pool, R only, S only, null and evaluation, deduplicated total).  The
  reused round-1 pools are inputs: their sampling is counted by the multi-round run that drew them.
  The sampling and multi-round entries do not write cost records yet.
