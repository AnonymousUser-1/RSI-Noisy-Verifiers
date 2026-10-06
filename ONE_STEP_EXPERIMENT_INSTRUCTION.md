# One-step R/S experiment: instructions

How to run the one-step comparison on any GPU, for each of the five model–task combinations in
this study. It runs from the same `experiments/` folders as the multi-round experiments and reads
the experiment's round-1 pools, shared adapter and round-1 matching, so the one-step comparison
trains on exactly the examples the multi-round run trains on in its first round. Setup, data,
model pins and the environment are as in [`experiments/README.md`](experiments/README.md).

## 1. What the experiment is

Two verifier rules, **R** and **S**, select which self-generated answers a model trains on. Both
accept the same numbers of correct and wrong answers (equal TPR and FPR) and differ only in
*which* wrong answers they accept. **R** takes wrong answers at random. **S** takes the task's
target error (graph: `nonshortest`, a valid path that is not the shortest; arithmetic:
`ignore_parentheses`, the value obtained by ignoring the parentheses). The one-step comparison
asks in which direction **one** update moves the model when R's and S's training sets differ only
in the identity of their wrong examples.

This is the paper's primary readout: the one-step protocol of its section 4.4, whose results table
is in the appendix. Its main quantity is hᵀΔθ, where Δθ is the parameter change made by the single
update and h is the reference gradient: the gradient,
at the shared adapter, of the reference risk (the mean response-token negative log-likelihood of
the reference answers to the 64 held-out `gradient_reference` prompts, which are used for this
diagnostic only). hᵀΔθ is the first-order change of the reference risk caused by the update: a
negative value means the update lowers the reference risk, a positive value that it raises it.

| | One-step (this document) | Multi-round ([`experiments/README.md`](experiments/README.md)) |
|---|---|---|
| Arms | R, S and **null** (no update) | R, S |
| Seed blocks | b00–b04 (five independent replicates) | b00–b04 |
| Round-1 pools, shared adapter | the experiment's own | the experiment's own |
| Training set per arm | the multi-round run's round-1 set: K examples, C:E = 3:1 (K = 64: 48 correct + 16 wrong) | round 1 the same; later rounds selected per arm |
| What R and S share | the same prompts, the same correct answers, and on each error prompt wrong answers of the same token length (graph, `token_tolerance` 0) or of any length (arithmetic, `token_tolerance` null) | round 1 the same; later rounds only the counts and rates |
| Update | the gradient over all K examples, **one** AdamW step (weight decay 0, gradient clip 1.0) | K/4 steps per round, four rounds |
| Learning rate | 5e-5 | 2e-4 |
| Auditing | off | off (unaudited) or 16 checks per arm and round (audited) |
| h | computed under the one-step config | computed under the multi-round config |
| Readouts | hᵀΔθ; \|Δθ\|; per-example losses and gradient norms; greedy Pass@1 and error (1 − Pass@1) on `eval_id` and `eval_ood`, and the error change against null | greedy Pass@1 after every round |

K is the largest of 64, 32 and 16 at which every seed block can be matched; it is the same in all
blocks and equal to the multi-round round 1's K.

**The learning rate.** The first AdamW step has a fixed size: with the bias correction, each
coordinate moves by about the learning rate in the direction opposite to the sign of its gradient.
A different learning rate therefore rescales Δθ, and with it hᵀΔθ, without changing its sign or
the ratio between R and S. The one-step study keeps 5e-5 because its readout is this direction;
2e-4 is the multi-round rate (`matched-dynamics/DECISION.md`, multi-round extension). The Pass@1
readout is reported at 5e-5.

**Why the one-step run uses the multi-round run's pools.** Given the pools and the seed, the
round-1 matching is deterministic, but the pools themselves cannot be redrawn identically: sampling
is not reproducible across GPUs and software versions. A one-step run on newly drawn pools would
train on other examples than the multi-round run's round 1 and would not be paired with it. The
one-step run therefore reads the experiment's existing pools and shared adapter, repeats the
round-1 matching on them, and is accepted only if the result equals the multi-round run's round-1
matching (section 4, check 4).

## 2. The eight model–task combinations

The study covers Qwen3-1.7B (main model), Qwen3-4B (scale) and Llama-3.2-3B/1B-Instruct (a second
model family) on the graph and arithmetic tasks: eight combinations. Five can be matched and are
run; three cannot be matched and are reported as infeasible.

| Model | Task | One-step | Experiment folder | Round-1 pools |
|---|---|---|---|---|
| Qwen3-1.7B | graph | **run** | `graph_unaudited` | drawn by the experiment |
| Qwen3-4B | graph | **run** | `graph_unaudited_qwen3-4b` | imported from the folder `RSI-Qwen3-4B-Graph-paired-inputs-20261004/` (section 9) |
| Llama-3.2-3B | graph | **run** | `graph_unaudited_llama3.2-3b` | drawn by the experiment |
| Llama-3.2-3B | arithmetic | **run** | `arithmetic_unaudited_llama3.2-3b` | drawn by the experiment |
| Llama-3.2-1B | arithmetic | **run** | `arithmetic_unaudited_llama3.2-1b` | drawn by the experiment |
| Qwen3-1.7B | arithmetic | infeasible | `arithmetic_unaudited` (kept for a later S target) | — |
| Qwen3-4B | arithmetic | infeasible | — | — |
| Llama-3.2-1B | graph | infeasible | — | — |

**Why three are infeasible.** Each seed block must supply C correct examples and E error examples
at a common K; an error example needs an answer with S's target error, and the smallest K of the
ladder (16 at 3:1) needs four per block. Qwen3-1.7B and Qwen3-4B almost never make the arithmetic
target error (`ignore_parentheses` in 1 and 0 of 4,096 pilot answers), so S's quota cannot be
filled at any K, ratio or token tolerance. Llama-3.2-1B gives almost no correct graph answers (at
most 0.1% in every sampling setting tried), so the correct quota cannot be filled. The pilot data
are in [`matched-dynamics/DECISION.md`](matched-dynamics/DECISION.md).

## 3. Inputs

`one_step.sh` reads four inputs and never creates them. If one is missing, it stops and names it.

| Input | Path |
|---|---|
| the data | `$RSI_ROOT/data/TASK/` (`DATA_NAME` instead of `TASK` for an imported experiment) |
| the round-1 pools, one per seed block | `$WORK/pools/b00.jsonl` … `b04.jsonl`, each with its `.meta.json` |
| the shared adapter | `$WORK/shared_adapter/` |
| the multi-round run's round-1 matching | `$WORK/out/matching/matched_subsets.json`, or `$WORK/round1_reference/matched_subsets.json` |

`$WORK` is `$RSI_ROOT/experiments/EXPERIMENT`.

**On the machine of the multi-round run.** The experiment's own stages produce all four:

```bash
bash experiments/run.sh graph_unaudited
```

The train stage writes the round-1 matching before its first update, so the one-step run can
start as soon as `$WORK/out/matching/matched_subsets.json` exists.

**On another machine.** Download the pinned model there, then copy the four inputs from the machine
of the multi-round run, keeping their paths; place the matching in `round1_reference/`. Copy the
data rather than generating it again: each pool records the hash of the exact data files it was
drawn from, and those bytes depend on the platform.

```bash
bash experiments/run.sh graph_unaudited model
```

```text
$RSI_ROOT/data/TASK/                                       ->  $RSI_ROOT/data/TASK/
$WORK/pools/bNN.jsonl, $WORK/pools/bNN.jsonl.meta.json     ->  $WORK/pools/
$WORK/shared_adapter/                                      ->  $WORK/shared_adapter/
$WORK/out/matching/matched_subsets.json                    ->  $WORK/round1_reference/matched_subsets.json
```

Copied inputs are verified like local ones: each pool against the config and the data
(`experiments/pool_check.py`), the shared adapter by its parameter hash and base model, and the
matching by check 4.

**`graph_unaudited_qwen3-4b`.** Its four inputs are in the folder
`RSI-Qwen3-4B-Graph-paired-inputs-20261004/` and imported with `experiments/import_pools.py`
(section 9).

## 4. Acceptance checks

`experiments/one_step_check.py` checks a finished one-step run. Every check must pass:

1. **Steps.** R and S each took exactly one optimizer step; null took none and its parameters are
   unchanged (\|Δθ\| = 0).
2. **Gradient composition.** In every arm the gradient over all K examples equals the sum of the
   correct-example and the error-example gradients, G_C + G_E = G, parameter by parameter
   (`torch.allclose`, rtol 1e-4, atol 1e-6), as `diagnostics.json` records.
3. **Independent blocks.** No two blocks share half or more of their training prompts. Every block
   draws its own pool and its own matching, so two blocks share only a few prompts by chance; a
   large overlap means the blocks are not independent replicates.
4. **Same training sets as the multi-round round 1.** Each block's R and S sets equal those in the
   multi-round run's round-1 `matched_subsets.json`. Without that file the check fails.
5. **Complete evaluation** (after the `evaluate` stage). Every split in `evaluation.json` has been
   evaluated on all 11 checkpoints (the base, which is also null, and R and S of each of the five
   blocks); every result covers all of the split's questions; every R and S result was taken on the
   adapter the arm holds now. A missing checkpoint is named.

`one_step.sh` runs checks 1–4 after training and again before every later use: when `train` finds
a completed run, and at the start of `evaluate` and `figures`. It runs check 5 at the end of
`evaluate` and at the start of `figures`. The results are written to `out_one_step/checks.json`. A
run that fails checks 1–4 is not evaluated; no figure or table is drawn unless all five pass.

The script prints `DONE` only when every requested stage has succeeded; any failure stops it first.

## 5. Running the experiment

```bash
bash experiments/one_step.sh graph_unaudited
```

runs all four stages in order. Stages can also be named:

```bash
bash experiments/one_step.sh graph_unaudited reference train
```

```bash
bash experiments/one_step.sh graph_unaudited evaluate figures
```

The five combinations, one after another on one GPU:

```bash
for exp in graph_unaudited graph_unaudited_qwen3-4b graph_unaudited_llama3.2-3b arithmetic_unaudited_llama3.2-3b arithmetic_unaudited_llama3.2-1b; do bash experiments/one_step.sh "$exp"; done
```

| Stage | What it does |
|---|---|
| `reference` | h at the shared adapter under the one-step config. Kept once computed. |
| `train` | `run_matched_experiment.py` on all five blocks, then the acceptance checks. A completed run is not trained again; it is checked again. |
| `evaluate` | Greedy Pass@1 (`evaluate_multiround.py`), then check 5. Resumes; evaluated checkpoints are skipped. |
| `figures` | After check 5: the figures and tables of section 8, and the cost table. |

| Variable | Default | Meaning |
|---|---|---|
| `RSI_ROOT` | `~/rsi-work` | root of the data and outputs; the same as for the multi-round run |
| `PY` | `python` | the Python of the study's environment |
| `CUDA_VISIBLE_DEVICES` | `0` | the GPU |
| `ONE_STEP_LR` | `5e-5` | the one-step learning rate |

**GPU.** One GPU per run. Qwen3-1.7B peaks at about 6.3 GiB in a one-step arm; for Qwen3-4B and
Llama-3.2-3B use a GPU with at least 16 GB. Training takes minutes. The evaluation covers 11
checkpoints (the base and R and S of five blocks) × 3,000 greedy answers. From the per-checkpoint
rates of the multi-round evaluation on an H100, this is about 40 minutes for Llama-3.2-3B graph and
about an hour for Qwen3-1.7B graph; Qwen3-4B and the arithmetic experiments take longer (estimates).

**Outputs**, under `$WORK/`:

| Path | Content |
|---|---|
| `one_step_config.json` | the one-step config (section 6) |
| `reference_one_step/` | h under the one-step config |
| `pools_one_step/` | the copy of the pools the run reads |
| `out_one_step/matching/` | the round-1 matching, `ladder.json` and `matched_subsets.json` |
| `out_one_step/bNN/{R,S,null}/round_001/` | per arm: `training.jsonl`, `diagnostics.json` (hᵀΔθ, \|Δθ\|, clipping, gradient composition, `per_sample` losses and gradient norms), `complete.json`, and for R and S the adapter |
| `out_one_step/checks.json` | the acceptance checks |
| `out_one_step/evaluation_greedy_SPLIT/` | Pass@1 per checkpoint and every judged answer |
| `out_one_step/cost_records/`, `one_step_cost_records/` | one cost record per stage attempt (section 8) |
| `figures/EXPERIMENT_one_step_*` | the figures, tables and cost table of section 8 |
| `logs/one_step_*.log` | the output of every stage |

## 6. The one-step config

`one_step.sh` writes `one_step_config.json` from the experiment's `config.json`:

- `rounds` is 1;
- `training.learning_rate` is `ONE_STEP_LR`; the LoRA settings (rank, alpha, dropout, target
  modules) are the experiment's, and the multi-round batch settings are dropped;
- every other field (base model, sampling, matching, auditing) is the experiment's, so the pools and
  the shared adapter are checked against the one-step config exactly as against the multi-round
  config.

Auditing must be off: `one_step.sh` refuses an audited experiment and runs on the unaudited one of
the same model and task.

`one_step_config.json` is written once. If the experiment's `config.json` or `ONE_STEP_LR` later
give a different one-step config, the script stops; set aside `one_step_config.json`,
`reference_one_step/` and `out_one_step/` together before running with the new settings.

## 7. What the train stage does

`run_matched_experiment.py` matches the five pools with the round-1 matching of the multi-round run
(the same function, the same seed), then runs three arms per block:

- **R** and **S**: the gradient over the arm's K examples, each example weighted 1/K, then one AdamW
  step from the shared adapter;
- **null**: the same passes over R's examples without the step; its model is the shared adapter,
  which computes the same function as the base model (LoRA B = 0).

Each arm records `diagnostics.json`: hᵀΔθ, \|Δθ\|, the gradient before and after clipping and
whether clipping fired, the gradient composition, and each example's loss and gradient norm.

## 8. Evaluation and figures

`evaluate_multiround.py` evaluates under the experiment's `evaluation.json`: one greedy answer to
each question of `eval_id` (2,000) and `eval_ood` (1,000). The base model is evaluated once as
round 0; this is also the null arm's Pass@1. R and S are evaluated after their step.

The `figures` stage draws only from a complete evaluation (check 5). Under `$WORK/figures/`, with
the prefix `EXPERIMENT_one_step`:

| Output | Content |
|---|---|
| `_SPLIT_pass1.png`, `_SPLIT_by_difficulty.png`, `_SPLIT_summary.json/.csv` | Pass@1 of R and S against the base and S − R by block (`plot_multiround.py`, the same figures as the multi-round runs) |
| `_projection.pdf/.png/.json` | hᵀΔθ of R and S per block, paired; S − R per block with its mean and 95% interval |
| `_displacement.pdf/.png/.json` | \|Δθ\|₂ of R and S per block, paired |
| `_gradients.pdf/.png/.json` | per-example gradient contributions ‖g_i‖/K by arm, correct and error examples apart; the correct examples are the same in R and S |
| `_error_SPLIT.pdf/.png/.json` | per split, the error change against null of R and S per block, paired; S − R with its mean and 95% interval |
| `_table.csv` | one row per block and arm: matching statistics (K, C, E, N₊, N₋, TPR, FPR, yield, precision), hᵀΔθ, \|Δθ\|, clipping, mean loss, and per split Pass@1, error and error change against null |
| `_table15.csv` | the paper's one-step results table: task, model, block, arm, ID error, OOD error, ID error change against null, hᵀΔθ, \|Δθ\|₂ |
| `_cost.csv` (with `_failed.csv`, `_memory.csv`, `_pools.csv`) | the cost table below |

The new figures are vector PDF with a 300 dpi PNG beside them, and the plotted numbers are in the
`.json` of the same name. Intervals are Student-t over the five blocks. The error is 1 − greedy
Pass@1; its change against null is the arm's error minus the base's, so a negative value is an
improvement.

**Blocks.** All five blocks are run and reported. The paper reports the one-step comparison on the
same five blocks b00–b04 as the multi-round runs, so that each one-step block is paired with the
multi-round block that uses the same round-1 pool (`matched-dynamics/DECISION.md`).

**Cost.** `compute_reference_gradient.py`, every arm of `run_matched_experiment.py` and every
checkpoint and split of `evaluate_multiround.py` write one cost record per attempt (`rsi/cost.py`):
monotonic wall seconds with GPU work settled at both ends, the GPUs held and the allocated GPU hours,
the responses and output tokens generated, and the allocated and reserved memory peaks. Failed
attempts keep their records. `experiments/one_step_cost.py` counts each `cost_id` once and writes
the paper's cost table: shared preparation and pool (h; the reused round-1 pools generate no new
responses here and their sampling time is counted by the multi-round run that drew them), R only, S
only, null and evaluation, and the deduplicated total. A total with a missing measurement is labelled
an observed lower bound; failed attempts are listed apart and not summed; memory peaks are listed per
stage and never added.

## 9. Qwen3-4B: imported round-1 pools

The Qwen3-4B round-1 pools were drawn before the sampling field `generation.repetition_stop`
existed, and a multi-round run on them already exists. To pair the one-step comparison with that
run, the pools are used as drawn:

- `experiments/graph_unaudited_qwen3-4b/config.json` states `"repetition_stop": null`, the sampling
  the pools were drawn with; `rsi.common.drawn_generation()` reads a pool record without the field
  as drawn with it off.
- `evaluation.json` states the current repetition stop for the greedy evaluation, as in every other
  experiment.
- The experiment has no audited sibling, and `run.sh` does not start a multi-round run from it.

When these results are reported, state that the Qwen3-4B round-1 pools were drawn without the
repetition stop, and that matching, training and evaluation follow the current protocol.

**The input package** is the folder `RSI-Qwen3-4B-Graph-paired-inputs-20261004/` at the repository root. It holds the four inputs, each file
as the run that produced it wrote it, with the two reference gradients of the four-round runs, the
configurations, `PACKAGE_MANIFEST.json`, `SHA256SUMS.txt` and `verify_inputs.py`. Verify it (CPU only):

```bash
cd RSI-Qwen3-4B-Graph-paired-inputs-20261004 && python verify_inputs.py && cd ..
```

Then, from the repository root, **import it** once and run the one-step comparison as for any other
experiment:

```bash
python experiments/import_pools.py graph_unaudited_qwen3-4b RSI-Qwen3-4B-Graph-paired-inputs-20261004
```

```bash
bash experiments/one_step.sh graph_unaudited_qwen3-4b
```

`import_pools.py` copies every file unchanged and rewrites no metadata. It first verifies that
each pool hashes to its record and was drawn with the config's base model, data type, device and
every sampling field, with its block's seed, from `train_001` of exactly the exported data, and
that the shared adapter has the config's base model and LoRA settings. On any mismatch it stops
and writes nothing. It writes `$WORK/pools/IMPORTED.json` (the source, the sha256 of every file and
the sampling settings of the pools) and puts the matching in `$WORK/round1_reference/`.

## 10. Troubleshooting

**"the one-step run reads … and does not create them.  Missing: …"** One of the inputs of section 3
is not in place. Produce it with the experiment's stages, copy it as in section 3, or (Qwen3-4B)
import it.

**"its round-1 pools, shared adapter and data are imported, not drawn"** Run `import_pools.py` first (section 9).

**"… fails its checks"** At least one acceptance check failed; `out_one_step/checks.json` lists
them. Set the run aside (`mv out_one_step out_one_step_failed`) and train again. If check 4 fails
on a fresh run, the pools or the matching file are not those of the multi-round run; if check 2
fails, report it with `checks.json` and the arm's `diagnostics.json`.

**"the evaluation of … is not complete"** Check 5 failed: `out_one_step/checks.json` names the
missing checkpoints or splits, or the results taken on fewer questions or another adapter. Run the
`evaluate` stage again; it evaluates only what is missing. If a result was taken on another adapter,
the arm was trained again after it was evaluated: set the whole run aside and start again.

**"holds a run that did not complete"** Training was interrupted; the train stage does not resume.
Set `out_one_step/` aside and train again.

**"differs from what config.json and ONE_STEP_LR give now"** The config or the rate changed after
the one-step config was written (section 6).

**"is an audited experiment"** Run the unaudited experiment of the same model and task.

**Exit code 2 from the train stage** Round-1 matching is infeasible on these pools; nothing was
trained. Report it (`out_one_step/matching/ladder.json`).

## 11. What to send back

`$WORK/out_one_step/` without the `adapter/` folders (keep them on the GPU machine), including
`checks.json`, `cost_records/` and the `evaluation_greedy_*` folders; `$WORK/one_step_cost_records/`
and any set-aside `out_one_step_*/cost_records/` (failed attempts are part of the cost);
`$WORK/figures/EXPERIMENT_one_step*`;
`$WORK/one_step_config.json`; `$WORK/reference_one_step/h.json`; `$WORK/logs/one_step_*.log`; and the
GPU model.
