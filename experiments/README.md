# Experiments: graph and arithmetic, unaudited and audited

Multi-round R/S experiments, each in its own folder with its own editable settings. Run any of
them on your GPU with one command. Each runs from its own config, so changing one experiment's
hyperparameters never affects another. Each comes as a pair, unaudited and audited (adaptive, 16
correctness checks per arm per round); the two differ only in auditing and share their round-1
pools. `graph_unaudited_qwen3-4b` has no audited sibling in this release, but uses the same
generation, multi-round training, evaluation and one-step drivers.

| Experiments (unaudited / audited) | Model | Task | Status |
|---|---|---|---|
| [`graph_unaudited/`](graph_unaudited/), [`graph_audited/`](graph_audited/) | Qwen3-1.7B (T 1.3) | graph shortest paths; S's target error is `nonshortest` | runnable |
| [`graph_unaudited_llama3.2-3b/`](graph_unaudited_llama3.2-3b/), [`graph_audited_llama3.2-3b/`](graph_audited_llama3.2-3b/) | Llama-3.2-3B-Instruct (T 0.6, top-p 0.9) | graph | runnable; starts at 24.5% correct, below the pilot's 30%, which the team accepts (`matched-dynamics/DECISION.md`) |
| [`arithmetic_unaudited_llama3.2-3b/`](arithmetic_unaudited_llama3.2-3b/), [`arithmetic_audited_llama3.2-3b/`](arithmetic_audited_llama3.2-3b/) | Llama-3.2-3B-Instruct | arithmetic expressions, worked out step by step and ending with `Answer: N`; S's target error is `ignore_parentheses` | runnable; round 1 most likely at K = 16 at 3:1 (a rough pilot estimate) |
| [`arithmetic_unaudited_llama3.2-1b/`](arithmetic_unaudited_llama3.2-1b/), [`arithmetic_audited_llama3.2-1b/`](arithmetic_audited_llama3.2-1b/) | Llama-3.2-1B-Instruct | arithmetic | runnable; K = 16 or 32 at 3:1 (rough); 9.5% format errors |
| [`graph_unaudited_qwen3-4b/`](graph_unaudited_qwen3-4b/) (no audited sibling) | Qwen3-4B (T 1.3) | graph | Generates inputs and runs four rounds from its config; then `one_step.sh` uses this run's inputs. Optional original-input replay is described in [OPTIONAL_INPUTS.md](../OPTIONAL_INPUTS.md). |
| [`arithmetic_unaudited/`](arithmetic_unaudited/), [`arithmetic_audited/`](arithmetic_audited/) | Qwen3-1.7B | arithmetic | **not runnable yet**: working the expression out, Qwen almost never makes S's target error (1 of 4,096 pilot answers), so round-1 matching is infeasible at every K and `run.sh` stops before training (exit 2) after drawing the pools. Kept for a later S target |

**One-step comparison.** The paper's primary readout, R/S/null with one update each, runs on an unaudited
experiment's own round-1 pools and shared adapter: `bash experiments/one_step.sh EXPERIMENT`
([`ONE_STEP_EXPERIMENT_INSTRUCTION.md`](../ONE_STEP_EXPERIMENT_INSTRUCTION.md)).

## What one experiment does

Two verifier rules pick which of the model's own answers it trains on:

- **R** takes wrong answers at random.
- **S** takes one kind of wrong answer: for graph, a valid path that is not the shortest; for
  arithmetic, the value obtained by ignoring the parentheses.

Both rules accept the same numbers of correct and wrong answers. Each seed block (5 by default,
b00 to b04) is an independent replicate.

1. **Round 1.** One pool per seed block: all 2,048 prompts of `train_001` x 8 answers from the
   base model. R and S are matched from the same pool: the same prompts, the same correct answers,
   and per prompt token counts within `matching.token_tolerance` (graph: 0, the same count;
   arithmetic: no limit). The training set per arm is K = 64 (48 correct + 16 wrong at
   `matching.error_fraction` 0.25), or 32 or 16 if 64 is infeasible.
2. **Rounds 2 to `rounds`** (graph 8, arithmetic 4). Each arm samples all 2,048 prompts of
   `train_00t` x 8 answers **from its own previous model** and selects its own K with the same C:E.
   R takes any error, S takes the target error.
3. **Training each round.** LoRA (rank 8, alpha 16, `q_proj`/`v_proj`), continuing the arm's
   adapter: K/4 AdamW steps at lr 2e-4, 4 examples per step, one pass.
4. **Audited experiments only.** After selection, each arm spends 16 correctness checks on its
   selected examples. Audited wrong answers are removed and the unaudited ones are down-weighted
   (`matched-dynamics/AUDITING.md`).
5. **Evaluation.** Pass@1 with one greedy answer per question, on all 2,000 `eval_id` and all 1,000
   `eval_ood` questions. It covers round 0 (the base) and every block, arm and round.
6. **Figures.** Pass@1 vs round for R and S (mean over blocks, 95% interval), S - R paired by
   block, the target-error rate, and Pass@1 per difficulty.

## Quick start (Linux GPU machine)

```bash
# from the root of this repository:
bash scripts/multiround/00_setup.sh ~/venvs/rsi        # once: venv, CUDA torch, tests
export PY=~/venvs/rsi/bin/python                       # the venv's Python
export RSI_ROOT=~/rsi-work                             # where data and outputs go (this is the default)

# Recommended before a task's first main run (10-20 min); read the report it prints:
bash experiments/pilot.sh arithmetic_unaudited_llama3.2-3b

# One experiment, end to end, detached:
nohup bash experiments/run.sh graph_unaudited > graph_unaudited.log 2>&1 &
```

Run the others the same way, one after another on one GPU, or side by side on several GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash experiments/run.sh graph_unaudited                  > graph_unaudited.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash experiments/run.sh graph_audited                    > graph_audited.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash experiments/run.sh arithmetic_unaudited_llama3.2-3b > arithmetic_unaudited_llama3.2-3b.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash experiments/run.sh arithmetic_audited_llama3.2-3b   > arithmetic_audited_llama3.2-3b.log 2>&1 &
```

The Llama-3.2-3B graph and the Llama-3.2-1B arithmetic pairs run the same way. Llama-3.2-3B needs a
GPU of at least 16 GB.

When both experiments of a task are evaluated, draw them on one figure:

```bash
bash experiments/compare.sh graph_unaudited graph_audited
bash experiments/compare.sh arithmetic_unaudited_llama3.2-3b arithmetic_audited_llama3.2-3b
```

**After an interruption**, run the same command again. Finished stages are skipped, training resumes
from its last completed round, and evaluation from its last evaluated checkpoint.

**Stages.** `run.sh EXPERIMENT` runs all of them in order. `run.sh EXPERIMENT STAGE ...` runs only
the stages named.

| Stage | What it does |
|---|---|
| `data` | the study's questions for the task (seed 2027, shared by every experiment; fingerprint-checked) |
| `model` | downloads the pinned base model |
| `pools` | one round-1 pool per seed block (or reuses the sibling experiment's, see below) |
| `adapter` | the shared LoRA initialisation (seed 0) and the reference gradient h at it |
| `train` | the multi-round run (`rounds` in `config.json`) for every block and both arms |
| `evaluate` | Pass@1 of every checkpoint |
| `figures` | this experiment's figures |

## Each experiment's settings: what to edit

Every experiment folder holds three files. Edit them to change that experiment only.

**`config.json`**: every hyperparameter of sampling, training, auditing and the R/S selection. Each entry refuses a
config that leaves out a field it reads, so every value is visible here. The JSON format allows no
comments; the fields are:

| Field | Value | Meaning |
|---|---|---|
| `model`, `revision` | Qwen/Qwen3-1.7B @ 70d244cc... | the base model; must match the pin file in `settings.sh` |
| `dtype`, `device` | bfloat16, cuda | how the model is loaded |
| `rounds` | 8 (graph), 4 (arithmetic) | training rounds, at most 8 (the data has `train_001` to `train_008`); raising it on a finished run extends that run (below) |
| `generation.prompts_per_pool` | null (= all 2,048) | prompts per pool, every round; a number N takes the first N |
| `generation.candidates` | 8 | answers per prompt, every round |
| `generation.batch_size` | 16 | prompts per generate call; raise it on a large GPU to sample faster |
| `generation.temperature`, `top_p`, `top_k` | 1.3, 1.0, null (off) | sampling |
| `generation.max_new_tokens`, `max_sequence_length` | 2048, 4096 | an answer cut at the cap is never trained on |
| `generation.repetition_stop` | span 128, max_period 32 | stops an answer whose last 128 tokens repeat with a period of at most 32 tokens (a loop); it is marked truncated and never trained on. null: off |
| `training.lora_rank`, `lora_alpha`, `lora_dropout`, `target_modules` | 8, 16, 0.0, q/v | the LoRA adapter (`lora_dropout` must be 0) |
| `training.learning_rate` | 2e-4 | AdamW rate per round |
| `training.epochs`, `batch_size`, `effective_batch_size` | 1, 2, 4 | passes per round; micro-batch; examples per optimizer step |
| `training.gradient_checkpointing` | true | saves GPU memory |
| `audit.policy`, `budget`, `weighting` | none/0 or adaptive/16, true | budgeted auditing: `none`, `uniform`, `balanced` or `adaptive`; checks per arm per round; reweight unaudited examples |
| `matching.error_fraction` | 0.25 | the share of wrong answers in each arm's K, every round: 0.25 is C:E = 3:1 (48 + 16), 0.125 is 7:1 (56 + 8), 0.5 is 1:1 |
| `matching.token_tolerance` | 0 (graph), null (arithmetic) | round 1: how many supervised tokens R's and S's wrong answers on one prompt may differ; 0 = the same count, null = no limit |

`configs/README.md` explains each field in more detail.

**`evaluation.json`**: `splits` (`eval_id`, `eval_ood`, or `dev`) and `batch_size` (answers per
generate call). A batch too large for the GPU is halved automatically. The batch size does not
change the answers.

**`settings.sh`**:

| Setting | Meaning |
|---|---|
| `TASK` | `graph` or `arithmetic` |
| `STUDY` | the study id recorded in every pool and run; change its `-v01` suffix when you restart with other settings |
| `PIN_FILE` | the base model's pin; it must name `config.json`'s model and revision (other models: `matched-dynamics/base_pin_*.json`) |
| `SEEDS` | the seed blocks, `"0 1 2 3 4"` |
| `SHARE_POOLS_WITH` | the sibling experiment whose round-1 pools this one may reuse (below); empty: never |
| `LABEL` | the name on the figures |

**Machine settings** are environment variables, the same for every experiment: `PY` (the venv's
Python), `RSI_ROOT` (default `~/rsi-work`) and `CUDA_VISIBLE_DEVICES` (default 0).

**To try another setting**, copy a folder (for example `cp -r experiments/graph_audited
experiments/graph_audited_b32`), edit the copy and give it its own `STUDY`. Then run
`bash experiments/run.sh graph_audited_b32`. Its outputs go to their own folder.

### After editing a config that already ran

A run records its config's hash, and the pools, the shared adapter and h are bound to what made
them. The scripts refuse stale outputs rather than mixing settings. To restart an experiment with
the edited settings:

1. Change `STUDY`'s suffix in its `settings.sh`, for example `-v01` to `-v02`.
2. Set aside what the old settings made, under `$RSI_ROOT/experiments/NAME/`:
   - `training` or `audit` fields changed: move `out/`, `shared_adapter/` and `reference/` aside.
   - `matching` fields changed: move `out/` and `reference/` aside (h is bound to the whole config;
     pools and the shared adapter stay valid). Keep two siblings' `matching` equal if their round-1
     selections should stay paired.
   - `generation`, `model` or `revision` changed: also move `pools/` aside.
   - `evaluation.json` changed: move `out/evaluation_greedy_*/` aside.
3. Run `run.sh` again.

**Only `rounds` raised** (for example 4 to 8): change nothing else and keep `STUDY`. Run `run.sh`
again, or just its `train evaluate figures` stages. The `train` stage extends the finished run
(`run_iterative_experiment.py --extend-rounds`): it keeps rounds 1 to 4 and runs 5 to 8 from each
arm's round-4 model. Nothing in a round depends on how many rounds follow it, so the result is
the run that 8 rounds from the start would have made. `out/experiment.json` keeps the 4-round
run's identity and code under `extensions`. The `evaluate` stage adds the new checkpoints.

- Only a finished run is extended. Finish an interrupted one first with the config and the code it
  began with (the clone it ran in).
- Any other edit is refused, and lowering `rounds` is refused.

## Shared round-1 pools between an experiment and its sibling

By default, `graph_unaudited` and `graph_audited` have identical configs except for auditing, and
so do the two arithmetic experiments. Drawing the round-1 pools twice would cost GPU time and
would make the two runs start from different answers.

So the `pools` stage first looks in the sibling's folder (`SHARE_POOLS_WITH`). It copies a block's
pool only if `experiments/pool_check.py` confirms it was drawn with exactly this experiment's
model, revision, dtype, device, every `generation` field, seed and data. Otherwise the experiment
draws its own.

With the defaults, the audited run therefore starts from the same round-1 answers as the
unaudited one. Its round-1 selection before the audit is the same, so the comparison is paired.
The order of the two runs doesn't matter: whichever runs second reuses the first one's pools.
Edit one sibling's `generation` settings and the two no longer share anything.

Before training, the `train` stage checks every pool against `config.json` again. It stops if a
pool was drawn with other settings, for example after `config.json` was edited.

## Outputs

Under `$RSI_ROOT/experiments/NAME/`:

```text
pools/b00.jsonl ...         round-1 pools, each with .meta.json (how it was drawn)
shared_adapter/             the LoRA initialisation both arms start from
reference/                  h (h.json, h.pt)
out/
  experiment.json           status, K, identity, completed rounds, stopped arms, extensions
  matching/                 round-1 matched subsets
  b00/R/ b00/S/ ...         per arm: run.json; round_00t/ with selection, training set, adapter/,
                            complete.json (loss, steps, |Delta theta|, h^T Delta theta, audit counts);
                            pool.jsonl from round 2; audited runs also pre_audit.jsonl, audit.json
  evaluation_greedy_eval_id/, evaluation_greedy_eval_ood/
                            per checkpoint: Pass@1 overall and per difficulty, error types; every answer
figures/NAME_eval_id_pass1.png, _by_difficulty.png, _summary.json/.csv (and _eval_ood_)
pilot/pilot_report.json     from pilot.sh
logs/                       every stage's output
```

The comparison figures from `compare.sh` are in `$RSI_ROOT/experiments/figures/`.

Snapshots of the team's own runs are in [`outputs/`](outputs/). Each holds the status, the figures and
the per-round summaries, without raw answers or weights.

## Time and GPU

Measured on an RTX 4070 Laptop (8 GB) with Qwen3-1.7B: about 4.5 answers/s when sampling at batch 16.
One experiment samples about 82,000 answers in round 1 (5 blocks x 2,048 x 8). Each later round
adds about 164,000 (5 blocks x 2 arms x 2,048 x 8): 492,000 at 4 rounds, 1,147,000 at 8.
Evaluation adds 1 + 10 x `rounds` checkpoints (41 at 4 rounds, 81 at 8), x 3,000 greedy answers.
At 4 rounds, that is roughly 1.5 to 2 days per graph experiment on such a laptop. A data-centre GPU
with a larger `generation.batch_size` is several times faster (estimate, not measured). Arithmetic
answers show their steps, so they are several times longer than graph answers (a short JSON path)
and arithmetic sampling and evaluation take correspondingly longer.

On one H100 at the configs' batch 16 and rounds, from the H100 runs of 2026-10-03 (estimates for
a whole experiment):

| Pair | Rounds | One later arm-round | Unaudited | Audited (reuses the round-1 pools) |
|---|---|---|---|---|
| Qwen3-1.7B graph | 8 | about 21 min | about 1.5 days | about 1.5 days |
| Llama-3.2-3B graph | 8 | about 11.5 min | about 18 h | about 16 h |
| Llama-3.2-3B arithmetic | 4 | about 1 h 55 min | about 3.5 days | about 3.3 days |
| Llama-3.2-1B arithmetic | 4 | about 1 h 50 min | 3.5 to 4.5 days | 3.5 to 4.5 days |

Most of it is the later rounds' pools. Evaluation adds about 2 min per checkpoint for Llama-3.2-3B
graph and about 6 min for Qwen3-1.7B graph. For arithmetic it is not measured yet and is estimated
at 15 to 45 min, because greedy answers loop to the 2,048-token cap more often than sampled ones.

Training itself is minutes per round and peaks at about 5.5 GB of GPU memory. Sampling memory grows
with `generation.batch_size`, because an answer that loops runs to `max_new_tokens` and holds its
batch (`repetition_stop` ends most loops early). **On an 8 GB GPU, batch 16 can run out of memory**
(it did before the loop stop): set `generation.batch_size` to 8 there,
and `batch_size` in `evaluation.json` to 8 as well (evaluation would halve 32 to 8 by itself, but
only after two failed attempts).
Raising it is the main speed-up on a large GPU. Keep it equal in two
siblings if they should share their pools: it is a `generation` field, and sampling depends on it.

## Exit codes and problems

`run.sh` exits 0 when every stage finished. The `train` stage:

- **0** complete.
- **3** complete, but an arm stopped at some round because its pool could not fill its quota. This
  is recorded in `out/experiment.json` and evaluation continues.
- **2** round-1 matching is infeasible: even K = 16 cannot be met in every block's pool. It stops
  before training and writes an infeasibility certificate in `out/`. Run `pilot.sh` and report its
  result.
- **1** failed; the log says why.

Other refusals name their cause, for example:

- a pool drawn with other settings;
- an `out/` of another study (the message gives the `mv` command);
- a config without a field a stage reads;
- a base other than the pin.

On Windows, the code caps PyTorch's GPU memory cache at 95% of the card, because the Windows
driver otherwise lets it spill into system RAM until a step fails. Nothing is set on Linux.

## What to send back

For each experiment, `$RSI_ROOT/experiments/NAME/` without the `adapter/` folders (keep those on
your machine), plus `$RSI_ROOT/experiments/figures/`. Include your GPU model and every edit you made
to the three files, or just send the edited files.
