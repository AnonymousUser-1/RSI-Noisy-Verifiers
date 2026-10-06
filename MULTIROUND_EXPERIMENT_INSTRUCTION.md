# Multi-round R/S experiment: instructions

How to run the multi-round experiment on your own GPU, for Qwen3-1.7B, Qwen3-4B,
Llama-3.2-3B-Instruct or Llama-3.2-1B-Instruct. The scripts are in
[`scripts/multiround/`](scripts/multiround/). The per-experiment folders in
[`experiments/`](experiments/README.md) run the same stages from their own configs.

## 1. What the experiment is

Two verifier rules, **R** and **S**, pick which self-generated answers a model trains on. They accept
the same number of correct and wrong answers (same TPR/FPR), but differ in *which* wrong answers:

- **R** takes wrong answers at random.
- **S** takes "valid but not shortest" paths (`nonshortest`), the graph blind spot the paper is about.

The claim under test: matched error rates do not fix the direction of learning, because S
concentrates accepted errors on more influential examples.

There are two experiments on the same data:

| | One-step (primary readout) | Multi-round (this document) |
|---|---|---|
| Rounds | 1 optimizer step | 4 rounds, K/4 steps each (16 at K = 64) |
| What R and S share | One answer pool; same prompts, same correct answers, same token lengths | Round 1: as one-step. Rounds 2+: only the counts and rates |
| Learning rate | 5e-5 | 2e-4 |
| Main readout | h^T Delta theta (direction of the update) | Pass@1 on held-out questions after each round |
| Entry | `run_matched_experiment.py` | `run_iterative_experiment.py` |

## 2. Quick start

```bash
# from the root of this repository:

bash scripts/multiround/00_setup.sh ~/venvs/rsi       # once: venv, CUDA torch, tests
export PY=~/venvs/rsi/bin/python
export MODEL_TAG=qwen3-4b                             # qwen3-1.7b | qwen3-4b | llama3.2-3b | llama3.2-1b
export TASK=graph                                     # graph | arithmetic

bash scripts/multiround/01_make_data.sh               # the study's questions (shared by all models)
bash scripts/multiround/02_download_model.sh          # the pinned model
bash scripts/multiround/03_pilot_check.sh             # does this model fit the protocol? (~10 min)
# read $WORK/pilot/pilot_report.json (WORK: ~/rsi-work/$MODEL_TAG, or $MODEL_TAG-arithmetic); go ahead only if section 6 says so

nohup bash scripts/multiround/run_all.sh > run_all_$MODEL_TAG.log 2>&1 &
# pools, adapter, h -> 06 unaudited 4 rounds -> 08 Pass@1 + figures -> 09 audited 4 rounds -> 10 Pass@1 + figures
```

`run_all.sh` can be run again after any interruption: finished stages are skipped, the
multi-round runs resume from their last completed round and the evaluations from their last
evaluated checkpoint. `RUN_AUDIT=0` stops after 08 (no audited run). The one-step comparison is
run from `experiments/` ([`ONE_STEP_EXPERIMENT_INSTRUCTION.md`](ONE_STEP_EXPERIMENT_INSTRUCTION.md)).

## 3. Before you start

**Branch.** Use `main` (`b-multiround-instructions` is merged and out of date: it has no
`llama3.2-1b` and no fixed Llama chat-template date). It contains the neighbour-list prompt, the pool settings, the real-model path of `run_matched_experiment.py`,
`run_iterative_experiment.py`, and the per-model configs and pins.

**Python.** 3.10 to 3.12, with `requirements-gpu.txt` (torch >= 2.6, transformers 4.57.1, peft 0.17.1,
accelerate, safetensors) plus `pytest`. `00_setup.sh` builds this. Python 3.13 is not supported by
the repository.

**GPU.** One GPU per run; nothing here is multi-GPU. Measured on an RTX 4070 Laptop (8 GiB) with
Qwen3-1.7B: peak 4.5 GiB while sampling, 6.3 GiB in a one-step arm. For Qwen3-4B (about 8 GB of
weights in bf16) and Llama-3.2-3B (about 6.4 GB), use a GPU with at least 16 GB, preferably 24 GB.

**Hugging Face access.** Qwen models are public. **The Llama-3.2 Instruct models are gated**: request
access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct (or `-1B-Instruct`) and log in once with
`huggingface-cli login` (or `export HF_TOKEN=...`) before `02_download_model.sh`.

**Settings.** All paths and choices are in [`scripts/multiround/config.sh`](scripts/multiround/config.sh)
and can be overridden by exporting the variable:

| Variable | Default | Meaning |
|---|---|---|
| `TASK` | `graph` | `graph` or `arithmetic`; selects the data, S's target and the output folder |
| `MODEL_TAG` | `qwen3-1.7b` | `qwen3-1.7b`, `qwen3-4b`, `llama3.2-3b` or `llama3.2-1b`; selects the configs and the pin |
| `PY` | `python` | the venv's Python |
| `RSI_ROOT` | `$HOME/rsi-work` | root of all outputs |
| `DATA` | `$RSI_ROOT/data/$TASK` | the dataset (shared by every model) |
| `WORK` | `$RSI_ROOT/$MODEL_TAG` (graph), `$RSI_ROOT/$MODEL_TAG-arithmetic` | everything for this model and task |
| `STUDY` | `md-main-$TASK-iter4-$MODEL_TAG-v02` | study id of the unaudited run; keep it fixed for a study, change the suffix for a new one |
| `AUDIT_STUDY` | `md-audit-$TASK-iter4-b16-$MODEL_TAG-v01` | study id of the audited run |
| `RUN_AUDIT` | `1` | `0`: `run_all.sh` skips the audited run and its evaluation |
| `SEEDS` | `0 1 2 3 4` | seed blocks b00 ... b04 |
| `CUDA_VISIBLE_DEVICES` | `0` | which GPU |
| `EVAL_CONFIG` | `configs/matched_evaluation.json` | evaluation splits and batch size (08, 10); on a large GPU point it at a copy with a larger `batch_size` |

Every sampling, length and training setting -- including the later-round pool size (512 prompts x 4
answers) -- is in the model's config files (pool, multi-round, audited multi-round; section 4) and
the evaluation config, not in these variables or the scripts; see [`configs/README.md`](configs/README.md).
A variable that used to hold such a setting (`EVAL_SPLITS`, `EVAL_BATCH`, `LATER_PROMPTS`,
`LATER_SAMPLES`) now stops the scripts with a message naming the config that holds it.

## 4. Models, pins and configs

Every entry refuses a model or revision other than the pinned one. The scripts select the pin with
the `RSI_BASE_PIN` environment variable; every run records the pin file and its sha256.

| `MODEL_TAG` | Model | Pinned revision | Pin file | Configs |
|---|---|---|---|---|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `configs/pins/base_pin.json` | `configs/matched_pool.json`, `configs/matched_iterative.json` |
| `qwen3-4b` | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `configs/pins/base_pin_qwen3-4b.json` | `configs/matched_pool_qwen3-4b.json`, `configs/matched_iterative_qwen3-4b.json` |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | `configs/pins/base_pin_llama3.2-3b.json` | `configs/matched_pool_llama3.2-3b.json`, `configs/matched_iterative_llama3.2-3b.json` |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | `configs/pins/base_pin_llama3.2-1b.json` | `configs/matched_pool_llama3.2-1b.json`, `configs/matched_iterative_llama3.2-1b.json` |

The Instruct model is used for Llama because the pipeline encodes every prompt with the model's chat
template. Each model has three configs: the pool config, the multi-round config, and the audited
multi-round config, `configs/matched_iterative_audit.json` (Qwen3-1.7B) or
`configs/matched_iterative_audit_<MODEL_TAG>.json`: the multi-round config with
`"audit": {"policy": "adaptive", "budget": 16, "weighting": true}`. Each model's configs are its
own: they state every setting explicitly (a stage refuses a config that leaves a field it reads to
the code defaults) and may differ between models, e.g. the Llama pool sampling.
[`configs/README.md`](configs/README.md) lists every field, the stage that reads it and each
model's values. To change a setting, change the config file and
start a new study: the config hash is part of every run's identity.

## 5. Data

`01_make_data.sh` generates the graph dataset with `generate_data.py --task graph --seed 2027`:

| Split | Prompts | Used for |
|---|---|---|
| `train_001` ... `train_008` | 2,048 each | round t samples from `train_00t` (4 rounds use 001 to 004) |
| `dev` | 512 | the pilot check only |
| `calibration` | 500 | not used here |
| `eval_id` / `eval_ood` | 2,000 / 1,000 | held-out evaluation |
| `gradient_reference` | 64 | the reference gradient h only |

These are the same questions and ids as the published snapshot
(`data/snapshots/generator-381f340-seed2027` on `data-snapshot-20261002`); only the prompt text
differs. The snapshot's files still carry the old prompt and must not be used for this experiment.
The precomputed reference answers on that branch stay valid (they are keyed by id).

**Prompt** (`rsi/tasks.py:graph_prompt`): the graph as a neighbour list, one worked example, and
"Return ONLY a JSON list of node names ... your entire reply must be the JSON list". With the old
JSON edge-list prompt Qwen3-1.7B explained its reasoning instead of answering and 256/256 answers
were format errors; with this prompt about 52% are correct and under 1% are format errors.

File hashes depend on the platform's line ending, so the script checks two OS-independent
fingerprints instead (ids `8057e1ac...`, prompts `78894a85...`). Both must print `OK`.

## 6. Pilot check (do this first for every new model)

`03_pilot_check.sh` draws 512 `dev` prompts x 8 answers with the study's pool settings
(`--phase pilot` reads `dev` only) and writes `pilot/pilot_report.json`:

| Field | Go when | Why |
|---|---|---|
| `truncated_answers` | 0 (`go.no_truncated_answers`) | an answer still running at `max_new_tokens` is never matched or trained on; any here means the model loops at this sampling setting |
| `format_error_rate` | <= 0.05 | unparseable answers are wasted samples |
| `accuracy` | 0.30 to 0.70 | arXiv 2602.10014: only a moderate starting accuracy improves under self-training |
| `eligible_estimate_at_round1_pool` | >= 16 (>= 20 for margin) at 3:1 | round-1 matching needs E = 64 x `matching.error_fraction` eligible prompts per block for K = 64 (16 at 3:1); the round-1 pool is `train_001` (2,048 prompts), or its first `generation.prompts_per_pool` |
| `distinct_answers_per_prompt` | report it | matching needs varied answers |

Reference values for Qwen3-1.7B (2,048 prompts x 8): accuracy 0.52, format 0.002, 1.40 distinct
answers per prompt, 20 eligible prompts, K = 64.

**If a criterion fails, stop.** The prompt, temperature and ladder are part of the study's
configuration; changing them makes a different study.

## 7. The procedure, step by step

### 7.1 Round-1 pools (`04_round1_pools.sh`)

One shared pool per seed block, drawn from the **base model** with `sample_candidates.py`:

| Setting | Value |
|---|---|
| Prompts | all 2,048 of `train_001` |
| Answers per prompt | 8 |
| Sampling | the pool config's temperature / top-p / top-k (Qwen3: 1.3 / 1.0 / off; Llama-3.2: 0.6 / 0.9 / off); non-thinking chat template |
| Length | at most `max_new_tokens` = 2,048 new tokens, 4,096 in all. Finished answers are far shorter (graph p99 about 50 tokens); an answer still running at 2,048, or stopped earlier because its last 128 tokens repeat (a loop, `generation.repetition_stop`), is marked `truncated`, judged, and never matched or trained on |
| Batch | 16 |
| Seed | `--seed` = 0 ... 4 for b00 ... b04 |
| Identity | `--split train_001 --study-id $STUDY --phase main` |

The pool's `.meta.json` records the split, config, decoding, seed, base pin and code. Temperature 1.3
is a pilot decision: at Qwen's default (0.7 / 0.8 / 20) a prompt's 8 samples were nearly always the
same path (1.1 distinct per prompt) and matching was infeasible at every K. Llama-3.2 degenerates
at 1.3 and samples at its own default, 0.6 / 0.9 / off; at that setting Llama-3.2-3B is still
below the 30% accuracy criterion on graph.

### 7.2 Shared adapter and h (`05_adapter_and_h.sh`)

- **Shared adapter** (`make_shared_adapter.py`): one LoRA initialisation every arm of every block
  starts from (rank 8, alpha 16, dropout 0, `q_proj` and `v_proj`, seed 0), recorded with its
  parameter hash and base revision.
- **Reference gradient h** (`compute_reference_gradient.py`): the raw (not normalised) gradient of
  the response-token NLL of the 64 `gradient_reference` prompts' reference answers, at the shared
  adapter. It is bound to the config hash: this is the multi-round config's h
  (`reference_iterative/`); the one-step comparison computes its own.

### 7.3 Multi-round run (`06_multiround.sh`, `run_iterative_experiment.py`)

Blocks run one after another, all 4 rounds of a block before the next. Within a round, R then S.

**Round 1, joint matching** (`rsi/matching.py`, as in the one-step study). From each block's
pool, built for all blocks at once:

- K is the largest of 64, 32, 16 that **every** block can meet; if none, the run stops with exit 2
  (an infeasibility certificate is written; nothing is relaxed).
- C:E is `matching.error_fraction`'s (0.25 = 3:1, so K = 64 is 48 correct + 16 wrong answers), one
  answer per prompt, weights 1.0.
- R and S have the same prompts, the same correct answers, and, per prompt, supervised token counts
  (response tokens + EOS from the model's tokenizer) within `matching.token_tolerance` (0: the same).
- An error prompt must hold a `nonshortest` error and a non-target error within that tolerance of
  it. S takes a `nonshortest` error (at tolerance 0 the one at the first such length; otherwise
  uniformly among those that qualify), R takes one uniformly among the errors within the tolerance
  of S's, S's own included.
- Which prompts take the correct and the error roles is a random draw from each block's own pool
  and the seed, so each block's examples are an independent draw that overlaps the other blocks'
  only by chance.
- Both arms train from the shared adapter.

**Rounds 2 to 4, each arm on its own** (`rsi/iterative.py`).

- Each arm samples the first 512 prompts of `train_00t`, 4 answers each, from **its own previous
  round's model**: the same prompts and sampling seed for both arms, different models.
- Each arm selects its own K answers, the K of round 1 with the same C:E (48 correct + 16 wrong at
  K = 64 and 3:1), one per prompt:
  - R takes its E error prompts uniformly among prompts with any error, then one error in each.
  - S takes them among prompts with a target error (`nonshortest` for graph, `ignore_parentheses` for
    arithmetic), then one target error in each.
  - Both take C = K - E correct answers from the remaining prompts.
- Only the counts and rates are matched; the pools and answers differ.
- If an arm cannot fill its quota, that arm stops at that round and the run records it (exit 3); the
  other arm continues.

**Training in every round:**

| Setting | Value |
|---|---|
| Start | round 1: shared adapter; round t: the same arm's round t-1 adapter |
| Steps | 16 = one pass over the 64 examples, 4 per step (2 x 2 gradient accumulation) |
| Optimizer | AdamW, lr 2e-4, weight decay 0, new optimizer state every round |
| Loss | per-example response-token mean NLL, averaged over the 4 examples of a step |
| Gradient clipping | norm 1.0 |
| Precision | bf16 model, LoRA parameters in fp32, gradient checkpointing on |
| Determinism | deterministic CUDA kernels (`rsi/determinism.py`): two identical runs give bitwise-equal gradients, about 4% slower |
| Recorded | loss, steps, \|Delta theta\|, and Delta theta projected on h at the shared adapter |

lr 2e-4 follows arXiv 2602.10014, which uses LoRA at the same alpha/r = 2 for iterative
self-improvement; the earlier 5e-5 is a full fine-tuning rate and barely moves a 16-step LoRA round.

### 7.4 One-step run

The one-step comparison (the primary readout) runs on an `experiments/` experiment's own round-1
pools, shared adapter and round-1 matching: `bash experiments/one_step.sh EXPERIMENT`, as described
in [`ONE_STEP_EXPERIMENT_INSTRUCTION.md`](ONE_STEP_EXPERIMENT_INSTRUCTION.md). These per-model
scripts have no one-step stage.

### 7.5 Evaluation and the audited run (`08_evaluate.sh`, `09_audit.sh`, `10_evaluate_audit.sh`)

**08: Pass@1 of the unaudited run** (`evaluate_multiround.py`, then `plot_multiround.py`).

- One greedy answer (temperature 0) per held-out question: all 2,000 of `eval_id` (in distribution)
  and all 1,000 of `eval_ood` (larger graphs). Each checkpoint is loaded once for both.
- Round 0 is the base model, evaluated once; the shared adapter starts with LoRA B = 0, so it
  computes the same function. Then every block, arm and round 1 to 4: 41 checkpoints at 5 blocks.
- Per checkpoint: Pass@1 overall and per difficulty, the error types, and the target-error rate.
- `protocol.json` freezes the decoding and the questions; a rerun with other settings is refused.
- Figures, one set per split: (a) Pass@1 vs round for R and S, mean over the seed blocks with a 95% t interval;
  (b) S - R per round, paired by block; (c) the `nonshortest` rate; and Pass@1 per difficulty.

**09: the audited run.** The same as 06, with the model's audit config (section 4).

- In every round from round 1, each arm spends 16 correctness checks on its selected K examples.
  The adaptive policy gives one check per difficulty first, then favours difficulties with more
  errors in the arm's previous round.
- An audited wrong answer is removed. Unaudited answers are weighted by their difficulty's
  estimated error rate. A round therefore trains on up to K examples and may take fewer steps.
- R and S get the same budget and policy. Round-1 matching holds **before** the audit, not after it.
- It uses the same pools, shared adapter and seed blocks as 06, its own h (`reference_audit/`,
  bound to the audit config), and its own study and output folder (`out_audit/`).

**10: Pass@1 of the audited run**, as 08, then the figures comparing the unaudited and audited runs.

## 8. Outputs

Under `$WORK` (default `~/rsi-work/<model>`):

```text
pilot/                         dev_pool.jsonl (+ .meta.json), pilot_report.json
pools/b00.jsonl ... b04.jsonl  round-1 pools, each with .meta.json
shared_adapter/                adapter + shared_adapter.json (parameter hash, base)
reference_iterative/           h.pt, h.json, per_sample.jsonl
out_multiround/
  experiment.json              status, K, identity, completed rounds, stopped arms
  matching/                    ladder.json and matched_subsets.json (round 1)
  b00/R/  b00/S/ ...           one directory per arm, laid out as evaluate.py reads a run:
    run.json                   config (rounds = 4), data path and hash, bindings
    round_001/ ... round_004/  selection.json, training.jsonl, adapter/, complete.json; pool.jsonl from round 2
    finished.json              when all 4 rounds ran (stopped.json instead if the arm stopped)
  evaluation_greedy_eval_id/   08, and the same for eval_ood: protocol.json; round_000.json/.jsonl (base) and
                               bXX_ARM_round_00t.json/.jsonl (Pass@1 summary / every answer, judged)
reference_audit/               h for the audit config
out_audit/                     09: the audited run, laid out as out_multiround/; each round also has
                               pre_audit.jsonl and audit.json; audit_ledger/ per arm;
                               evaluation_greedy_eval_id/ and _eval_ood/ from 10
figures/                       ${TASK}_${MODEL_TAG}_unaudited_SPLIT_* (08) and _unaudited_vs_audited_SPLIT_*
                               (10): _pass1.png, _by_difficulty.png, _summary.json/.csv
logs/                          every stage's output
```

`training.jsonl` holds only prompt, response, weight and token count: no correctness label enters
a training file. Labels and the selection audit are in `selection.json`.

## 9. Evaluation and metrics

Following arXiv 2602.10014, the main metric is **Pass@1 on held-out questions after every round**.
The protocol: **greedy decoding, one answer per question, all 2,000 `eval_id` and
all 1,000 `eval_ood` questions**, at rounds 0 (the base) to 4, for each arm and seed block, for the
unaudited and the audited run. `08_evaluate.sh` and `10_evaluate_audit.sh` do this (section 7.5).
Report the per-round trajectory, final minus initial for each arm, and the S minus R difference
per round, paired by seed block, with a confidence interval; `_summary.csv` holds these numbers.
A mechanism check: the `nonshortest` rate in the evaluation answers (figure panel c), and in each
arm's own training pools per round (`pool.jsonl`).

Do not use `evaluate.py` for these runs: it samples with the run's own generation settings, which
are the pool settings (temperature 1.3).

## 10. Time and resume

Measured with Qwen3-1.7B on an RTX 4070 Laptop:

| Stage | Time |
|---|---|
| Pilot pool (512 x 8) | about 10 min |
| One round-1 pool (2,048 x 8) | 38 min (5 blocks: about 3.2 h) |
| Shared adapter / one h | 18 s / 22 s |
| Multi-round, one block (4 rounds, 512 x 4 later pools) | about 45 min (5 blocks: about 3.5 to 4 h) |
| Audited multi-round (09) | about as long as 06 |
| Evaluation of one run (41 checkpoints x 3,000 greedy answers, 08 or 10) | H100, graph, with `repetition_stop`: Llama-3.2-3B 211 s per checkpoint (about 2.4 h per run), Qwen3-1.7B 333 s (about 3.8 h); without the rule Llama-3.2-3B took 3,544 s per checkpoint |

Larger models take longer roughly in proportion to their size (estimate, not measured). A large GPU
evaluates much faster with a larger `batch_size`: point `EVAL_CONFIG` at a copy of
`configs/matched_evaluation.json` with a larger one.

**Interruptions.** Re-run `run_all.sh` (or the stage script). Pools with a `.meta.json`, the adapter
and h are kept; the multi-round run resumes with `--resume`, keeps every round with
`complete.json`, and moves an interrupted round aside as `round_00t.incomplete-N`. A resume is
refused if anything that defines the run changed (config, code, data, pools, adapter, h, seed,
later pool size, study or phase). Do not edit code in the checkout while a study is running: the
source hash is part of that identity.

**Exit codes** of `06_multiround.sh` and `09_audit.sh`: 0 complete; 2 round-1 matching infeasible
(stop and report); 3 complete with a stopped arm (report which); 1 failed (`experiment.json` in the
run's folder has the reason). `run_all.sh` evaluates a run that exits 0 or 3 and stops otherwise.

## 11. What to send back

The whole `$WORK/out_multiround/` and `$WORK/out_audit/` without
the `adapter/` directories (keep those on your machine), including their `evaluation_greedy_*/`
folders; `$WORK/figures/`; the pool `.meta.json` files, `pilot/pilot_report.json`, `logs/`, and your
GPU model. Do not change `STUDY` or `AUDIT_STUDY` within a study.
