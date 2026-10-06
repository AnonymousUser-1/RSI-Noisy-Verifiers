# Configs

Every setting a matched-line run uses is in its config file. On a real model (hf) each stage
refuses a config that leaves a field it reads to the code defaults (`rsi.common.STAGE_FIELDS`,
`require_explicit`), and the scripts in `scripts/multiround/` pass no setting of their own. Every
pool meta and run record holds the resolved config (pools and runs also the config file's path and
sha256; h its hash; the shared adapter its LoRA settings). Nothing pins a value: to
change one, edit the config, record why in `matched-dynamics/DECISION.md`, and start a new study
(the config hash is part of every run's identity, and a pool drawn with other settings is refused:
04 stops on a round-1 pool whose recorded generation settings are not `POOL_CONFIG`'s, and 06/09 on
round-1 pools whose sampling -- temperature, top-p, top-k,
`max_new_tokens`, `max_sequence_length` -- is not `ITER_CONFIG`'s or that were not all drawn alike).
The exception is `matched_evaluation.json`, which no training run reads: its `batch_size` may change
at any time (point `EVAL_CONFIG` at a copy with a larger one on a large GPU).

Each model has three configs, named by `MODEL_TAG` in `scripts/multiround/config.sh`, and one
evaluation config serves every model:

| Config | Used by | For |
|---|---|---|
| `matched_pool[_TAG].json` (`POOL_CONFIG`) | `03_pilot_check.sh`, `04_round1_pools.sh` | the pilot and round-1 pools; also a standalone one-step config for `run_matched_experiment.py` (the study's one-step comparison runs from `experiments/one_step.sh`) |
| `matched_iterative[_TAG].json` (`ITER_CONFIG`) | `05_adapter_and_h.sh` (the shared adapter and its h), `06_multiround.sh` | the multi-round run: its later-round sampling and its training |
| `matched_iterative_audit[_TAG].json` (`AUDIT_CONFIG`) | `09_audit.sh` (its own h, then the audited run) | the multi-round run with budgeted auditing (`matched-dynamics/AUDITING.md`): `ITER_CONFIG` with `audit` adaptive / 16 |
| `matched_evaluation.json` (`EVAL_CONFIG`) | `08_evaluate.sh`, `10_evaluate_audit.sh` (`evaluate_multiround.py`) | Pass@1 evaluation: its held-out splits and batch size |

`matched_pool_audit.json` is the Qwen3-1.7B pool config with budgeted auditing on, for the one-step
run (`matched-dynamics/AUDITING.md`).

## Fields

| Field | Meaning | Read by (stage in `rsi.common.STAGE_FIELDS`) |
|---|---|---|
| `backend` | `hf` (a real model) or `mock` (plumbing tests only) | every stage except `pilot` |
| `model`, `revision` | the base model and its commit; must equal the pin file `RSI_BASE_PIN` names | every stage |
| `dtype`, `device` | how the model is loaded (`bfloat16`, `cuda`) | every stage except `pilot` |
| `rounds` | rounds of the multi-round run (iterative config: 4; a finished run is extended to a larger `rounds` with `run_iterative_experiment.py --extend-rounds`); the pool config has 1, which each one-step arm's run record carries for `evaluate.py` | `iterative`, `one_step` |
| `generation.candidates` | answers per prompt: per round-1 / pilot pool prompt (pool config, 8), per later-round prompt (iterative config, 4) | `pool`, `iterative` |
| `generation.prompts_per_pool` | how many prompts of the split a pool samples, the first N; `null` = all (pool config: all of `train_001` or `dev`; iterative config: 512 of `train_00t`). It applies to every pool the config draws, the 512-prompt `dev` pilot included, and a value above a split's size is refused | `pool`, `pilot`, `iterative` |
| `generation.batch_size` | rows per generate call (16); part of what reproduces a pool. One answer that loops holds its whole batch to `max_new_tokens`; on a small GPU a smaller batch bounds memory | `pool`, `iterative` |
| `generation.temperature`, `top_p`, `top_k` | sampling; `top_k` `null` is off | `pool`, `pilot`, `iterative` |
| `generation.max_new_tokens` | the generation cap, 2,048, far above any finished answer (graph: Qwen3-1.7B p99 54 tokens; Llama-3.2-3B at most 121 at its sampling, `DECISION.md`); what reaches it is a loop that would not stop at any cap. An answer that reaches it without a stop token is marked `truncated`: it is judged and kept in its pool, but never matched, selected or trained on, and evaluation counts it (`truncated_answers`) | `pool`, `iterative`; evaluation |
| `generation.repetition_stop` | `{"span": 128, "max_period": 32}`: a row stops once its last 128 generated tokens repeat with a period of at most 32 tokens -- a loop, which would otherwise run to `max_new_tokens` and hold its whole batch there (greedy evaluation and sampling alike). It is cut there and marked `truncated`, like a row cut at the cap (`completion_tokens` tells them apart). A finished answer never ends in such a tail, so none is changed; `null` turns it off (`matched-dynamics/DECISION.md`) | `pool`, `pilot`, `iterative`; evaluation |
| `generation.max_sequence_length` | prompt + `max_new_tokens` must fit (4,096); a longer training sequence is refused, never cut | `pool`, `iterative` |
| `training.lora_rank`, `lora_alpha`, `lora_dropout`, `target_modules` | the LoRA adapter, built from these by `make_shared_adapter.py` (from `ITER_CONFIG`). Every later stage loads that saved adapter and refuses a config whose values differ from its record (`rsi.shared_adapter.lora_mismatch`), so keep them equal in a model's configs | `adapter`, `reference`, `one_step`, `iterative` |
| `training.learning_rate` | one-step: the size of the single AdamW step (5e-5, pool config); multi-round: the per-round rate (2e-4, iterative config) | `one_step`, `iterative` |
| `training.epochs`, `batch_size`, `effective_batch_size`, `gradient_checkpointing` | multi-round training: passes per round, micro-batch, examples per optimizer step; steps per round = ceil(K / `effective_batch_size`) x `epochs` | `iterative` |
| `audit.policy`, `budget`, `weighting` | budgeted auditing (off: `none` / 0) | `reference`, `one_step`, `iterative` |
| `matching.error_fraction` | the share of wrong answers in each arm's K: E = K x error_fraction, C = K - E, in round 1 and every later round. 0.25 is C:E = 3:1 (48 + 16 at K = 64); 0.125 is 7:1; 0.5 is 1:1. It must give a whole E from 1 to K-1 at every K of the ladder (64, 32, 16) | `one_step`, `iterative`, `pilot` |
| `matching.token_tolerance` | round-1 matching: on an error prompt, S takes a target error that has a non-target error within the tolerance (at 0 the one at the first such length, the frozen rule; otherwise uniformly among those that qualify) and R an error at most this many supervised tokens longer or shorter than S's (0: the same length; `null`: no limit). The rows each arm trains on are checked against it. The pilot's J_E estimate uses it | `one_step`, `iterative`, `pilot` |
| `training.examples`, `verifier`, `frozen` | the legacy eight-round loop only; not read by the matched line | -- |
| `splits` (`matched_evaluation.json`) | the held-out splits evaluated (`eval_id`, `eval_ood`); each is its own `evaluation_greedy_SPLIT/` with its own protocol | `evaluate_multiround.py`, `08`/`10` (figures) |
| `batch_size` (`matched_evaluation.json`) | answers per generate call in Pass@1 evaluation; halved automatically when the GPU runs out of memory. Greedy answers do not depend on it beyond padding noise, so it is not part of the evaluation protocol and an evaluation may resume with another; each checkpoint's record holds it | `evaluate_multiround.py` |

## Fixed in code, not in the config

These are protocol rules rather than settings; they are listed so that nothing a run does is hidden.

| Rule | Where |
|---|---|
| K ladder 64 -> 32 -> 16, one answer per prompt (the C:E ratio and the token tolerance are config fields, `matching.*`) | `rsi/matching.py` (`K_LADDER`) |
| `lora_dropout` must be 0 (every matched entry: one-step, multi-round and h, `one_step.check_isolation`); one-step: every weight 1.0, one AdamW step, weight decay 0, gradient clip 1.0 | `one_step.py` |
| multi-round: a fresh AdamW per round, weight decay 0, gradient clip 1.0, per-example response-token mean loss | `rsi/backends.py` (`HFBackend.train`) |
| LoRA scaling `alpha / r` (`use_rslora` off) | `rsi/backends.py` (`lora_config_from_training`) |
| h is the raw gradient (`normalize` off) on `gradient_reference` | `compute_reference_gradient.py` |
| pilot go criteria: no truncated answers, format errors at most 5%, accuracy 30-70%, J_E estimate per 2,048 prompts at least the wrong answers K = 64 needs (64 x `matching.error_fraction`: 16 at 3:1; 1.25 x that with margin) | `scripts/multiround/pilot_report.py` |
| evaluation: one greedy answer per question (Pass@1), the run's own `max_new_tokens`; an answer cut there is judged as it stands and counted (`truncated_answers`) | `evaluate_multiround.py` (`eval_config`) |
| splits: pilot `dev`, round-1 pools `train_001`, round t `train_00t` (evaluation: `matched_evaluation.json`) | `scripts/multiround/03`, `04`, `run_iterative_experiment.py`, `08`, `10` |
| data: seed 2027 and the split sizes of `generate_data.py`; seed blocks `SEEDS`; matching and training `--seed 0`; adapter seed 0 | `scripts/multiround/01`, `config.sh`, `05`, `06` |

## Values

The config files are the source; `scripts/multiround/config_table.py --write` regenerates these
tables, and `tests/test_matched_pool_config.py` fails while they differ from the files.

Sampling (pools):

| Config | Model | Rounds | Prompts per pool | Answers per prompt | Batch | Max new tokens | Max sequence | Temperature | Top-p | Top-k | Repetition stop (span / period) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `matched_iterative.json` | Qwen/Qwen3-1.7B | 4 | 512 | 4 | 16 | 2048 | 4096 | 1.3 | 1.0 | off | 128 / 32 |
| `matched_iterative_audit.json` | Qwen/Qwen3-1.7B | 4 | 512 | 4 | 16 | 2048 | 4096 | 1.3 | 1.0 | off | 128 / 32 |
| `matched_iterative_audit_llama3.2-1b.json` | meta-llama/Llama-3.2-1B-Instruct | 4 | 512 | 4 | 16 | 2048 | 4096 | 0.6 | 0.9 | off | 128 / 32 |
| `matched_iterative_audit_llama3.2-3b.json` | meta-llama/Llama-3.2-3B-Instruct | 4 | 512 | 4 | 16 | 2048 | 4096 | 0.6 | 0.9 | off | 128 / 32 |
| `matched_iterative_audit_qwen3-4b.json` | Qwen/Qwen3-4B | 4 | 512 | 4 | 16 | 2048 | 4096 | 1.3 | 1.0 | off | 128 / 32 |
| `matched_iterative_llama3.2-1b.json` | meta-llama/Llama-3.2-1B-Instruct | 4 | 512 | 4 | 16 | 2048 | 4096 | 0.6 | 0.9 | off | 128 / 32 |
| `matched_iterative_llama3.2-3b.json` | meta-llama/Llama-3.2-3B-Instruct | 4 | 512 | 4 | 16 | 2048 | 4096 | 0.6 | 0.9 | off | 128 / 32 |
| `matched_iterative_qwen3-4b.json` | Qwen/Qwen3-4B | 4 | 512 | 4 | 16 | 2048 | 4096 | 1.3 | 1.0 | off | 128 / 32 |
| `matched_pool.json` | Qwen/Qwen3-1.7B | 1 | all | 8 | 16 | 2048 | 4096 | 1.3 | 1.0 | off | 128 / 32 |
| `matched_pool_audit.json` | Qwen/Qwen3-1.7B | 1 | all | 8 | 16 | 2048 | 4096 | 1.3 | 1.0 | off | 128 / 32 |
| `matched_pool_llama3.2-1b.json` | meta-llama/Llama-3.2-1B-Instruct | 1 | all | 8 | 16 | 2048 | 4096 | 0.6 | 0.9 | off | 128 / 32 |
| `matched_pool_llama3.2-3b.json` | meta-llama/Llama-3.2-3B-Instruct | 1 | all | 8 | 16 | 2048 | 4096 | 0.6 | 0.9 | off | 128 / 32 |
| `matched_pool_qwen3-4b.json` | Qwen/Qwen3-4B | 1 | all | 8 | 16 | 2048 | 4096 | 1.3 | 1.0 | off | 128 / 32 |

Training, auditing and the R/S selection (`--`: not read by that config's stages):

| Config | Learning rate | LoRA r / alpha / dropout | Target modules | Epochs | Micro-batch | Examples per step | Gradient checkpointing | Audit policy / budget / weighting | C:E (error fraction) | Token tolerance |
|---|---|---|---|---|---|---|---|---|---|---|
| `matched_iterative.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | none / 0 / on | 3:1 (0.25) | 0 |
| `matched_iterative_audit.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | adaptive / 16 / on | 3:1 (0.25) | 0 |
| `matched_iterative_audit_llama3.2-1b.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | adaptive / 16 / on | 3:1 (0.25) | 0 |
| `matched_iterative_audit_llama3.2-3b.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | adaptive / 16 / on | 3:1 (0.25) | 0 |
| `matched_iterative_audit_qwen3-4b.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | adaptive / 16 / on | 3:1 (0.25) | 0 |
| `matched_iterative_llama3.2-1b.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | none / 0 / on | 3:1 (0.25) | 0 |
| `matched_iterative_llama3.2-3b.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | none / 0 / on | 3:1 (0.25) | 0 |
| `matched_iterative_qwen3-4b.json` | 0.0002 | 8 / 16 / 0.0 | q_proj, v_proj | 1 | 2 | 4 | on | none / 0 / on | 3:1 (0.25) | 0 |
| `matched_pool.json` | 5e-05 | 8 / 16 / 0.0 | q_proj, v_proj | -- | -- | -- | -- | none / 0 / on | 3:1 (0.25) | 0 |
| `matched_pool_audit.json` | 5e-05 | 8 / 16 / 0.0 | q_proj, v_proj | -- | -- | -- | -- | adaptive / 16 / on | 3:1 (0.25) | 0 |
| `matched_pool_llama3.2-1b.json` | 5e-05 | 8 / 16 / 0.0 | q_proj, v_proj | -- | -- | -- | -- | none / 0 / on | 3:1 (0.25) | 0 |
| `matched_pool_llama3.2-3b.json` | 5e-05 | 8 / 16 / 0.0 | q_proj, v_proj | -- | -- | -- | -- | none / 0 / on | 3:1 (0.25) | 0 |
| `matched_pool_qwen3-4b.json` | 5e-05 | 8 / 16 / 0.0 | q_proj, v_proj | -- | -- | -- | -- | none / 0 / on | 3:1 (0.25) | 0 |

Evaluation (`matched_evaluation.json`, every model): splits eval_id, eval_ood, batch size 32.
