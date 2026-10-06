# One-step R/S/null comparison: graph, three models (2026-10-04)

The one-step comparison of [`ONE_STEP_EXPERIMENT_INSTRUCTION.md`](../../../ONE_STEP_EXPERIMENT_INSTRUCTION.md), run with
`bash experiments/one_step.sh EXPERIMENT` on the three unaudited graph experiments. All three are complete and pass all
43 acceptance checks (section 4 of the instruction, including check 5, complete evaluation).

| Folder | Model | Inputs (round-1 pools, shared adapter, data, round-1 matching) | GPU |
|---|---|---|---|
| [`graph_unaudited/`](graph_unaudited/) | Qwen3-1.7B | the H100 runs' (`h100-inputs-v1.zip`, the multi-round run's own) | RTX 4070 Laptop 8 GB (Windows 11) |
| [`graph_unaudited_qwen3-4b/`](graph_unaudited_qwen3-4b/) | Qwen3-4B | the folder `RSI-Qwen3-4B-Graph-paired-inputs-20261004/`, imported with `experiments/import_pools.py` | Colab A100-SXM4 40 GB |
| [`graph_unaudited_llama3.2-3b/`](graph_unaudited_llama3.2-3b/) | Llama-3.2-3B-Instruct | the H100 runs' (`h100-inputs-v1.zip`) | Colab A100-SXM4 40 GB |

- **Code:** `main` at `a908fb7`, clean checkout, in every run (`out_one_step/experiment.json`). transformers 4.57.1 and
  peft 0.17.1 everywhere; torch 2.14.1+cu126 on the laptop, 2.11.0+cu130 on Colab.
- **Settings:** the one-step config of section 6 (each experiment's `config.json`, rounds 1, lr 5e-5): five blocks
  b00–b04, K = 64 (48 correct + 16 wrong) in every block, one AdamW step per arm; greedy Pass@1 on all of `eval_id`
  (2,000) and `eval_ood` (1,000) for the base (= null) and R and S of each block.
- **Same training sets as the multi-round round 1** (check 4) in every block of every model.

## Results

Means over the five blocks; S − R is paired by block, mean ± 95% t half-width (n = 5). Error = 1 − greedy Pass@1. A
negative hᵀΔθ means the update lowers the reference risk.

| | Qwen3-1.7B | Qwen3-4B | Llama-3.2-3B |
|---|---|---|---|
| Base Pass@1, eval_id / eval_ood | 49.8% / 35.3% | 61.5% / 39.2% | 24.6% / 20.7% |
| R Pass@1, eval_id / eval_ood | 50.3% / 35.3% | 61.1% / 39.6% | 25.0% / 21.5% |
| S Pass@1, eval_id / eval_ood | 50.3% / 35.3% | 61.0% / 39.8% | 25.1% / 21.6% |
| **S − R, eval_id error (primary contrast)** | −0.0002 ± 0.0009 | +0.0006 ± 0.0036 | −0.0009 ± 0.0028 |
| S − R, eval_ood error | +0.0002 ± 0.0058 | −0.0022 ± 0.0006 | −0.0010 ± 0.0029 |
| hᵀΔθ, R / S | −0.0050 / −0.0054 | −0.0069 / −0.0080 | −0.0185 / −0.0185 |
| S − R, hᵀΔθ | −0.0004 ± 0.0008 | −0.0012 ± 0.0038 | +0.00003 ± 0.0006 |
| \|Δθ\|₂ (R and S alike) | 0.0414 | 0.0605 | 0.0478 |

- **Primary contrast** (S − R in eval_id Pass@1, section 8): no difference detectable in any of the three models; the
  intervals include 0 and the per-block signs differ.
- **hᵀΔθ:** no S − R difference detectable either. One step lowers the reference risk in every block for Qwen3-1.7B and
  Llama-3.2-3B, and in 4 of 5 for Qwen3-4B (b00 is positive for both arms).
- **Qwen3-4B eval_ood (secondary):** S's error is lower than R's in all five blocks, by 0.002–0.003 (2–3 of 1,000
  questions). It is one of several secondary comparisons and not corrected for multiplicity.
- **Size of the step:** Pass@1 moves by under a point against the base; \|Δθ\| is the same for R and S, as the first
  AdamW step's size is fixed (section 1).

Per block and arm: `figures/EXPERIMENT_one_step_table.csv` and `_table15.csv`; figures `_projection`,
`_displacement`, `_gradients`, `_error_SPLIT`, `_SPLIT_pass1`; cost `_cost.csv`.

## When reporting

- **Qwen3-4B:** its round-1 pools were drawn without the loop stop (`generation.repetition_stop`); matching, training
  and evaluation follow the current protocol (section 9).
- **Hardware differs** between the models (table above). On the 8 GB laptop the evaluation halved its batch from 32 to
  8 after running out of memory (`logs/one_step_evaluate.log`); greedy answers do not depend on the batch size.
- **Wall time** (`_cost.csv`, deduplicated total): Qwen3-1.7B 5.3 h (laptop), Qwen3-4B 2.8 h, Llama-3.2-3B 2.0 h (A100).

## What is here

Each folder holds what section 11 asks to send back: `out_one_step/` (matching, per-arm `training.jsonl`,
`diagnostics.json`, `complete.json`, `checks.json`, the per-checkpoint evaluation summaries, cost records),
`one_step_cost_records/`, `one_step_config.json`, `reference_one_step/h.json`, `figures/`, `logs/` and `gpu.txt`.

Not here, as in the 2026-10-03 snapshot: the adapters and every judged answer (`evaluation_greedy_*/*.jsonl`,
34 MB). They are kept on the machines that ran them and are available on request.
