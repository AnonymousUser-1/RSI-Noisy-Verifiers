# Qwen3-4B graph, multi-round R/S: unaudited condition only (2026-10-05)

**This snapshot holds the unaudited condition only.** The audited condition was not run for Qwen3-4B.

The unaudited multi-round R/S experiment on Qwen3-4B, graph shortest paths, five seed blocks (b00–b04), four
rounds. Training and evaluation are complete: 40 / 40 arm-rounds trained, 41 / 41 checkpoints evaluated on
`eval_id` (2,000) and `eval_ood` (1,000), greedy.

- **Code:** `main` at `f1fd1f8` (#28), the code of the H100 runs (`a742f6f`).
- **Settings:** those of [`experiments/graph_unaudited/`](../../graph_unaudited/) with the base model
  `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`: temperature 1.3, top-p 1.0, top-k off,
  `max_new_tokens` 2,048, LoRA r 8 / α 16 on q/v, lr 2e-4, effective batch 4, K = 64 at C:E 48:16,
  16 steps per round. Every round samples 2,048 prompts × 8 answers.
- **Round 1:** the Qwen3-4B round-1 pools, shared adapter and matching of `RSI-Qwen3-4B-Graph-paired-inputs-20261004/`,
  the same inputs as the Qwen3-4B one-step comparison ([`../2026-10-04-one-step/`](../2026-10-04-one-step/)).
- **Loop stop:** the round-1 pools were drawn without the loop stop (`generation.repetition_stop`), as in
  the one-step comparison ([`ONE_STEP_EXPERIMENT_INSTRUCTION.md`](../../../ONE_STEP_EXPERIMENT_INSTRUCTION.md),
  section 9). Later-round sampling used it from block b01 round 4 on; block b00 and rounds 1–3 of b01 were
  sampled without it. Both arms of a block share the setting in every round. Every unit records its setting
  (`run.json` `training_profiles`, `protocol-receipt.json`; boundary in `out/protocol_profile.json`), and all
  checkpoints were evaluated with it. No training set of any unit contains a looped answer.
- **h:** graph is not affected by the arithmetic reference-answer correction (draft #32); h is used as recorded.
  The h of the two sampling settings is tensor-equal (`reference/equivalence.json`).

## Results

Greedy Pass@1, mean over the five blocks; S − R paired by block, mean ± 95% t half-width (n = 5), with the
paired t-test p-value. From [`figures/`](figures/) (`plot_multiround.py`).

| round | eval_id R | eval_id S | eval_id S − R | eval_ood R | eval_ood S | eval_ood S − R |
|---|---|---|---|---|---|---|
| 0 (base) | 0.6125 | 0.6125 | — | 0.3970 | 0.3970 | — |
| 1 | 0.6141 | 0.6074 | −0.0067 ± 0.0280 (p 0.54) | 0.3938 | 0.3922 | −0.0016 ± 0.0216 (p 0.85) |
| 2 | 0.6342 | 0.6504 | +0.0162 ± 0.0344 (p 0.26) | 0.3978 | 0.4286 | **+0.0308 ± 0.0243 (p 0.024)** |
| 3 | 0.6803 | 0.7031 | +0.0228 ± 0.0388 (p 0.18) | 0.4322 | 0.4710 | **+0.0388 ± 0.0233 (p 0.010)** |
| 4 | 0.7309 | 0.7339 | +0.0030 ± 0.0968 (p 0.94) | 0.4988 | 0.4982 | −0.0006 ± 0.0991 (p 0.99) |

- Both arms improve: eval_id 0.61 → 0.73, eval_ood 0.40 → 0.50 at round 4.
- On eval_ood, S is ahead of R in rounds 2 and 3 in every block. At round 4 the blocks disagree
  (S − R from −0.10 to +0.12) and the mean is zero.
- The intervals are descriptive and not adjusted for multiple comparisons.

## Sampling at temperature 1.3 in later rounds

The sampled pools (temperature 1.3) degrade from round 3 on, although greedy Pass@1 keeps improving:

- Pool accuracy falls from 63% (base) to 44–63% at round 4.
- Some answers continue past the path into unrelated multilingual text and code fragments. These are not
  loops, so `repetition_stop` does not end them; some reach the 2,048-token cap (round 4: 7–262 per pool).
  This makes round-4 sampling several times slower than earlier rounds.
- Such an answer is a `format` error. R draws its errors from all error types, so it can select one: 9 of R's
  320 round-4 training examples contain such text (b01: 1, b02: 2, b03: 4, b04: 2). S takes only
  `nonshortest` and never selects one.
- At round 4, R's pools hold more `format` errors than S's in four of five blocks (R 2.5–23.8%, S 1.7–4.3%).

## What is here

As in the H100 snapshot: per arm `run.json`, `finished.json` and per round `complete.json` and
`selection.json`, plus each round's `protocol-receipt.json` (its sampling setting); `out/experiment.json`,
`out/matching/` and `out/protocol_profile.json`; the round-1 pool metas; the shared adapter's record;
`reference/h.json`, `per_sample.jsonl` and `equivalence.json`; the evaluation summaries
(`out/evaluation_greedy_SPLIT/*.json`, with `protocol.json`); `figures/`; `logs/` without progress lines.
Paths in the records are relative to the run; `round1/` is the run that drew the round-1 pools.

`out/protocol_profile.json` is the run's `migration.json` with machine paths reduced. `experiment.json` and
each `run.json` record both identities under `protocol_migration`: `original` (the run's file, sha256
`1cd475eb…`) and `published` (this file, sha256 `aa21e26c…`).

Not here: the adapters, the later-round pools, the training sets, `h.pt` and the judged answers
(`evaluation_greedy_*/*.jsonl`). They are not part of this release.
