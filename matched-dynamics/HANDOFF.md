# Matched-dynamics study: who does what

This document preserves the original A/B/C division of work, not live execution status.
The original handoff used a 2026-10-06 internal target and serial single-GPU execution.
Current runnable experiments and dated results are linked from [experiments/README.md](../experiments/README.md).
The L=1 correction and numerical verification are recorded in [THEORY_STATUS.md](THEORY_STATUS.md).

## Frozen inputs

- Design and exact scope: `PLAN.md` in this directory (numeric contract, matching constraints,
  failure semantics, readout).
- Original defect dispositions and remaining scientific limits: `KNOWN_GAPS.md`.
- Original model pin: `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
  Model-specific experiment configurations and actual GPU results now have their own records.
- Candidate sizing (`K`, steps, hours) is a **pilot** open question, not a frozen power promise.

## Role A - data and theory

Owns: `generate_data.py`, `rsi/logistic.py`, `tests/test_logistic_matched.py`.

Deliverables:
1. A `gradient_reference.jsonl` split (64 prompts), mutually exclusive with `train_*`, `dev`,
   `calibration`, `eval_id`, `eval_ood`, plus its manifest hash. Diagnostic use only - it must never
   influence pool construction, matching, or tuning.
2. Numeric check of the analytic core at the frozen parameters, plus the two ablations that keep
   the construction honest: `L = 1` (no separation) and `L = 2` (`g_R = -0.15`, `g_S = -0.05`).
   The general S numerator and the L=1 named-group fixture are corrected. Reproduction commands
   and completed checks are in `THEORY_STATUS.md`; these are not remaining implementation tasks.
3. The per-block candidate pools, hash-bound, before any matching or training.

Hands off to: B (pool files plus hashes), C (split manifest).

## Role B - model and controlled update

Owns: `rsi/matching.py`, `rsi/backends.py`, `rsi/experiment.py`, `run_job.py`,
`sample_candidates.py`, `configs/`.

This is the critical path. B must finish **all** pools and the joint matching, with a written
feasibility or infeasibility certificate, before any formal block is trained. Deliverables:
1. `shared_adapter/` - one common LoRA initialisation (rank 8 on q/v, recorded in config), with
   parameter hash, seed, and base revision.
2. `matched_subsets.json` - the R and S final subsets per block, with the hard-constraint audit, the
   `N+ / N-` denominators, `final_TPR / final_FPR`, and the ladder attempt record.
3. `diagnostics/` - streaming per-sample norms, correct/error gradient sums, clipping state,
   `Delta theta`, and `h^T Delta theta`. Do not materialise an `N x P` gradient matrix.
4. The one-step main entry: all `K` examples accumulated, exactly one `optimizer.step()`, dropout 0,
   uniform weights, auditing off, identical optimizer state, learning rate, prompt order, and clip
   threshold across branches.

Hands off to: C (subsets, diagnostics, adapter references).

## Role C - independent evaluation and analysis

Owns: `evaluate.py`, `analyze.py`, `tests/test_matching.py`.

Deliverables:
1. Zero-step baseline risk and post-training independent risk on `eval_id` / `eval_ood` under a
   frozen decoding configuration and shared sampling seeds, with repeated answers grouped by prompt.
2. Paired intervals per block; zero-effect and no-difference results retained.
3. Final-training-subset statistics recomputed **with the common original pool as the denominator**
   (replacing the current behaviour that reports stage statistics from the retained set), keeping the
   selection / provisional / retained stages all visible.

Hands off to: the write-up. C is also the owner of the analysis freeze: the evaluation code path,
decode settings, and seed are fixed before the confirmatory blocks are trained.

## Order of work

1. Contract and open checks frozen (done).
2. A: splits and pools. C: evaluation and analysis scripts ready in parallel.
3. B: shared adapter, joint matching, feasibility certificate. **Stop here if infeasible at K = 16.**
4. B: one-step updates for the feasible `K`; R, S, and the un-updated null.
5. C: independent risk, paired readout.
6. Optional, only after the main comparison: 2-4 fixed-pool steps, and the audited equal-query
   variant.

## What must not happen

- No training of a formal block before every pool exists and matching is certified.
- No seed change, no truncation to force a length match, no relaxation of the token-count tolerance.
- No substitution of mock or legacy numbers for a real result.
- No claim that the language-model study ran until a real GPU run is recorded with its cost.
