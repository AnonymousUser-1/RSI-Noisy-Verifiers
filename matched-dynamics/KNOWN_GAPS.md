# Implementation gaps and their disposition

Updated 2026-10-04. Section A records the disposition of the original theory defects;
[THEORY_STATUS.md](THEORY_STATUS.md) gives the current checks and evidence. Sections B and C
retain historical implementation/readiness gaps and are not current claims that no training
code or GPU results exist. See the [experiment entry](../experiments/README.md) and its dated
outputs for actual execution evidence. Scientific limitations remain separate from implementation status.

## A. Original theory implementation defects — resolved

These come from the implementation check record kept with the design. The analytic core values
themselves were recomputed for this version and are correct; the defects are around it.

1. **`finite_pool_trajectory` in `rsi/logistic.py` was not part of the frozen contract -- removed.**
   It drew correctness as `sigma(x theta)` directly (treating every negative `x` as incorrect),
   accepted all correct candidates without applying `a`, and then advanced `theta` using the
   **population** gradient rather than the empirical gradient of the sampled accepted subset. That
   is a random simulation of something other than the agreed model, and it invited the wrong
   reading. It has been deleted in this version, along with its two helpers `_sigma_np` and
   `sigma_np`; the correct population flow and the deterministic 128-candidate construction already
   cover what the contract needs.

2. **L=1 finite-fixture group identity — fixed.** The old `abs(x) == L` test treated all four
   named groups as large at L=1, retaining 48 correct + 64 incorrect examples for S, with FPR
   1.0 and empirical gradient +1/14. The fixture now uses its explicit group identity. Both
   rules retain 48 correct + 32 incorrect examples, with purity .6, TPR .75, FPR .5, and
   empirical initial gradient -1/10. Three regressions check group membership, actual counts
   and rates, and exact retained-example BCE gradients at L=1,2,3,10. The results and commands
   are in [THEORY_STATUS.md](THEORY_STATUS.md). The LLM K64=48+16 protocol is unchanged.

3. **`s_gradient_numerator(theta, L=...)` hard-coded the coefficients of the `L = 10`, `theta <= 0`
   branch as if they were general -- fixed.** The original body returned
   `2.5 q (.5 - p + .5 q) - .375 p (1 - p)`. The claim previously recorded here (that the
   coefficients "belong to `L = 10`") was itself wrong on two counts. First, `2.5` is not the
   `L = 10` general coefficient: at `L = 10` and `theta = 0.31` the old form returns `0.869094`
   where the true numerator is `0.014833`, and at `theta = 1.0` it gives `0.598537` against
   `-0.024528` -- a sign error on the frozen contract path itself. Second, the `L = 2` value quoted
   here (`0.21875` instead of `-0.03125`) was right as a measurement but the diagnosis was not; the
   divergence comes from the `r_L`/`r_1` branch structure, not from a dimensionality constant. For
   `theta >= 0` the budget `Q = b(m_1 + m_L)` saturates the large-magnitude group (`r_L = 1`) and
   the remainder goes to the small group with an extra `(1-q)/(1-p)` factor, so no single
   `max(0, q-p)` correction can be right. A second "general closed form"
   `(L q^2 + L q + 3 p^2 - 3 p - 2 L p q)/8 + p max(0, q-p)/4` was proposed and is **also wrong**,
   for the same branch reason; it is not installed. The function now computes the numerator from
   `acceptance_rates`, verified against `stopped_gradient * acceptance_rate` over `theta` in
   `[-4, 4]` step `.01` and `L` in `{1, 2, 4, 10, 25, 100}` (max deviation `1.1e-16`), and it is
   regression-tested.

These three are exactly the local-check problems handed over with the checkpoint; the disposition
(remove, or fix with a guarded domain) is an implementation decision recorded in `DECISION.md`.
Note that the general population formulas in the same module are **not** affected: `stopped_gradient`
reproduces `g_R(0) = -0.55`, `g_S(0) = +0.35`, `theta* = -0.11146`, sampling accuracy `.35959`,
`L = 1` -> both `-0.1`, and `L = 2` -> `-0.15` / `-0.05`.

## B. Original language-model implementation gaps (historical)

The first review recorded the following missing capabilities. Subsequent work added the
matched and iterative runners, shared adapters, reference gradients, and evaluation described
in [experiments/README.md](../experiments/README.md). The list below is historical, not a
current acceptance checklist:

- `run_job.py` does not forward the shared candidate pool that `run_experiment.py` already accepts;
- there is no shared LoRA initialisation exposed as a reusable artefact, so two branches cannot be
  shown to start from the same parameters;
- there is no per-sample gradient, clipping-state, or `Delta theta` recorder;
- `analyze.py` still computes stage TPR/FPR with the retained set as the denominator, which is the
  statistic the design explicitly rejects;
- the one-step main entry, the joint matching CLI path, and the ladder/infeasibility reporting are
  not wired into any command.

No new command should be described as available until it exists in code, has a working `--help`,
and has a CPU end-to-end record.

## C. Original GPU readiness gap (historical)

The bullets below describe the first-review environment. They are superseded by real GPU
records, including the completed Llama-3.2-3B graph pair in the
[2026-10-03 H100 snapshot](../experiments/outputs/2026-10-03-h100/README.md).

- CPU environment only (`torch 2.14.0+cpu`, no CUDA). Generation, training, and evaluation
  throughput and memory are **unmeasured**.
- Specific GPU model and GPU-hour budget are **unconfirmed**. The old four-A100 / ten-day ceiling
  belongs to the legacy plan and is not a budget for this study.
- The pinned model revision is public metadata only; no weights were downloaded here.
- Decoding settings and chat-template handling are unverified against the real tokenizer.

## D. Scientific limits that stay limits

- Three seed blocks give three paired comparisons; a two-sided sign test on three same-signed
  differences has minimum `p = .25`. No power claim and no regression on six points.
- `h^T Delta theta` agreeing with the reference NLL change is first a Taylor self-consistency check.
  It is not validation that this quantity predicts error rate on new tasks.
- The logistic separation is a construction. It depends on the influence imbalance: at `L = 1` it
  vanishes. Whether an analogous property is measurable in a 1.7B language model is the open
  empirical question, not a settled result.
- Regression and numerical checks do not constitute an independent review of the universal
  mathematical proofs. The historical review of code-to-manuscript fit is not such a review either.
- Manuscript integration and acceptance must be checked against the actual manuscript version;
  the initial draft's empty experiment chapter is not a current project-status claim.

## E. Auditing is a separate experimental intervention

The primary matched one-step comparison is unaudited. Budgeted correctness auditing is a separate
implemented intervention; its current protocol and boundaries are in [AUDITING.md](AUDITING.md).
This theory fixture correction does not change either experimental protocol.
