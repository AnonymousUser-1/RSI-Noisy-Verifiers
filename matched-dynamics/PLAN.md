# Matched Verifier Error Rates Do Not Determine Self-Training Dynamics

Status: **research plan and implementation contract, not results.** Prepared 2026-09-28.
Separate from `../EXPERIMENT_PLAN.md`, which describes the earlier eight-round protocol and is
retained as `legacy` (see the banner in that file). Nothing in this directory reports a completed
experiment, a completed language-model pipeline, or a submission.

## 1. Claim under study

Iterative self-training uses a verifier to select model-generated answers for later parameter
updates. Verifier quality is usually summarised by true- and false-acceptance rates (TPR/FPR).
Those two numbers do not describe **how the accepted examples influence learning**.

In a controlled logistic model, two verifiers applied to the same initial learner have identical
operating characteristics, acceptance rate, and accepted-data precision, yet induce **opposite**
update directions. Under population gradient flow their TPR and FPR stay matched while one learner
improves and the other converges **below** its initial accuracy. The separation comes from
concentrating accepted errors on examples with larger influence on the shared parameter.

The planned small-model study asks whether measurable properties of the accepted errors carry
predictive value **beyond** accepted-pool precision, training volume, task coverage, and response
length. Correctness-only auditing is a secondary intervention under an equal query budget.

Scope limit: this is a mechanism demonstration. Three seed blocks give three paired comparisons;
they cannot support a multi-covariate regression or a general claim that these diagnostics predict
harm in arbitrary self-training runs.

## 2. Logistic core (analytic, frozen)

Group magnitude `s in {1, L}` with probability `rho_1 = 1 - b`, `rho_L = b`; sign independent and
equiprobable; true label `y* = 1{x > 0}`. The learner emits `y_hat ~ Bernoulli(sigma(theta x))`
with `p_s = sigma(s theta)`, loss is BCE of the candidate label, candidates and acceptance rule are
constants under differentiation (stop-gradient). Both rules accept a correct candidate with
probability `a`; `R` accepts errors with probability `b`; `S` prefers large-magnitude errors and
tops up to the same total error mass `b`, so the two rules match TPR/FPR by construction while
acceptance rate and purity are equal only at the shared initialisation.

Frozen readouts at `a = .75, b = .5, L = 10, theta_0 = 0`:

| quantity | value |
| --- | --- |
| TPR, FPR, acceptance rate `Z`, purity | .75, .5, .625, .6 |
| `g_R(0)` (random rule) | -0.55 |
| `g_S(0)` (structured rule) | +0.35 |
| one Euler step, `eta` | `dtheta_R = +0.55 eta`, `dtheta_S = -0.35 eta` |
| equilibrium parameter of the S flow, `theta*` | -0.1114604402 |
| **sampling** accuracy at `theta*` | .3595885926 |

`theta*` is the equilibrium of the **population flow** of `S` over many steps, not a one-step
update, and `P(theta*)` is a sampling accuracy, not greedy classification accuracy. After the
branches separate their acceptance rates and purities differ: only TPR/FPR remain matched.

Ablations that keep the design honest: at `L = 1` the named groups remain distinct and both
initial gradients are -0.1. At `L = 2`, the gradients are -0.15 for R and -0.05 for S, so both
initial updates increase sampling accuracy; there is no opposite-direction separation.
The 128-candidate construction in `rsi/logistic.py` is a hand-built fixture that reproduces
these initial numbers exactly. A general random finite pool with matching counts is not
guaranteed to separate the rules. See [the theory checks and their scope](THEORY_STATUS.md).

## 3. Matched-subset construction (the main intervention)

The earlier pipeline filtered the candidate pool with per-stratum quotas, then kept one answer per
prompt, then audited, then capped the training set. Those are separate sampling steps over the same
pool, so two branches could agree on **every candidate-level acceptance statistic** and still hand
the trainer different numbers of examples, different prompts, different error counts, and different
supervised token counts. Under those conditions a difference in outcome is confounded with volume,
coverage, and length.

The main path therefore builds both branches **jointly, from the common original pool**:

1. All blocks' pools are generated and hash-bound first. No training starts before that.
2. One common checkpoint, one shared LoRA initialisation, one candidate pool per block.
3. `R` draws error responses uniformly inside a `(prompt, length bucket)`; `S` prefers the
   `nonshortest` signature inside the **same** bucket.  (With a nonzero `matching.token_tolerance`
   the bucket is every error within the tolerance of S's, and S draws uniformly among the target
   errors that have a non-target error within it; DECISION.md, 2026-10-02.) If `R` happens to draw the same candidate as
   `S` (or the same signature class), that coincidence is **kept, never redrawn**.
4. Hard constraints asserted before any update: same prompt set; same correct-response ids; per
   prompt identical supervised response-token count including the real EOS and excluding prompt and
   padding, **tolerance 0**; `C:E = 3:1` with common `K = C + E`; all weights 1.0; exactly one
   optimizer update per branch.  (Tolerance 0 and 3:1 are the defaults of the config fields
   `matching.token_tolerance` and `matching.error_fraction`; an experiment that sets other values
   says so in its config and in DECISION.md, 2026-10-02.)
5. Task role assignment is a set problem, not greedy add/drop: assign correct roles from
   `J_C \ J_E` first, then from the intersection, then fill error roles from the remaining `J_E`.
   The documented counterexample (task A has a correct answer and both a target and a
   non-target error at one length; task B has only a correct answer; `C = E = 1`) resolves as
   `B -> correct`, `A -> error`. Pairwise "drop one error from each branch" is unsound because it
   preserves the difference `E_R - E_S`. In the solver, tasks with one legal role (correct-only or
   error-only) go before flexible ones, so error roles come from `J_E \ J_C` first. Equally
   constrained tasks are taken in a random order drawn from the seed and the block's own pool,
   not in task-id order, so seed blocks draw their prompts independently and overlap only by
   chance (DECISION.md, 2026-10-02).
6. Sizing: `K` is the largest value in the declared ladder `64 -> 32 -> 16` that is feasible for
   **all** blocks simultaneously. If `K = 16` is infeasible for any block, the main contract stops
   and the gap is recorded. There is no seed change, no truncation to force a match, and no
   relaxation of the length tolerance.
7. Diagnostics use the shared original pool as the denominator, so
   `final_TPR = C / N+`, `final_FPR = E / N-`, `yield = K / (N+ + N-)`, `precision = C / K`. These
   are identical across branches within a block by construction, but they are **not** required to
   equal the legacy quota target, which is reported separately as `quota_target`.

## 4. Measurements

Recorded per run, in a single fixed LoRA coordinate system on a frozen base model:

- per-sample supervised response-token loss (existing per-response mean normalisation, retained);
- streaming per-sample gradient norms, plus the correct-error split sums `G_C`, `G_E`, `G`;
- gradient before clipping, gradient actually clipped, and whether clipping fired;
- the actual parameter change `Delta theta = theta_after - theta_before`, and its projection
  `h^T Delta theta` onto an independent reference gradient `h = grad R_reference`;
- independent task error rate on held-out prompts under a frozen decoding setup, reported
  separately from the reference NLL proxy.

The main update diagnostic is `h^T Delta theta`; the raw gradient is reported for mechanism
interpretation only. AdamW with zero weight decay does not make `|Delta theta|` proportional to
`|G|`, and the update does not decompose linearly into correct and error contributions. Response
length is a covariate reported beside the norms, never a synonym for influence.

CPU numerical agreement target: `rtol = 1e-4`, `atol = 1e-6` for the sum-of-sample-gradients
identity. GPU acceptance target: relative L2 error below 1% for gradient composition, with the
absolute error reported; exceeding it is fixed or run at higher precision, never waived.
Snapshots are compared in at least FP32, and a projection that is within numerical noise does not
get a sign claim.

## 5. Data isolation

`train_*`, `dev`, `calibration`, `gradient_reference`, `eval_id`, and `eval_ood` are mutually
exclusive; instance ids carry no split name, and `generate_data.py` enforces a cross-split
content-hash `seen` set. The gradient reference set is used for diagnostics only and never for
selection or tuning. Final-test prompts take no part in pool construction, matching, diagnostics,
or tuning, and are decoded only after training completes. Test metrics never feed back.

## 6. Readout

Three seed blocks, two rules, three paired comparisons. Report per-block change in true task risk,
the actual update, and intervals conditioned on this small model set. Keep null and
no-difference results in the record. The frozen null is a common-initialisation un-updated model
plus a repeated-measurement baseline. One step is the primary comparison; 2-4 fixed-pool updates
are a pre-registered optional sensitivity check and are **not** re-sampled multi-round
self-training. Auditing is a single optional equal-query-budget protocol with its own query count.

## 7. Status of the code in this repository

`rsi/logistic.py` and `rsi/matching.py` are **new, unaudited candidates** for the design above.
They implement the analytic core and the joint construction; they are not accepted, not reviewed,
and known defects are listed in `KNOWN_GAPS.md`. The full HF/PEFT generation-training-evaluation
path for this new main line is **not implemented**: `run_job.py` still does not forward a shared
candidate pool, there is no shared-adapter initialisation, there is no per-sample gradient or
`Delta theta` recorder, and `analyze.py` still reports the final-training-subset statistics with
the retained set as the denominator. No GPU run of any kind has been performed for this line, and
no study weights have been downloaded on this machine. CPU unit tests passing is baseline
evidence, not evidence that the new pipeline works.
