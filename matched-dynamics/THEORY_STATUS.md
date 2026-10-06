# Theory implementation: L=1 correction and numerical checks

Updated 2026-10-04. This page describes the correction included in this branch and its bounded
verification. It does not imply that an unmerged branch has changed GitHub main.

## What is resolved

The deterministic 128-candidate fixture previously inferred group membership from `abs(x) == L`.
At L=1 this incorrectly classified every group as large. The corrected implementation keeps the
named groups distinct: both R and S retain 48 correct + 32 incorrect examples, TPR .75 and FPR .5.
Their empirical initial BCE gradients are both -0.1. At L=2 the gradients are -0.15/-0.05, so both
initial updates improve sampling accuracy. The L=10 construction still gives -0.55/+0.35.

| Component | Repository source | Verification |
|---|---|---|
| Corrected fixture and root-finding scope | [rsi/logistic.py](../rsi/logistic.py) | Explicit groups; a bracketed root is not a proof of uniqueness |
| Regression tests | [test_logistic_matched.py](../tests/test_logistic_matched.py) | 13 tests, including membership, counts/rates and exact finite-sample gradient checks |
| Numerical companion | [theory_checks.py](../theory_checks.py) | Nine groups covering the logistic core, initial values, covariance, auditing, bounds, budget, finite pools, smooth loss and flow |
| Branch verification | [verification record](verification/l1-regression-20261004.json) | Source hashes, current results and a baseline failure control |

The implementation and test changes are identical to the local fix
`628b10301fa1b5e0295eb1d2fb05c4845c3aafc9`, whose engineering check recorded Pass on 2026-10-03.
They are ported onto main `7295a6bdc8b1869739cff909577421d01a56a926` here. The verification record
reports fresh checks against the files in this branch; it does not reuse the old pass as evidence
for a different source version.

## Reproduce from the repository root

Python with NumPy is sufficient; no model weights or GPU are used.

```bash
python -B -m unittest discover -s tests -p test_logistic_matched.py -v
python -B theory_checks.py --code-dir .
```

The regressions pass all 13 tests without skips. The numerical companion passes all nine groups,
with 13,714 check operations; these are not 13,714 independent experiments. Applying the corrected
tests to the pre-fix implementation produces the three expected failures, so the tests detect the
original defect. Source-bound details are in the linked verification record.

## Interpretation and remaining publication work

- The theory fixture's 48+32 construction is distinct from the LLM K64=48+16 selection protocol.
  This correction does not change generation, matching, training, evaluation or running GPU jobs.
- Numerical checks do not prove the universal theorems or certify an LLM smoothness constant.
- The companion script's manuscript hash and table labels identify the supplied 21-page draft
  for which it was written. The newer 42-page draft renumbers figures/tables. Publishing its exact
  TeX and figure sources and updating that correspondence is a separate manuscript task; this
  branch does not claim validation of the entire newer draft.
- The local theory/figure deliveries and the current manuscript PDF are separate artifacts.
  They are not represented as newly uploaded by this source-and-regression synchronization.
- Real LLM evidence has its own dated [experiment snapshot](../experiments/outputs/2026-10-03-h100/README.md).
  Its complete and incomplete runs must not be inferred from historical handoff text.
