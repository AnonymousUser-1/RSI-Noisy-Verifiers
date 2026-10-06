# arithmetic_unaudited_llama3.2-3b and arithmetic_audited_llama3.2-3b  (2026-10-06 02:50:44, code a742f6f)

```
4657560            p-arithmetic_unaudited_llama3.2-3b  COMPLETED      0:0   06:59:43 
4657561            t-arithmetic_unaudited_llama3.2-3b  COMPLETED      0:0 2-17:55:48 
4657562              t-arithmetic_audited_llama3.2-3b  COMPLETED      0:0 2-16:22:17 
```

## arithmetic_unaudited_llama3.2-3b

- status **complete** (exit 0), K = 64 (16 steps per round), matching {'error_fraction': 0.25, 'token_tolerance': None}, 40 arm-rounds completed, stopped arms: none, attempts 1
- round-1 matching options on these pools:

```
{
 "pools": {
  "b00": {
   "answers": 16384,
   "truncated": 0,
   "correct": 0.74481201171875,
   "target_errors": 74
  },
  "b01": {
   "answers": 16384,
   "truncated": 0,
   "correct": 0.747802734375,
   "target_errors": 81
  },
  "b02": {
   "answers": 16384,
   "truncated": 2,
   "correct": 0.7470703125,
   "target_errors": 71
  },
  "b03": {
   "answers": 16384,
   "truncated": 0,
   "correct": 0.74664306640625,
   "target_errors": 74
  },
  "b04": {
   "answers": 16384,
   "truncated": 2,
   "correct": 0.748291015625,
   "target_errors": 87
  }
 }
}
C:E fraction 0.25  tolerance None -> feasible K=64  J_E per block [30, 32, 17, 28, 33]   <- config.json
C:E fraction 0.25  tolerance 0    -> INFEASIBLE K=None  J_E per block [2, 2, 1, 0, 1]
C:E fraction 0.125 tolerance None -> feasible K=64  J_E per block [30, 32, 17, 28, 33]
C:E fraction 0.125 tolerance 0    -> INFEASIBLE K=None  J_E per block [2, 2, 1, 0, 1]
```

eval_id (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.740 ± 0.000 (n=5) | 0.742 ± 0.008 (n=5) | 0.746 ± 0.017 (n=5) | 0.755 ± 0.008 (n=5) | 0.753 ± 0.020 (n=5) |
| S | 0.740 ± 0.000 (n=5) | 0.745 ± 0.007 (n=5) | 0.742 ± 0.006 (n=5) | 0.742 ± 0.010 (n=5) | 0.739 ± 0.015 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.003 ± 0.006 (n=5) p=0.208 | -0.003 ± 0.021 (n=5) p=0.670 | -0.013 ± 0.012 (n=5) p=0.038 | -0.014 ± 0.019 (n=5) p=0.115 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.004 ± 0.000 (n=5) | 0.005 ± 0.002 (n=5) | 0.006 ± 0.002 (n=5) | 0.007 ± 0.001 (n=5) | 0.007 ± 0.002 (n=5) |
| S | 0.004 ± 0.000 (n=5) | 0.005 ± 0.002 (n=5) | 0.006 ± 0.002 (n=5) | 0.008 ± 0.003 (n=5) | 0.010 ± 0.002 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.000 ± 0.002 (n=5) p=0.794 | 0.000 ± 0.003 (n=5) p=0.842 | 0.002 ± 0.003 (n=5) p=0.119 | 0.003 ± 0.003 (n=5) p=0.050 |

figures: /cluster/rsi-runs/work/experiments/arithmetic_unaudited_llama3.2-3b/figures/arithmetic_unaudited_llama3.2-3b_eval_id_by_difficulty.png, /cluster/rsi-runs/work/experiments/arithmetic_unaudited_llama3.2-3b/figures/arithmetic_unaudited_llama3.2-3b_eval_id_pass1.png

eval_ood (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.243 ± 0.000 (n=5) | 0.264 ± 0.014 (n=5) | 0.261 ± 0.014 (n=5) | 0.269 ± 0.023 (n=5) | 0.262 ± 0.012 (n=5) |
| S | 0.243 ± 0.000 (n=5) | 0.267 ± 0.015 (n=5) | 0.255 ± 0.013 (n=5) | 0.249 ± 0.017 (n=5) | 0.249 ± 0.033 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.004 ± 0.015 (n=5) p=0.538 | -0.006 ± 0.025 (n=5) p=0.549 | -0.020 ± 0.024 (n=5) p=0.090 | -0.013 ± 0.038 (n=5) p=0.405 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.004 ± 0.000 (n=5) | 0.004 ± 0.001 (n=5) | 0.003 ± 0.001 (n=5) | 0.003 ± 0.002 (n=5) | 0.003 ± 0.001 (n=5) |
| S | 0.004 ± 0.000 (n=5) | 0.002 ± 0.001 (n=5) | 0.004 ± 0.001 (n=5) | 0.003 ± 0.001 (n=5) | 0.003 ± 0.002 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.001 ± 0.002 (n=5) p=0.109 | 0.001 ± 0.002 (n=5) p=0.178 | -0.000 ± 0.002 (n=5) p=0.587 | -0.001 ± 0.002 (n=5) p=0.294 |

figures: /cluster/rsi-runs/work/experiments/arithmetic_unaudited_llama3.2-3b/figures/arithmetic_unaudited_llama3.2-3b_eval_ood_by_difficulty.png, /cluster/rsi-runs/work/experiments/arithmetic_unaudited_llama3.2-3b/figures/arithmetic_unaudited_llama3.2-3b_eval_ood_pass1.png

## arithmetic_audited_llama3.2-3b

- status **complete** (exit 0), K = 64 (16 steps per round), matching {'error_fraction': 0.25, 'token_tolerance': None}, 40 arm-rounds completed, stopped arms: none, attempts 1

eval_id (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.740 ± 0.000 (n=5) | 0.744 ± 0.012 (n=5) | 0.751 ± 0.014 (n=5) | 0.756 ± 0.014 (n=5) | 0.754 ± 0.020 (n=5) |
| S | 0.740 ± 0.000 (n=5) | 0.751 ± 0.008 (n=5) | 0.750 ± 0.010 (n=5) | 0.753 ± 0.013 (n=5) | 0.759 ± 0.015 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.007 ± 0.006 (n=5) p=0.031 | -0.001 ± 0.022 (n=5) p=0.857 | -0.003 ± 0.016 (n=5) p=0.582 | 0.006 ± 0.019 (n=5) p=0.429 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.004 ± 0.000 (n=5) | 0.005 ± 0.003 (n=5) | 0.005 ± 0.001 (n=5) | 0.006 ± 0.002 (n=5) | 0.005 ± 0.002 (n=5) |
| S | 0.004 ± 0.000 (n=5) | 0.006 ± 0.001 (n=5) | 0.006 ± 0.001 (n=5) | 0.007 ± 0.002 (n=5) | 0.008 ± 0.003 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.000 ± 0.003 (n=5) p=0.922 | 0.001 ± 0.001 (n=5) p=0.002 | 0.001 ± 0.003 (n=5) p=0.516 | 0.003 ± 0.003 (n=5) p=0.040 |

figures: /cluster/rsi-runs/work/experiments/arithmetic_audited_llama3.2-3b/figures/arithmetic_audited_llama3.2-3b_eval_id_by_difficulty.png, /cluster/rsi-runs/work/experiments/arithmetic_audited_llama3.2-3b/figures/arithmetic_audited_llama3.2-3b_eval_id_pass1.png

eval_ood (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.243 ± 0.000 (n=5) | 0.258 ± 0.020 (n=5) | 0.261 ± 0.017 (n=5) | 0.256 ± 0.028 (n=5) | 0.263 ± 0.032 (n=5) |
| S | 0.243 ± 0.000 (n=5) | 0.266 ± 0.013 (n=5) | 0.274 ± 0.013 (n=5) | 0.280 ± 0.031 (n=5) | 0.291 ± 0.052 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.008 ± 0.010 (n=5) p=0.082 | 0.013 ± 0.021 (n=5) p=0.171 | 0.024 ± 0.051 (n=5) p=0.258 | 0.028 ± 0.072 (n=5) p=0.341 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|
| R | 0.004 ± 0.000 (n=5) | 0.004 ± 0.001 (n=5) | 0.003 ± 0.001 (n=5) | 0.002 ± 0.002 (n=5) | 0.002 ± 0.001 (n=5) |
| S | 0.004 ± 0.000 (n=5) | 0.003 ± 0.001 (n=5) | 0.003 ± 0.003 (n=5) | 0.002 ± 0.002 (n=5) | 0.002 ± 0.002 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.001 ± 0.001 (n=5) p=0.305 | 0.000 ± 0.002 (n=5) p=0.828 | 0.000 ± 0.001 (n=5) p=0.374 | 0.000 ± 0.003 (n=5) p=1.000 |

figures: /cluster/rsi-runs/work/experiments/arithmetic_audited_llama3.2-3b/figures/arithmetic_audited_llama3.2-3b_eval_ood_by_difficulty.png, /cluster/rsi-runs/work/experiments/arithmetic_audited_llama3.2-3b/figures/arithmetic_audited_llama3.2-3b_eval_ood_pass1.png

## Comparison figures

- /cluster/rsi-runs/work/experiments/figures/arithmetic_unaudited_llama3.2-3b_vs_arithmetic_audited_llama3.2-3b_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/arithmetic_unaudited_llama3.2-3b_vs_arithmetic_audited_llama3.2-3b_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/arithmetic_unaudited_llama3.2-3b_vs_arithmetic_audited_llama3.2-3b_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/arithmetic_unaudited_llama3.2-3b_vs_arithmetic_audited_llama3.2-3b_eval_ood_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_ood_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_rounds0-4_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_rounds0-4_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_rounds0-4_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_rounds0-4_eval_ood_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_ood_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_rounds0-4_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_rounds0-4_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_rounds0-4_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_rounds0-4_eval_ood_pass1.png
