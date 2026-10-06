# graph_unaudited and graph_audited  (2026-10-04 17:13:37, code a742f6f)

```
4657546                             p-graph_unaudited  COMPLETED      0:0   01:24:24 
4657547                             t-graph_unaudited  COMPLETED      0:0   15:42:31 
4671293                             x-graph_unaudited  COMPLETED      0:0   22:49:46 
4657548                               t-graph_audited  COMPLETED      0:0   14:34:36 
4671294                               x-graph_audited  COMPLETED      0:0   21:11:04 
```

## graph_unaudited

- status **complete** (exit 0), K = 64 (16 steps per round), matching {'error_fraction': 0.25, 'token_tolerance': 0}, 80 arm-rounds completed, stopped arms: none, attempts 2
- round-1 matching options on these pools:

```
{
 "pools": {
  "b00": {
   "answers": 16384,
   "truncated": 27,
   "correct": 0.50311279296875,
   "target_errors": 2357
  },
  "b01": {
   "answers": 16384,
   "truncated": 29,
   "correct": 0.5059814453125,
   "target_errors": 2351
  },
  "b02": {
   "answers": 16384,
   "truncated": 33,
   "correct": 0.5042724609375,
   "target_errors": 2322
  },
  "b03": {
   "answers": 16384,
   "truncated": 30,
   "correct": 0.50830078125,
   "target_errors": 2354
  },
  "b04": {
   "answers": 16384,
   "truncated": 29,
   "correct": 0.5050048828125,
   "target_errors": 2346
  }
 }
}
C:E fraction 0.25  tolerance None -> feasible K=64  J_E per block [142, 138, 127, 137, 125]
C:E fraction 0.25  tolerance 0    -> feasible K=64  J_E per block [27, 38, 32, 31, 28]   <- config.json
C:E fraction 0.125 tolerance None -> feasible K=64  J_E per block [142, 138, 127, 137, 125]
C:E fraction 0.125 tolerance 0    -> feasible K=64  J_E per block [27, 38, 32, 31, 28]
```

eval_id (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.499 ± 0.000 (n=5) | 0.517 ± 0.017 (n=5) | 0.529 ± 0.029 (n=5) | 0.536 ± 0.040 (n=5) | 0.556 ± 0.043 (n=5) | 0.564 ± 0.030 (n=5) | 0.580 ± 0.030 (n=5) | 0.597 ± 0.032 (n=5) | 0.606 ± 0.038 (n=5) |
| S | 0.499 ± 0.000 (n=5) | 0.527 ± 0.019 (n=5) | 0.532 ± 0.037 (n=5) | 0.541 ± 0.055 (n=5) | 0.553 ± 0.040 (n=5) | 0.577 ± 0.064 (n=5) | 0.573 ± 0.079 (n=5) | 0.632 ± 0.049 (n=5) | 0.646 ± 0.030 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.010 ± 0.012 (n=5) p=0.077 | 0.003 ± 0.035 (n=5) p=0.824 | 0.005 ± 0.039 (n=5) p=0.761 | -0.003 ± 0.044 (n=5) p=0.878 | 0.013 ± 0.043 (n=5) p=0.460 | -0.007 ± 0.083 (n=5) p=0.833 | 0.034 ± 0.052 (n=5) p=0.139 | 0.040 ± 0.045 (n=5) p=0.071 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.135 ± 0.000 (n=5) | 0.132 ± 0.029 (n=5) | 0.126 ± 0.045 (n=5) | 0.121 ± 0.031 (n=5) | 0.103 ± 0.029 (n=5) | 0.105 ± 0.059 (n=5) | 0.077 ± 0.049 (n=5) | 0.067 ± 0.046 (n=5) | 0.049 ± 0.036 (n=5) |
| S | 0.135 ± 0.000 (n=5) | 0.129 ± 0.024 (n=5) | 0.143 ± 0.038 (n=5) | 0.168 ± 0.032 (n=5) | 0.171 ± 0.037 (n=5) | 0.148 ± 0.047 (n=5) | 0.162 ± 0.061 (n=5) | 0.132 ± 0.027 (n=5) | 0.151 ± 0.038 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.003 ± 0.012 (n=5) p=0.587 | 0.017 ± 0.043 (n=5) p=0.325 | 0.046 ± 0.022 (n=5) p=0.004 | 0.068 ± 0.039 (n=5) p=0.008 | 0.043 ± 0.075 (n=5) p=0.185 | 0.086 ± 0.081 (n=5) p=0.042 | 0.065 ± 0.057 (n=5) p=0.034 | 0.102 ± 0.033 (n=5) p=0.001 |

figures: /cluster/rsi-runs/work/experiments/graph_unaudited/figures/graph_unaudited_eval_id_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_unaudited/figures/graph_unaudited_eval_id_pass1.png

eval_ood (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.347 ± 0.000 (n=5) | 0.352 ± 0.011 (n=5) | 0.349 ± 0.026 (n=5) | 0.362 ± 0.022 (n=5) | 0.363 ± 0.026 (n=5) | 0.362 ± 0.016 (n=5) | 0.363 ± 0.032 (n=5) | 0.377 ± 0.019 (n=5) | 0.385 ± 0.039 (n=5) |
| S | 0.347 ± 0.000 (n=5) | 0.366 ± 0.015 (n=5) | 0.369 ± 0.016 (n=5) | 0.385 ± 0.034 (n=5) | 0.389 ± 0.025 (n=5) | 0.391 ± 0.032 (n=5) | 0.386 ± 0.051 (n=5) | 0.422 ± 0.042 (n=5) | 0.432 ± 0.015 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.014 ± 0.021 (n=5) p=0.142 | 0.020 ± 0.020 (n=5) p=0.049 | 0.023 ± 0.022 (n=5) p=0.044 | 0.026 ± 0.038 (n=5) p=0.131 | 0.029 ± 0.030 (n=5) p=0.054 | 0.023 ± 0.063 (n=5) p=0.366 | 0.045 ± 0.045 (n=5) p=0.050 | 0.047 ± 0.042 (n=5) p=0.036 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.060 ± 0.000 (n=5) | 0.075 ± 0.025 (n=5) | 0.071 ± 0.032 (n=5) | 0.068 ± 0.016 (n=5) | 0.057 ± 0.017 (n=5) | 0.073 ± 0.040 (n=5) | 0.057 ± 0.034 (n=5) | 0.050 ± 0.021 (n=5) | 0.042 ± 0.035 (n=5) |
| S | 0.060 ± 0.000 (n=5) | 0.065 ± 0.021 (n=5) | 0.088 ± 0.029 (n=5) | 0.108 ± 0.025 (n=5) | 0.124 ± 0.032 (n=5) | 0.109 ± 0.025 (n=5) | 0.124 ± 0.030 (n=5) | 0.114 ± 0.034 (n=5) | 0.145 ± 0.060 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.009 ± 0.007 (n=5) p=0.024 | 0.017 ± 0.026 (n=5) p=0.138 | 0.040 ± 0.015 (n=5) p=0.002 | 0.066 ± 0.038 (n=5) p=0.009 | 0.036 ± 0.052 (n=5) p=0.128 | 0.067 ± 0.050 (n=5) p=0.021 | 0.064 ± 0.034 (n=5) p=0.007 | 0.103 ± 0.030 (n=5) p=0.001 |

figures: /cluster/rsi-runs/work/experiments/graph_unaudited/figures/graph_unaudited_eval_ood_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_unaudited/figures/graph_unaudited_eval_ood_pass1.png

## graph_audited

- status **complete** (exit 0), K = 64 (16 steps per round), matching {'error_fraction': 0.25, 'token_tolerance': 0}, 80 arm-rounds completed, stopped arms: none, attempts 2

eval_id (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.499 ± 0.000 (n=5) | 0.513 ± 0.027 (n=5) | 0.530 ± 0.021 (n=5) | 0.526 ± 0.039 (n=5) | 0.533 ± 0.051 (n=5) | 0.551 ± 0.045 (n=5) | 0.548 ± 0.037 (n=5) | 0.541 ± 0.054 (n=5) | 0.589 ± 0.038 (n=5) |
| S | 0.499 ± 0.000 (n=5) | 0.509 ± 0.024 (n=5) | 0.532 ± 0.012 (n=5) | 0.561 ± 0.045 (n=5) | 0.551 ± 0.062 (n=5) | 0.569 ± 0.038 (n=5) | 0.594 ± 0.045 (n=5) | 0.638 ± 0.044 (n=5) | 0.618 ± 0.108 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.003 ± 0.017 (n=5) p=0.609 | 0.003 ± 0.015 (n=5) p=0.638 | 0.036 ± 0.056 (n=5) p=0.151 | 0.017 ± 0.093 (n=5) p=0.631 | 0.018 ± 0.074 (n=5) p=0.535 | 0.046 ± 0.018 (n=5) p=0.002 | 0.097 ± 0.043 (n=5) p=0.003 | 0.029 ± 0.091 (n=5) p=0.425 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.135 ± 0.000 (n=5) | 0.138 ± 0.016 (n=5) | 0.132 ± 0.014 (n=5) | 0.136 ± 0.053 (n=5) | 0.129 ± 0.064 (n=5) | 0.118 ± 0.057 (n=5) | 0.133 ± 0.034 (n=5) | 0.141 ± 0.031 (n=5) | 0.105 ± 0.050 (n=5) |
| S | 0.135 ± 0.000 (n=5) | 0.139 ± 0.022 (n=5) | 0.140 ± 0.022 (n=5) | 0.139 ± 0.049 (n=5) | 0.158 ± 0.034 (n=5) | 0.187 ± 0.027 (n=5) | 0.153 ± 0.047 (n=5) | 0.115 ± 0.028 (n=5) | 0.140 ± 0.084 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.002 ± 0.009 (n=5) p=0.645 | 0.008 ± 0.020 (n=5) p=0.312 | 0.003 ± 0.073 (n=5) p=0.914 | 0.029 ± 0.074 (n=5) p=0.343 | 0.069 ± 0.077 (n=5) p=0.068 | 0.020 ± 0.065 (n=5) p=0.437 | -0.026 ± 0.053 (n=5) p=0.246 | 0.035 ± 0.071 (n=5) p=0.242 |

figures: /cluster/rsi-runs/work/experiments/graph_audited/figures/graph_audited_eval_id_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_audited/figures/graph_audited_eval_id_pass1.png

eval_ood (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.347 ± 0.000 (n=5) | 0.360 ± 0.023 (n=5) | 0.368 ± 0.011 (n=5) | 0.362 ± 0.016 (n=5) | 0.362 ± 0.028 (n=5) | 0.366 ± 0.022 (n=5) | 0.379 ± 0.031 (n=5) | 0.362 ± 0.043 (n=5) | 0.388 ± 0.018 (n=5) |
| S | 0.347 ± 0.000 (n=5) | 0.354 ± 0.017 (n=5) | 0.375 ± 0.013 (n=5) | 0.385 ± 0.035 (n=5) | 0.386 ± 0.037 (n=5) | 0.398 ± 0.031 (n=5) | 0.401 ± 0.033 (n=5) | 0.430 ± 0.048 (n=5) | 0.424 ± 0.076 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.006 ± 0.016 (n=5) p=0.332 | 0.007 ± 0.015 (n=5) p=0.263 | 0.023 ± 0.031 (n=5) p=0.111 | 0.024 ± 0.038 (n=5) p=0.156 | 0.033 ± 0.038 (n=5) p=0.074 | 0.022 ± 0.035 (n=5) p=0.156 | 0.068 ± 0.014 (n=5) p=0.000 | 0.037 ± 0.063 (n=5) p=0.183 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.060 ± 0.000 (n=5) | 0.074 ± 0.017 (n=5) | 0.069 ± 0.009 (n=5) | 0.075 ± 0.028 (n=5) | 0.073 ± 0.032 (n=5) | 0.071 ± 0.040 (n=5) | 0.078 ± 0.024 (n=5) | 0.085 ± 0.024 (n=5) | 0.071 ± 0.028 (n=5) |
| S | 0.060 ± 0.000 (n=5) | 0.071 ± 0.015 (n=5) | 0.081 ± 0.024 (n=5) | 0.085 ± 0.032 (n=5) | 0.102 ± 0.020 (n=5) | 0.141 ± 0.008 (n=5) | 0.120 ± 0.047 (n=5) | 0.108 ± 0.026 (n=5) | 0.127 ± 0.051 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.002 ± 0.010 (n=5) p=0.531 | 0.012 ± 0.026 (n=5) p=0.269 | 0.010 ± 0.044 (n=5) p=0.576 | 0.029 ± 0.043 (n=5) p=0.132 | 0.069 ± 0.034 (n=5) p=0.005 | 0.042 ± 0.060 (n=5) p=0.125 | 0.022 ± 0.045 (n=5) p=0.237 | 0.056 ± 0.055 (n=5) p=0.047 |

figures: /cluster/rsi-runs/work/experiments/graph_audited/figures/graph_audited_eval_ood_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_audited/figures/graph_audited_eval_ood_pass1.png

## Comparison figures

- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_ood_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_ood_pass1.png
