# graph_unaudited_llama3.2-3b and graph_audited_llama3.2-3b  (2026-10-03 23:34:36, code a742f6f)

```
4657553                 p-graph_unaudited_llama3.2-3b  COMPLETED      0:0   01:45:28 
4657554                 t-graph_unaudited_llama3.2-3b  COMPLETED      0:0   07:12:42 
4671288                 x-graph_unaudited_llama3.2-3b  COMPLETED      0:0   09:46:29 
4657555                   t-graph_audited_llama3.2-3b  COMPLETED      0:0   07:16:06 
4671289                   x-graph_audited_llama3.2-3b  COMPLETED      0:0   08:48:55 
```

## graph_unaudited_llama3.2-3b

- status **complete** (exit 0), K = 64 (16 steps per round), matching {'error_fraction': 0.25, 'token_tolerance': 0}, 80 arm-rounds completed, stopped arms: none, attempts 2
- round-1 matching options on these pools:

```
{
 "pools": {
  "b00": {
   "answers": 16384,
   "truncated": 147,
   "correct": 0.23486328125,
   "target_errors": 2095
  },
  "b01": {
   "answers": 16384,
   "truncated": 138,
   "correct": 0.23663330078125,
   "target_errors": 2112
  },
  "b02": {
   "answers": 16384,
   "truncated": 121,
   "correct": 0.23223876953125,
   "target_errors": 2095
  },
  "b03": {
   "answers": 16384,
   "truncated": 153,
   "correct": 0.2325439453125,
   "target_errors": 2140
  },
  "b04": {
   "answers": 16384,
   "truncated": 132,
   "correct": 0.2342529296875,
   "target_errors": 2124
  }
 }
}
C:E fraction 0.25  tolerance None -> feasible K=64  J_E per block [382, 382, 382, 376, 390]
C:E fraction 0.25  tolerance 0    -> feasible K=64  J_E per block [184, 180, 178, 188, 192]   <- config.json
C:E fraction 0.125 tolerance None -> feasible K=64  J_E per block [382, 382, 382, 376, 390]
C:E fraction 0.125 tolerance 0    -> feasible K=64  J_E per block [184, 180, 178, 188, 192]
```

eval_id (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.248 ± 0.000 (n=5) | 0.471 ± 0.039 (n=5) | 0.499 ± 0.049 (n=5) | 0.509 ± 0.091 (n=5) | 0.558 ± 0.081 (n=5) | 0.630 ± 0.056 (n=5) | 0.645 ± 0.038 (n=5) | 0.662 ± 0.042 (n=5) | 0.640 ± 0.062 (n=5) |
| S | 0.248 ± 0.000 (n=5) | 0.482 ± 0.038 (n=5) | 0.532 ± 0.056 (n=5) | 0.544 ± 0.057 (n=5) | 0.560 ± 0.068 (n=5) | 0.587 ± 0.106 (n=5) | 0.603 ± 0.116 (n=5) | 0.586 ± 0.136 (n=5) | 0.611 ± 0.102 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.011 ± 0.020 (n=5) p=0.203 | 0.033 ± 0.099 (n=5) p=0.403 | 0.034 ± 0.100 (n=5) p=0.395 | 0.002 ± 0.105 (n=5) p=0.962 | -0.042 ± 0.110 (n=5) p=0.345 | -0.042 ± 0.120 (n=5) p=0.389 | -0.077 ± 0.140 (n=5) p=0.201 | -0.029 ± 0.111 (n=5) p=0.513 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.127 ± 0.000 (n=5) | 0.152 ± 0.013 (n=5) | 0.150 ± 0.023 (n=5) | 0.152 ± 0.020 (n=5) | 0.147 ± 0.041 (n=5) | 0.104 ± 0.032 (n=5) | 0.105 ± 0.052 (n=5) | 0.077 ± 0.032 (n=5) | 0.099 ± 0.062 (n=5) |
| S | 0.127 ± 0.000 (n=5) | 0.170 ± 0.018 (n=5) | 0.176 ± 0.050 (n=5) | 0.192 ± 0.055 (n=5) | 0.183 ± 0.069 (n=5) | 0.185 ± 0.075 (n=5) | 0.216 ± 0.074 (n=5) | 0.253 ± 0.111 (n=5) | 0.215 ± 0.120 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.018 ± 0.017 (n=5) p=0.042 | 0.026 ± 0.064 (n=5) p=0.322 | 0.040 ± 0.067 (n=5) p=0.177 | 0.036 ± 0.103 (n=5) p=0.387 | 0.081 ± 0.104 (n=5) p=0.098 | 0.110 ± 0.118 (n=5) p=0.061 | 0.177 ± 0.140 (n=5) p=0.025 | 0.116 ± 0.172 (n=5) p=0.134 |

figures: /cluster/rsi-runs/work/experiments/graph_unaudited_llama3.2-3b/figures/graph_unaudited_llama3.2-3b_eval_id_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_unaudited_llama3.2-3b/figures/graph_unaudited_llama3.2-3b_eval_id_pass1.png

eval_ood (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.209 ± 0.000 (n=5) | 0.372 ± 0.028 (n=5) | 0.378 ± 0.037 (n=5) | 0.399 ± 0.064 (n=5) | 0.430 ± 0.070 (n=5) | 0.466 ± 0.029 (n=5) | 0.467 ± 0.041 (n=5) | 0.472 ± 0.027 (n=5) | 0.454 ± 0.048 (n=5) |
| S | 0.209 ± 0.000 (n=5) | 0.390 ± 0.024 (n=5) | 0.424 ± 0.042 (n=5) | 0.435 ± 0.046 (n=5) | 0.447 ± 0.054 (n=5) | 0.458 ± 0.075 (n=5) | 0.472 ± 0.077 (n=5) | 0.463 ± 0.099 (n=5) | 0.478 ± 0.070 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.018 ± 0.019 (n=5) p=0.056 | 0.045 ± 0.059 (n=5) p=0.100 | 0.037 ± 0.070 (n=5) p=0.218 | 0.017 ± 0.081 (n=5) p=0.587 | -0.008 ± 0.064 (n=5) p=0.736 | 0.005 ± 0.085 (n=5) p=0.868 | -0.009 ± 0.088 (n=5) p=0.794 | 0.024 ± 0.064 (n=5) p=0.352 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.073 ± 0.000 (n=5) | 0.070 ± 0.009 (n=5) | 0.077 ± 0.018 (n=5) | 0.079 ± 0.008 (n=5) | 0.090 ± 0.040 (n=5) | 0.073 ± 0.028 (n=5) | 0.075 ± 0.039 (n=5) | 0.049 ± 0.010 (n=5) | 0.069 ± 0.046 (n=5) |
| S | 0.073 ± 0.000 (n=5) | 0.082 ± 0.010 (n=5) | 0.102 ± 0.048 (n=5) | 0.126 ± 0.051 (n=5) | 0.118 ± 0.061 (n=5) | 0.134 ± 0.036 (n=5) | 0.183 ± 0.028 (n=5) | 0.235 ± 0.053 (n=5) | 0.206 ± 0.108 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.012 ± 0.012 (n=5) p=0.048 | 0.025 ± 0.061 (n=5) p=0.314 | 0.047 ± 0.054 (n=5) p=0.074 | 0.029 ± 0.084 (n=5) p=0.398 | 0.061 ± 0.054 (n=5) p=0.035 | 0.107 ± 0.033 (n=5) p=0.001 | 0.186 ± 0.056 (n=5) p=0.001 | 0.137 ± 0.144 (n=5) p=0.057 |

figures: /cluster/rsi-runs/work/experiments/graph_unaudited_llama3.2-3b/figures/graph_unaudited_llama3.2-3b_eval_ood_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_unaudited_llama3.2-3b/figures/graph_unaudited_llama3.2-3b_eval_ood_pass1.png

## graph_audited_llama3.2-3b

- status **complete** (exit 0), K = 64 (16 steps per round), matching {'error_fraction': 0.25, 'token_tolerance': 0}, 80 arm-rounds completed, stopped arms: none, attempts 2

eval_id (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.248 ± 0.000 (n=5) | 0.452 ± 0.023 (n=5) | 0.506 ± 0.020 (n=5) | 0.548 ± 0.050 (n=5) | 0.586 ± 0.073 (n=5) | 0.585 ± 0.080 (n=5) | 0.626 ± 0.027 (n=5) | 0.626 ± 0.045 (n=5) | 0.651 ± 0.075 (n=5) |
| S | 0.248 ± 0.000 (n=5) | 0.447 ± 0.011 (n=5) | 0.531 ± 0.066 (n=5) | 0.585 ± 0.044 (n=5) | 0.620 ± 0.031 (n=5) | 0.634 ± 0.061 (n=5) | 0.655 ± 0.046 (n=5) | 0.669 ± 0.073 (n=5) | 0.620 ± 0.070 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | -0.006 ± 0.015 (n=5) p=0.344 | 0.025 ± 0.058 (n=5) p=0.293 | 0.036 ± 0.033 (n=5) p=0.037 | 0.034 ± 0.067 (n=5) p=0.239 | 0.049 ± 0.097 (n=5) p=0.235 | 0.030 ± 0.060 (n=5) p=0.241 | 0.043 ± 0.065 (n=5) p=0.138 | -0.031 ± 0.086 (n=5) p=0.378 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.127 ± 0.000 (n=5) | 0.175 ± 0.039 (n=5) | 0.147 ± 0.024 (n=5) | 0.155 ± 0.055 (n=5) | 0.129 ± 0.049 (n=5) | 0.113 ± 0.088 (n=5) | 0.083 ± 0.070 (n=5) | 0.063 ± 0.056 (n=5) | 0.061 ± 0.043 (n=5) |
| S | 0.127 ± 0.000 (n=5) | 0.193 ± 0.046 (n=5) | 0.152 ± 0.063 (n=5) | 0.136 ± 0.063 (n=5) | 0.137 ± 0.068 (n=5) | 0.133 ± 0.095 (n=5) | 0.144 ± 0.047 (n=5) | 0.155 ± 0.066 (n=5) | 0.222 ± 0.083 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.018 ± 0.021 (n=5) p=0.081 | 0.005 ± 0.054 (n=5) p=0.794 | -0.019 ± 0.063 (n=5) p=0.459 | 0.007 ± 0.063 (n=5) p=0.757 | 0.021 ± 0.082 (n=5) p=0.523 | 0.061 ± 0.078 (n=5) p=0.094 | 0.092 ± 0.113 (n=5) p=0.086 | 0.161 ± 0.101 (n=5) p=0.011 |

figures: /cluster/rsi-runs/work/experiments/graph_audited_llama3.2-3b/figures/graph_audited_llama3.2-3b_eval_id_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_audited_llama3.2-3b/figures/graph_audited_llama3.2-3b_eval_id_pass1.png

eval_ood (greedy Pass@1, mean over seed blocks ± 95% half width):

| pass1 | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.209 ± 0.000 (n=5) | 0.366 ± 0.019 (n=5) | 0.396 ± 0.016 (n=5) | 0.437 ± 0.037 (n=5) | 0.449 ± 0.032 (n=5) | 0.429 ± 0.036 (n=5) | 0.454 ± 0.052 (n=5) | 0.435 ± 0.050 (n=5) | 0.459 ± 0.087 (n=5) |
| S | 0.209 ± 0.000 (n=5) | 0.370 ± 0.011 (n=5) | 0.411 ± 0.027 (n=5) | 0.445 ± 0.025 (n=5) | 0.476 ± 0.037 (n=5) | 0.479 ± 0.051 (n=5) | 0.488 ± 0.046 (n=5) | 0.500 ± 0.049 (n=5) | 0.474 ± 0.024 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.004 ± 0.009 (n=5) p=0.307 | 0.015 ± 0.023 (n=5) p=0.150 | 0.008 ± 0.048 (n=5) p=0.655 | 0.027 ± 0.044 (n=5) p=0.163 | 0.050 ± 0.067 (n=5) p=0.110 | 0.034 ± 0.072 (n=5) p=0.253 | 0.065 ± 0.033 (n=5) p=0.006 | 0.015 ± 0.081 (n=5) p=0.636 |

| target_error_rate | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.073 ± 0.000 (n=5) | 0.083 ± 0.033 (n=5) | 0.072 ± 0.022 (n=5) | 0.091 ± 0.058 (n=5) | 0.074 ± 0.044 (n=5) | 0.075 ± 0.075 (n=5) | 0.059 ± 0.065 (n=5) | 0.042 ± 0.048 (n=5) | 0.049 ± 0.037 (n=5) |
| S | 0.073 ± 0.000 (n=5) | 0.095 ± 0.046 (n=5) | 0.086 ± 0.048 (n=5) | 0.101 ± 0.062 (n=5) | 0.120 ± 0.085 (n=5) | 0.113 ± 0.099 (n=5) | 0.122 ± 0.052 (n=5) | 0.130 ± 0.050 (n=5) | 0.189 ± 0.104 (n=5) |
| S-R | 0.000 ± 0.000 (n=5) | 0.011 ± 0.021 (n=5) p=0.211 | 0.014 ± 0.027 (n=5) p=0.224 | 0.010 ± 0.047 (n=5) p=0.580 | 0.046 ± 0.091 (n=5) p=0.231 | 0.039 ± 0.050 (n=5) p=0.097 | 0.063 ± 0.060 (n=5) p=0.043 | 0.089 ± 0.091 (n=5) p=0.054 | 0.139 ± 0.123 (n=5) p=0.035 |

figures: /cluster/rsi-runs/work/experiments/graph_audited_llama3.2-3b/figures/graph_audited_llama3.2-3b_eval_ood_by_difficulty.png, /cluster/rsi-runs/work/experiments/graph_audited_llama3.2-3b/figures/graph_audited_llama3.2-3b_eval_ood_pass1.png

## Comparison figures

- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_eval_ood_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_id_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_id_pass1.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_ood_by_difficulty.png
- /cluster/rsi-runs/work/experiments/figures/graph_unaudited_vs_graph_audited_eval_ood_pass1.png
