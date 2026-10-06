# Settings of the graph_unaudited_qwen3-4b experiment.  Its hyperparameters are in config.json (sampling,
# training, auditing, the R/S selection) and evaluation.json (Pass@1 evaluation) next to this file; editing any of the
# three changes this experiment only.  See experiments/README.md.

# Task: graph (S targets nonshortest paths) | arithmetic (S targets ignore_parentheses errors).
TASK=graph

# Study id, recorded in every pool and run.  Keep it fixed while the study runs; change the -v01
# suffix when you start it again with other settings (and give it a fresh output folder).
STUDY=md-graph-unaudited-qwen3-4b-2048x8-v01

# The pin file of the base model; it must name config.json's "model" and "revision" (the entries
# refuse any other base).  Other models' pins are in matched-dynamics/base_pin_*.json.
PIN_FILE=matched-dynamics/base_pin_qwen3-4b.json

# Seed blocks: one round-1 pool and one independent R/S replicate per seed (b00, b01, ...).
SEEDS="0 1 2 3 4"

# This experiment draws its own pools. There is no audited sibling in experiments/.
SHARE_POOLS_WITH=

# Default: run.sh generates data, pools and the shared adapter, then trains and evaluates.
# config.json explicitly keeps repetition_stop off for sampling; evaluation.json states its
# evaluation stop. Neither file is changed when the optional original inputs are imported.
# An explicit import into a fresh RSI_ROOT selects original-input one-step replay instead.
# That replay keeps its original matching and cannot start a new multi-round run in that folder.
POOLS_IMPORT_ALLOWED=yes
# Keep task data isolated so optional imports cannot replace another model's generated data.
DATA_NAME=graph-qwen3-4b-import

# The name on this experiment's figures.
LABEL="unaudited"
