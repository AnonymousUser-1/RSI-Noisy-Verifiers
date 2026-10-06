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

# Pools are not shared: this experiment's round-1 pools are imported (below), and it has no audited
# sibling in experiments/.
SHARE_POOLS_WITH=

# The round-1 pools, shared adapter, data and round-1 matching are in the folder
# RSI-Qwen3-4B-Graph-paired-inputs-20261004 at the repository root
# and imported (experiments/import_pools.py),
# not drawn: they are the Qwen3-4B pools drawn before generation.repetition_stop was introduced, and
# this experiment serves the one-step comparison on them (experiments/one_step.sh).  config.json states
# the sampling those pools were drawn with (repetition_stop null); evaluation.json states the current
# loop stop for the greedy answers.
POOLS_IMPORTED=yes
# The folder that holds those inputs (named in the error when they are missing).
INPUTS_RELEASE=RSI-Qwen3-4B-Graph-paired-inputs-20261004
# The data those pools were drawn from, byte for byte (the same questions as data/graph).
DATA_NAME=graph-qwen3-4b-import

# The name on this experiment's figures.
LABEL="unaudited"
