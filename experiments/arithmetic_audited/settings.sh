# Settings of the arithmetic_audited experiment.  Its hyperparameters are in config.json (sampling,
# training, auditing, the R/S selection) and evaluation.json (Pass@1 evaluation) next to this file; editing any of the
# three changes this experiment only.  See experiments/README.md.

# Task: graph (S targets nonshortest paths) | arithmetic (S targets ignore_parentheses errors).
TASK=arithmetic

# Study id, recorded in every pool and run.  Keep it fixed while the study runs; change the -v01
# suffix when you start it again with other settings (and give it a fresh output folder).
STUDY=md-arithmetic-audited-qwen3-1.7b-2048x8-v01

# The pin file of the base model; it must name config.json's "model" and "revision" (the entries
# refuse any other base).  Other models' pins are in configs/pins/base_pin_*.json.
PIN_FILE=configs/pins/base_pin.json

# Seed blocks: one round-1 pool and one independent R/S replicate per seed (b00, b01, ...).
SEEDS="0 1 2 3 4"

# Reuse the round-1 pools of this experiment when they were drawn with exactly the same base, data
# and sampling (generation fields) as config.json, so the two runs start from the same answers;
# otherwise this experiment draws its own.  Empty: always draw its own.
SHARE_POOLS_WITH=arithmetic_unaudited

# The name on this experiment's figures.
LABEL="audited B=16"
