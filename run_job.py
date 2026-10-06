#!/usr/bin/env python3
"""One GPU job: one trajectory followed by independent checkpoint evaluation."""
import argparse
from evaluate import evaluate
from rsi.common import load_config
from rsi.experiment import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--frozen-null", action="store_true")
    # The matched-error comparison only exists if both branches are drawn from one
    # identical candidate pool.  run_experiment.py already accepts --initial-pool;
    # without this flag run_job.py would silently regenerate a fresh pool per
    # branch, which is the defect KNOWN_GAPS.md B records.
    p.add_argument("--initial-pool", help="Shared round-1 candidate JSONL generated with sample_candidates.py")
    a = p.parse_args()
    c = load_config(a.config, a.seed)
    run(c, a.data, a.out, a.resume, a.initial_pool)
    evaluate(a.out, full_rounds=sorted({0, c["rounds"]//2, c["rounds"]}))
    if a.frozen_null:
        evaluate(a.out, full_rounds=sorted({0, c["rounds"]//2, c["rounds"]}), frozen_null=True)
