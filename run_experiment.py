#!/usr/bin/env python3
"""Run one trajectory. No final-test evaluation occurs inside the training loop."""
import argparse
from rsi.common import load_config
from rsi.experiment import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--backend", choices=["hf", "mock"])
    p.add_argument("--calibration", help="Override verifier calibration path")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--initial-pool", help="Shared round-1 candidate JSONL generated with sample_candidates.py")
    args = p.parse_args()
    config = load_config(args.config, args.seed, args.backend)
    if args.calibration:
        config["verifier"]["calibration"] = args.calibration
    print(run(config, args.data, args.out, args.resume, args.initial_pool))
