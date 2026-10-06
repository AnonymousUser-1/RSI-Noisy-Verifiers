#!/usr/bin/env python3
"""Experiment 1: exact population recursion and finite-sample Monte Carlo."""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from rsi.common import seed_for, write_json


def trajectory(p0, a, b, n, rounds, eta, seed):
    rng = np.random.default_rng(seed)
    p, population = p0, p0
    rows = [{"round": 0, "accuracy": p, "population_accuracy": population, "accepted": None}]
    for t in range(1, rounds + 1):
        good, bad, _ = rng.multinomial(n, [p * a, (1-p) * b, max(0, 1-p*a-(1-p)*b)])
        if good + bad:
            p = (1-eta)*p + eta*good/(good+bad)
        denominator = population*a + (1-population)*b
        if denominator:
            population = (1-eta)*population + eta*population*a/denominator
        rows.append({"round": t, "accuracy": float(p), "population_accuracy": population,
                     "accepted": int(good+bad)})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--p0", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    p.add_argument("--tpr", nargs="+", type=float, default=[0.6, 0.8, 0.95])
    p.add_argument("--fpr", nargs="+", type=float, default=[0, 0.05, 0.15, 0.3, 0.6, 0.9])
    p.add_argument("--samples", nargs="+", type=int, default=[128, 512, 2048])
    args = p.parse_args()
    if any(not 0 <= x <= 1 for x in args.p0 + args.tpr + args.fpr + [args.eta]):
        p.error("Probabilities and eta must lie in [0,1]")
    if min(args.rounds, args.seeds, *args.samples) < 1:
        p.error("Counts must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trajectories.csv").exists():
        p.error("Use a new output folder")
    fields = ["p0", "tpr", "fpr", "samples", "seed", "round", "accuracy", "population_accuracy", "accepted"]
    with (out / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for p0, a, b, n, seed in itertools.product(args.p0, args.tpr, args.fpr, args.samples, range(args.seeds)):
            for row in trajectory(p0, a, b, n, args.rounds, args.eta, seed_for(p0, a, b, n, seed)):
                writer.writerow(dict(p0=p0, tpr=a, fpr=b, samples=n, seed=seed, **row))
    write_json(out / "config.json", vars(args))
    print("Saved", out / "trajectories.csv")
