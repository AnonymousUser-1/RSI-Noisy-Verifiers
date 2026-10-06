#!/usr/bin/env python3
"""CPU-only end-to-end plumbing exercise. All model behavior is explicitly fake."""
import argparse
import copy
from pathlib import Path

from analyze import analyze
from evaluate import evaluate
from generate_data import generate
from rsi.common import DEFAULTS, write_json
from rsi.experiment import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--plots", action="store_true")
    a = p.parse_args()
    root = Path(a.out)
    if root.exists() and any(root.iterdir()):
        p.error("Use a new output directory")
    generate(root/"data", "graph", 11, 2, 24, 6, 12, 12, 12)
    for kind in ["iid", "persistent"]:
        for seed in [0, 1]:
            cfg = copy.deepcopy(DEFAULTS)
            cfg.update(backend="mock", rounds=2, seed=seed)
            cfg["training"]["examples"] = 12
            cfg["verifier"]["kind"] = kind
            cfg["audit"].update(policy="adaptive", budget=6)
            output = root/"runs"/kind/("seed_"+str(seed))
            run(cfg, root/"data", output)
            evaluate(output, full_rounds=[0, 2], full_draws=2)
    analyze(root/"runs", root/"analysis", include_demo=True, plots=a.plots)
    write_json(root/"SMOKE_REPORT.json", {"status": "passed", "DEMO_ONLY": True,
               "meaning": "Pipeline wiring exercised; no actual language model training or scientific evidence"})
    print("SMOKE TEST PASSED; outputs are DEMO ONLY:", root)
