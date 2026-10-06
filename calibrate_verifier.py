#!/usr/bin/env python3
"""Calibrate a fixed threshold on a separate, counted set of correctness labels."""
import argparse
import math
from pathlib import Path

from rsi.backends import backend
from rsi.common import digest, load_config, read_jsonl, seed_for, verify_dataset, write_json, write_jsonl
from rsi.experiment import pin_config
from rsi.tasks import judge
from rsi.verifiers import verification_prompt, verifier_identity


def calibrate(config, data, out, target_tpr=0.9):
    out, data = Path(out), Path(data)
    if out.exists():
        raise FileExistsError(out)
    dataset = verify_dataset(data)
    cfg = pin_config(config)
    tasks_list = read_jsonl(data / "calibration.jsonl")
    tasks = {r["id"]: r for r in tasks_list}
    solver = backend(cfg)
    try:
        candidates = solver.generate(tasks_list, cfg["generation"]["candidates"], seed_for("calibration", dataset["seed"]))
    finally:
        solver.close()
    identity = verifier_identity(cfg)
    model = backend(cfg, model_name=identity["model"], revision=identity["revision"])
    try:
        scores = model.score_verdicts([verification_prompt(tasks[c["task_id"]]["prompt"], c["response"]) for c in candidates])
    finally:
        model.close()
    rows = [dict(c, score=s, **judge(tasks[c["task_id"]], c["response"])) for c, s in zip(candidates, scores)]
    positive = sorted(r["score"] for r in rows if r["correct"])
    negative = [r["score"] for r in rows if not r["correct"]]
    if not positive or not negative:
        raise ValueError("Calibration requires both correct and incorrect examples; adjust development task difficulty")
    threshold = positive[min(len(positive)-1, math.floor((1-target_tpr)*len(positive)))]
    write_jsonl(str(out) + ".pairs.jsonl", rows)
    result = {"verifier": identity, "backend": cfg["backend"], "dataset_hash": digest(dataset),
              "threshold": threshold, "target_tpr": target_tpr, "trusted_queries": len(rows),
              "calibration_tpr": sum(s >= threshold for s in positive)/len(positive),
              "calibration_fpr": sum(s >= threshold for s in negative)/len(negative),
              "solver": {"model": cfg["model"], "revision": cfg["revision"]}, "config": cfg}
    write_json(out, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-tpr", type=float, default=0.9)
    p.add_argument("--backend", choices=["hf", "mock"])
    a = p.parse_args()
    if not 0 < a.target_tpr <= 1:
        p.error("target-tpr must be in (0,1]")
    c = load_config(a.config, backend=a.backend)
    if not c["verifier"]["kind"].startswith("llm_"):
        p.error("Choose an llm_fixed or llm_self config")
    print(calibrate(c, a.data, a.out, a.target_tpr))
