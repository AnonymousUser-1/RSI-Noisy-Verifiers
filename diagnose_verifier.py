#!/usr/bin/env python3
"""Post-hoc verifier drift on a fixed, disjoint bank generated from development tasks."""
import argparse
import time
from pathlib import Path

from rsi.backends import backend
from rsi.common import digest, read_json, read_jsonl, seed_for, verify_dataset, write_json, write_jsonl
from rsi.tasks import judge
from rsi.verifiers import verification_prompt, verifier_identity

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    a = p.parse_args()
    run = Path(a.run).resolve()
    manifest = read_json(run/"run.json")
    cfg, data = manifest["config"], Path(manifest["data_path"])
    if not (run/"finished.json").exists():
        p.error("Finish training before diagnostics")
    if not cfg["verifier"]["kind"].startswith("llm_"):
        p.error("This diagnostic is for learned verifiers")
    if digest(verify_dataset(data)) != manifest["dataset_hash"]:
        p.error("Dataset changed")
    out = run/"verifier_diagnostics"
    tasks_list = read_jsonl(data/"dev.jsonl")
    tasks = {r["id"]: r for r in tasks_list}
    bank = out/"bank.jsonl"
    if not bank.exists():
        model = backend(cfg)
        try:
            rows = model.generate(tasks_list, cfg["generation"]["candidates"], seed_for("diagnostic_bank", cfg["seed"]))
        finally:
            model.close()
        write_jsonl(bank, rows)
    rows = read_jsonl(bank)
    prompts = [verification_prompt(tasks[r["task_id"]]["prompt"], r["response"]) for r in rows]
    identity = verifier_identity(cfg)
    for t in range(1, cfg["rounds"]+1):
        target = out/("round_%03d.jsonl" % t)
        if target.exists():
            continue
        start = time.perf_counter()
        previous = read_json(run/("round_%03d/complete.json" % (t-1))) if t > 1 else {"adapter": None}
        adapter = run/previous["adapter"] if previous["adapter"] and cfg["verifier"]["kind"] == "llm_self" else None
        verifier = backend(cfg, adapter=adapter, model_name=identity["model"], revision=identity["revision"])
        try:
            scores = verifier.score_verdicts(prompts)
        finally:
            verifier.close()
        result = [dict(r, score=s, accepted=s >= manifest["calibration"]["threshold"],
                       **judge(tasks[r["task_id"]], r["response"])) for r, s in zip(rows, scores)]
        write_jsonl(target, result)
        write_json(target.with_suffix(".json"), {"round": t, "offline_measurement_labels": len(result),
                   "brier": sum((r["score"]-r["correct"])**2 for r in result)/len(result),
                   "seconds": time.perf_counter()-start, "DEMO_ONLY": manifest["DEMO_ONLY"]})
