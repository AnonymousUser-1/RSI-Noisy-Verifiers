#!/usr/bin/env python3
"""Evaluate committed checkpoints after training; repeated draws estimate pass@1."""
import argparse
import time
from collections import defaultdict
from pathlib import Path

from rsi.backends import backend
from rsi.common import digest, read_json, read_jsonl, seed_for, verify_dataset, write_json, write_jsonl
from rsi.tasks import judge, stratum


def evaluate(run, splits=None, rounds=None, draws=1, full_draws=4,
             full_rounds=(0, 4, 8), frozen_null=False):
    run = Path(run).resolve()
    manifest = read_json(run / "run.json")
    cfg, data = manifest["config"], Path(manifest["data_path"])
    if not (run / "finished.json").exists():
        raise ValueError("Finish the trajectory before final-test evaluation")
    dataset = verify_dataset(data)
    if digest(dataset) != manifest["dataset_hash"]:
        raise ValueError("Dataset mismatch")
    if splits is None:
        # The dataset's held-out splits; an imported dataset may have no eval_ood (GSM8K).
        splits = [s for s in ("eval_id", "eval_ood") if s + ".jsonl" in dataset["files"]]
    missing = [s for s in splits if s + ".jsonl" not in dataset["files"]]
    if missing:
        raise ValueError("The dataset has no %s split" % ", ".join(missing))
    rounds = list(range(cfg["rounds"]+1)) if rounds is None else rounds
    mode = "frozen_null" if frozen_null else "evaluation"
    results = []
    for t in rounds:
        if not 0 <= t <= cfg["rounds"]:
            raise ValueError("Invalid checkpoint round")
        record = read_json(run / ("round_%03d/complete.json" % t)) if t else {"adapter": None}
        adapter = run / record["adapter"] if record["adapter"] and not frozen_null else None
        k = full_draws if t in full_rounds else draws
        pending = []
        for split in splits:
            output = run / mode / ("round_%03d_%s.json" % (t, split))
            if output.exists():
                old = read_json(output)
                if old["draws"] != k:
                    raise ValueError("Existing evaluation uses a different draw count")
                results.append(old)
            else:
                pending.append((split, output))
        if not pending:
            continue
        start = time.perf_counter()
        model = backend(cfg, adapter=adapter)
        try:
            for split, output in pending:
                tasks_list = read_jsonl(data / (split + ".jsonl"))
                tasks = {r["id"]: r for r in tasks_list}
                rows = model.generate(tasks_list, k, seed_for(cfg["seed"], t, split, "evaluation"))
                rows = [dict(r, stratum=stratum(tasks[r["task_id"]]), **judge(tasks[r["task_id"]], r["response"])) for r in rows]
                write_jsonl(output.with_suffix(".jsonl"), rows)
                groups = defaultdict(list)
                for r in rows:
                    groups[r["stratum"]].append(r["correct"])
                truncated = sum(bool(r.get("truncated")) for r in rows)
                if truncated:
                    print("WARNING: %d answers of round %d %s hit max_new_tokens=%d without a stop token; they are "
                          "judged as they stand and counted in truncated_answers" % (
                              truncated, t, split, cfg["generation"]["max_new_tokens"]), flush=True)
                result = {"round": t, "split": split, "draws": k, "problems": len(tasks),
                          "answers": len(rows), "truncated_answers": truncated,
                          "max_new_tokens": cfg["generation"]["max_new_tokens"],
                          "accuracy": sum(r["correct"] for r in rows)/len(rows),
                          "group_accuracy": {g: sum(xs)/len(xs) for g, xs in groups.items()},
                          "cumulative_audit_queries": record.get("cumulative_audit_queries", 0),
                          "calibration_queries": (manifest["calibration"] or {}).get("trusted_queries", 0),
                          "frozen_null": frozen_null, "DEMO_ONLY": manifest["DEMO_ONLY"]}
                write_json(output, result)
                results.append(result)
        finally:
            model.close()
        write_json(run / mode / ("timing_%03d.json" % t),
                   {"seconds": time.perf_counter()-start, "round": t,
                    "gpu_count": 1 if cfg["backend"] == "hf" and cfg["device"].startswith("cuda") else 0})
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--splits", nargs="+", choices=["dev", "eval_id", "eval_ood"],
                   help="Default: eval_id and eval_ood, whichever the dataset has")
    p.add_argument("--rounds", nargs="+", type=int)
    p.add_argument("--draws", type=int, default=1)
    p.add_argument("--full-draws", type=int, default=4)
    p.add_argument("--full-rounds", nargs="*", type=int, default=[0, 4, 8])
    p.add_argument("--frozen-null", action="store_true")
    a = p.parse_args()
    if min(a.draws, a.full_draws) < 1:
        p.error("Draw counts must be positive")
    for row in evaluate(a.run, a.splits, a.rounds, a.draws, a.full_draws, a.full_rounds, a.frozen_null):
        print(row)
