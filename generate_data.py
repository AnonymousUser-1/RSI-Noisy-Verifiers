#!/usr/bin/env python3
"""Generate immutable, disjoint task splits with content hashes.

`gradient_reference` (64 prompts by default) is the held-out set the reference
gradient h is computed on (HANDOFF A.1).  It is for diagnostics only: nothing
that builds candidate pools, matches or tunes reads it.  It is drawn like the
training prompts -- same task, same generator, difficulty i % 3 -- and it is
generated last.  `seen` is shared across splits, so a split generated earlier
would take instances that a later split then has to redraw; generated last, it
leaves every other split, and its file hash, exactly as before.
"""
import argparse
from pathlib import Path

from rsi.common import file_hash, rng_for, write_json, write_jsonl
from rsi.tasks import make_instance


def generate(out, kind, seed, rounds, per_round, dev, calibration, eval_id, eval_ood, gradient_reference=64):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be new or empty: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    seen, files = set(), {}
    specs = [("train_%03d" % r, per_round) for r in range(1, rounds + 1)]
    specs += [("dev", dev), ("calibration", calibration), ("eval_id", eval_id), ("eval_ood", eval_ood)]
    specs += [("gradient_reference", gradient_reference)]  # last: see the module docstring
    for split, count in specs:
        rng, rows = rng_for(seed, kind, split), []
        for i in range(count):
            for _ in range(10000):
                row = make_instance(kind, rng, split, i % 3)
                if row["id"] not in seen:
                    seen.add(row["id"])
                    rows.append(row)
                    break
            else:
                raise RuntimeError("Could not generate a disjoint instance")
        name = split + ".jsonl"
        write_jsonl(out / name, rows)
        files[name] = file_hash(out / name)
    manifest = {"version": 1, "task": kind, "seed": seed, "rounds": rounds, "files": files,
                "counts": dict(specs), "unique_instances": len(seen)}
    write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--task", choices=["graph", "arithmetic"], required=True)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--per-round", type=int, default=2048)
    p.add_argument("--dev", type=int, default=512)
    p.add_argument("--calibration", type=int, default=500, help="500 prompts x 4 answers = 2000 calibration pairs")
    p.add_argument("--eval-id", type=int, default=2000)
    p.add_argument("--eval-ood", type=int, default=1000)
    p.add_argument("--gradient-reference", type=int, default=64,
                   help="Prompts in gradient_reference.jsonl, the diagnostic-only set h is computed on")
    a = p.parse_args()
    if min(a.rounds, a.per_round, a.dev, a.calibration, a.eval_id, a.eval_ood, a.gradient_reference) < 1:
        p.error("All counts must be positive")
    print(generate(a.out, a.task, a.seed, a.rounds, a.per_round, a.dev, a.calibration, a.eval_id, a.eval_ood,
                   a.gradient_reference))
