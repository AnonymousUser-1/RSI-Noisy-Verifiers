#!/usr/bin/env python3
"""Write explicit run configs and a job manifest. Never starts GPU jobs."""
import argparse
import copy
from pathlib import Path

from rsi.common import DEFAULTS, write_json
from rsi.backends import resolve_revision
from rsi.tasks import TASKS


def prepare(study, out, data_root, runs_root, calibration_dir, backend_name="hf", pin=False,
            model=None, revision=None, tasks=None):
    out, data_root, runs_root = Path(out).resolve(), Path(data_root).resolve(), Path(runs_root).resolve()
    if revision and not model:
        raise ValueError("--revision pins the --model checkpoint; name the model as well")
    if revision and len(revision) == 40 and revision != revision.lower():
        raise ValueError("--revision: write the commit id in lowercase, as the Hub cache names it")
    if model and study in ("learned", "all"):
        # read_calibration checks the verifier, not the solver, so a threshold calibrated on the
        # default solver's answers would be reused for another solver without complaint.
        raise ValueError("--model is not supported for the learned study: its verifier calibrations "
                         "are made with the default solver")
    if tasks and study != "pilot":
        raise ValueError("--tasks selects the pilot's tasks; the other studies fix their own")
    pilot_tasks = list(tasks or ["graph"])
    if len(set(pilot_tasks)) != len(pilot_tasks) or not set(pilot_tasks) <= set(TASKS):
        raise ValueError("--tasks takes tasks from %s, each at most once" % ", ".join(TASKS))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new suite directory")
    base = copy.deepcopy(DEFAULTS)
    base["backend"] = backend_name
    # --model replaces the default solver; --revision pins that checkpoint, and a commit id
    # lets an offline GPU job load it without asking the Hub.
    if model:
        base["model"] = model
    if revision:
        base["revision"] = revision
    requested = {base["model"]: base["revision"]}
    pins = {}

    def pin_one(name):
        if name not in pins:
            pins[name] = resolve_revision(name, requested.get(name, "main"))
        return pins[name]

    jobs = []

    def add(section, task, condition, changes, seeds, frozen_null=False):
        from rsi.common import merge_dict
        cfg = merge_dict(base, changes)
        if cfg["model"] != base["model"]:
            # A study that names its own solver (replication) does not inherit --revision.
            cfg["revision"] = DEFAULTS["revision"]
        if pin and backend_name == "hf":
            cfg["revision"] = pin_one(cfg["model"])
            if cfg["verifier"]["kind"] == "llm_fixed":
                cfg["verifier"]["revision"] = pin_one(cfg["verifier"]["model"])
        name = section + "_" + task + "_" + condition
        config_path = out / (name + ".json")
        write_json(config_path, cfg)
        for seed in seeds:
            job_id = name + "_seed" + str(seed)
            argv = ["run_job.py", "--config", str(config_path), "--data", str(data_root/task),
                    "--out", str(runs_root/section/task/condition/("seed_"+str(seed))),
                    "--seed", str(seed), "--resume"]
            if frozen_null:
                argv.append("--frozen-null")
            jobs.append({"id": job_id, "argv": argv, "study": section})

    studies = ["controlled", "learned", "auditing", "replication"] if study == "all" else [study]
    for section in studies:
        if section == "pilot":
            for task in pilot_tasks:
                for kind in ["exact", "iid", "persistent"]:
                    add(section, task, kind, {"rounds": 2, "training": {"examples": 64},
                                              "verifier": {"kind": kind}}, range(2), kind == "exact")
        elif section == "controlled":
            for task, kinds in [("graph", ["exact", "unfiltered", "iid", "persistent", "rotating"]),
                                ("arithmetic", ["exact", "iid", "persistent"])]:
                for kind in kinds:
                    add(section, task, kind, {"verifier": {"kind": kind}}, range(5), kind == "exact")
        elif section == "learned":
            for name, kind, model in [("fixed", "llm_fixed", "Qwen/Qwen3-1.7B"),
                                       ("self", "llm_self", "Qwen/Qwen3-1.7B"),
                                       ("cross", "llm_fixed", "HuggingFaceTB/SmolLM3-3B")]:
                cal_name = "graph_" + ("smol" if name == "cross" else "qwen") + ".json"
                add(section, "graph", name, {"verifier": {"kind": kind, "model": model,
                    "calibration": str(Path(calibration_dir).resolve()/cal_name)}}, range(5))
        elif section == "auditing":
            for policy in ["uniform", "balanced", "adaptive"]:
                for budget in [16, 64, 256]:
                    add(section, "graph", policy+"_"+str(budget), {"verifier": {"kind": "persistent"},
                        "audit": {"policy": policy, "budget": budget}}, range(5))
            add(section, "graph", "adaptive_64_deletion", {"verifier": {"kind": "persistent"},
                "audit": {"policy": "adaptive", "budget": 64, "weighting": False}}, range(5))
        elif section == "replication":
            for name, kind, policy, budget in [("iid", "iid", "none", 0), ("persistent", "persistent", "none", 0),
                                               ("audit64", "persistent", "adaptive", 64)]:
                add(section, "graph", name, {"model": "Qwen/Qwen3-4B", "verifier": {"kind": kind},
                    "audit": {"policy": policy, "budget": budget}}, range(3))
    write_json(out / "jobs.json", {"backend": backend_name, "jobs": jobs, "pinned_revisions": pins})
    print("Prepared %d jobs at %s. No jobs launched." % (len(jobs), out/"jobs.json"))
    return jobs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=["pilot", "controlled", "learned", "auditing", "replication", "all"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--calibration-dir", default="calibration")
    p.add_argument("--backend", choices=["hf", "mock"], default="hf")
    p.add_argument("--pin", action="store_true", help="Resolve model revisions now; requires Hub access")
    p.add_argument("--model", help="Solver model ID for every study that does not name its own "
                                   "(default %s)" % DEFAULTS["model"])
    p.add_argument("--revision", help="Revision of --model; a 40-hex commit id needs no Hub access")
    p.add_argument("--tasks", nargs="+", choices=TASKS,
                   help="Pilot tasks (default: graph); gsm8k and dmmath read data_root/<task> made by import_data.py")
    a = p.parse_args()
    prepare(a.study, a.out, a.data_root, a.runs_root, a.calibration_dir, a.backend, a.pin,
            a.model, a.revision, a.tasks)
