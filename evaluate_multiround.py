#!/usr/bin/env python3
"""Pass@1 after every round of a multi-round R/S run (run_iterative_experiment.py output).

  python evaluate_multiround.py --run OUT --config configs/matched_evaluation.json

One greedy answer per held-out question (temperature 0: Pass@1 is the share answered correctly),
for round 0 -- the base model, the same for every block and arm, evaluated once -- and for every
completed round of every block/arm.  The round-0 model is the base without an adapter: the shared
adapter starts with LoRA B = 0, so base + shared adapter computes the same function.  --config
names the splits (configs/matched_evaluation.json: eval_id, in distribution, and eval_ood, larger
graphs or expressions) and the batch size (configs/README.md); each checkpoint is loaded once for
all splits.  The answer length cap (max_new_tokens, max_sequence_length) is the run's own; an
answer still running at the cap is judged as it stands and counted in truncated_answers.

Writes, for each split, OUT/evaluation_greedy_SPLIT/:
  protocol.json             the frozen decoding, split and base; a rerun with other settings is refused
  round_000.json/.jsonl     the base model
  bXX_ARM_round_00t.json/.jsonl
Each .json holds Pass@1 overall and per difficulty, the error-type counts and the target-error
rate (S's target: nonshortest for graph, ignore_parentheses for arithmetic); each .jsonl holds every
answer with its judgement.  With --cost-records DIR, each checkpoint-split attempt also writes a cost
record there (rsi/cost.py), failed attempts included.  Re-runnable: finished checkpoints are skipped, so it can follow a
training run that is still going (incomplete rounds are left for the next call).
Questions are generated longest prompt first, one batch at a time.  A batch the GPU cannot hold --
at the 2,048-token cap usually the first one in which an answer loops to the cap -- is generated
again at half the batch size, which then holds for the rest of the run; the batches already done
are kept (the batch size is not part of the protocol).
Exit 0 every completed checkpoint evaluated, 1 refused or failed.
"""
import argparse
import contextlib
import copy
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rsi import cost
from rsi.backends import backend
from rsi.common import digest, file_hash, read_json, read_jsonl, verify_dataset, write_json, write_jsonl
from rsi.matching import targets_for
from rsi.tasks import judge

BASE_KEYS = ("backend", "model", "revision", "dtype", "device")
# evaluation.json without "repetition_stop": the evaluation stops loops as the run sampled.
KEEP = object()
HELD_OUT = ("eval_id", "eval_ood")


def arms(run):
    """{(block, arm): arm directory} for every arm with a run.json."""
    found = {(p.parent.parent.name, p.parent.name): p.parent for p in sorted(run.glob("b*/*/run.json"))}
    if not found:
        raise ValueError("%s holds no BLOCK/ARM/run.json: pass the run's --out directory" % run)
    return found


def adapter_hash(path):
    files = sorted(p for p in Path(path).rglob("*") if p.is_file())
    return digest({str(p.relative_to(path)): file_hash(p) for p in files})


def eval_config(config, batch_size, repetition_stop=KEEP):
    cfg = copy.deepcopy(config)
    cfg["generation"].update(temperature=0.0, top_p=1.0, top_k=None, candidates=1, batch_size=batch_size)
    if repetition_stop is not KEEP:
        cfg["generation"]["repetition_stop"] = repetition_stop
    return cfg


def decoding_protocol(generation):
    """The decoding every checkpoint is evaluated under; repetition_stop only when the run sets it, so an
    evaluation of a run from before that field keeps its protocol and resumes."""
    decoding = {k: generation[k] for k in ("temperature", "top_p", "top_k", "max_new_tokens", "max_sequence_length")}
    if generation.get("repetition_stop"):
        decoding["repetition_stop"] = generation["repetition_stop"]
    return decoding


def protocol_for(manifests, split, tasks, batch_size, repetition_stop=KEEP):
    """The settings every checkpoint of this run is evaluated under; all arms must share the base."""
    first = next(iter(manifests.values()))
    for key, m in manifests.items():
        same = all(m["config"][k] == first["config"][k] for k in BASE_KEYS)
        if not same or m["dataset_hash"] != first["dataset_hash"] or m["data_path"] != first["data_path"]:
            raise ValueError("Arm %s/%s has another base or dataset than the others" % key)
    cfg = eval_config(first["config"], batch_size, repetition_stop)
    return {"split": split, "questions": len(tasks), "question_ids": digest(sorted(t["id"] for t in tasks)),
            "data_path": first["data_path"], "dataset_hash": first["dataset_hash"],
            "task": first["task"], "target_signatures": list(targets_for(first["task"])),
            "base": {k: cfg[k] for k in BASE_KEYS},
            "decoding": decoding_protocol(cfg["generation"]),
            "answers_per_question": 1, "round_0": "base model, no adapter (shared adapter has LoRA B = 0)"}


def summarize(rows, tasks, targets):
    n = len(rows)
    errors = Counter(r["error"] for r in rows if not r["correct"])
    by_difficulty = defaultdict(list)
    for r in rows:
        by_difficulty[tasks[r["task_id"]]["difficulty"]].append(r["correct"])
    target = sum(errors[s] for s in targets)
    wrong = n - sum(r["correct"] for r in rows)
    return {"questions": n, "pass1": sum(r["correct"] for r in rows) / n,
            "pass1_by_difficulty": {d: sum(v) / len(v) for d, v in sorted(by_difficulty.items())},
            "questions_by_difficulty": {d: len(v) for d, v in sorted(by_difficulty.items())},
            "error_counts": dict(sorted(errors.items())),
            "target_error_rate": target / n, "target_share_of_errors": target / wrong if wrong else None,
            "format_error_rate": errors["format"] / n,
            "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / n,
            "truncated_answers": sum(bool(r.get("truncated")) for r in rows)}


def out_of_memory(error):
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def generate_all(model, cfg, tasks_list):
    """One greedy answer per question, longest prompt first, one batch per generate call; a batch that
    runs out of GPU memory is generated again at half the size, and the batches already done are kept."""
    ordered = sorted(tasks_list, key=lambda t: (-len(t["prompt"]), t["id"]))
    rows, start = [], 0
    while start < len(ordered):
        chunk = ordered[start:start + cfg["generation"]["batch_size"]]
        try:
            rows.extend(model.generate(chunk, 1, 0))
        except RuntimeError as e:
            if not out_of_memory(e) or cfg["generation"]["batch_size"] == 1:
                raise
            # cfg is the model's own config dict, so the smaller batch also holds for later checkpoints.
            cfg["generation"]["batch_size"] //= 2
            print("out of GPU memory: batch size now %d" % cfg["generation"]["batch_size"], flush=True)
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            continue
        start += len(chunk)
    return rows


def evaluate_checkpoint(cfg, adapter, pending, record, cost_records=None):
    """Load the checkpoint once and evaluate it on every pending split: [(split, tasks, targets, output)].
    With cost_records, each split's generation is one cost attempt; the first one includes the load.
    Every evaluation is owned by "evaluation" (the arm is in the record's block and detail)."""
    owner = "evaluation"
    checkpoint = "round_000" if record["block"] is None else "%s/%s/round_%03d" % (
        record["block"], record["arm"], record["round"])
    model, results = None, []
    try:
        for split, tasks_list, tasks, targets, output in pending:
            timing = (cost.stage(cost_records, "evaluate", owner, record["block"],
                                 {"checkpoint": checkpoint, "split": split, "includes_load": model is None})
                      if cost_records else contextlib.nullcontext())
            with timing as attempt:
                if model is None:
                    model = backend(cfg, adapter=str(adapter) if adapter else None)
                start = time.perf_counter()
                rows = generate_all(model, cfg, tasks_list)
                if attempt is not None:
                    attempt.generated(len(rows), sum(r["completion_tokens"] for r in rows))
            rows = [dict(r, difficulty=tasks[r["task_id"]]["difficulty"], **judge(tasks[r["task_id"]], r["response"]))
                    for r in rows]
            write_jsonl(output.with_suffix(".jsonl"), rows)
            result = dict(record, split=split, **summarize(rows, tasks, targets),
                          batch_size=cfg["generation"]["batch_size"], seconds=time.perf_counter() - start)
            write_json(output, result)
            print("%s %s pass1=%.4f target_error_rate=%.4f truncated=%d seconds=%.0f"
                  % (split, output.stem, result["pass1"], result["target_error_rate"], result["truncated_answers"],
                     result["seconds"]), flush=True)
            if result["truncated_answers"]:
                print("WARNING: %d answers of %s %s reached max_new_tokens=%d without a stop token; they are "
                      "judged as they stand" % (result["truncated_answers"], split, output.stem,
                                                cfg["generation"]["max_new_tokens"]), flush=True)
            results.append(result)
    finally:
        if model is not None:
            model.close()
    return results


def read_evaluation_config(path):
    """The evaluation settings (configs/matched_evaluation.json): {"splits": [...], "batch_size": N}, and
    optionally "repetition_stop" (null or {"span": S, "max_period": P}): the loop stop of the greedy
    answers, stated when it is not the run's own, e.g. a run whose round-1 pools were drawn before the
    field existed but is evaluated under the current rule."""
    config = read_json(path)
    if not {"splits", "batch_size"} <= set(config) <= {"splits", "batch_size", "repetition_stop"}:
        raise ValueError("%s must hold splits and batch_size, and may hold repetition_stop (has %s)"
                         % (path, ", ".join(sorted(config)) or "nothing"))
    stop = config.get("repetition_stop")
    if stop is not None and (not isinstance(stop, dict) or set(stop) != {"span", "max_period"}
                             or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in stop.values())
                             or stop["span"] < 2 * stop["max_period"]):
        raise ValueError('%s: repetition_stop must be null or {"span": S, "max_period": P}, S >= 2P' % path)
    splits = config["splits"]
    if (not isinstance(splits, list) or not splits or len(set(splits)) != len(splits)
            or any(s not in HELD_OUT + ("dev",) for s in splits)):
        raise ValueError("%s: splits must be a non-empty list of distinct names from %s"
                         % (path, ", ".join(HELD_OUT + ("dev",))))
    if type(config["batch_size"]) is not int or config["batch_size"] < 1:
        raise ValueError("%s: batch_size must be a positive integer" % path)
    return config


def evaluate_run(run, splits=None, batch_size=32, rounds=None, deterministic=True, repetition_stop=KEEP,
                 cost_records=None):
    run = Path(run).resolve()
    found = arms(run)
    manifests = {key: read_json(path / "run.json") for key, path in found.items()}
    first = next(iter(manifests.values()))
    data = Path(first["data_path"])
    dataset = verify_dataset(data)
    if digest(dataset) != first["dataset_hash"]:
        raise ValueError("The dataset at %s changed since the run" % data)
    if splits is None:
        # Both held-out splits the dataset has (an imported dataset may have no eval_ood).
        splits = [s for s in HELD_OUT if s + ".jsonl" in dataset["files"]]
    missing = [s for s in splits if s + ".jsonl" not in dataset["files"]]
    if missing:
        raise ValueError("The dataset has no %s split" % ", ".join(missing))
    per_split = []
    for split in splits:
        tasks_list = read_jsonl(data / (split + ".jsonl"))
        protocol = protocol_for(manifests, split, tasks_list, batch_size, repetition_stop)
        out = run / ("evaluation_greedy_" + split)
        if (out / "protocol.json").exists():
            old = read_json(out / "protocol.json")
            # The batch size is not part of the protocol: greedy answers do not depend on it beyond
            # numerical noise from padding, so a resumed evaluation may use another.
            if old != protocol:
                changed = sorted(k for k in set(old) | set(protocol) if old.get(k) != protocol.get(k))
                raise ValueError("%s was evaluated under other settings (%s); use a fresh run or remove it"
                                 % (out, ", ".join(changed)))
        else:
            write_json(out / "protocol.json", protocol)
        per_split.append((split, tasks_list, {t["id"]: t for t in tasks_list}, protocol["target_signatures"], out))
    if deterministic and first["config"]["backend"] == "hf":
        from rsi.determinism import enable_determinism
        enable_determinism()
    cfg = eval_config(first["config"], batch_size, repetition_stop)
    todo = [("round_000", None, {"block": None, "arm": None, "round": 0, "adapter": None, "adapter_hash": None})]
    for (block, arm), path in sorted(found.items()):
        for t in range(1, manifests[(block, arm)]["config"]["rounds"] + 1):
            if rounds is not None and t not in rounds:
                continue
            done = path / ("round_%03d" % t) / "complete.json"
            if not done.exists():
                continue
            if read_json(done)["adapter"] is None:
                # A one-step run's null arm (run_matched_experiment.py) takes no step and saves no
                # adapter: it is the shared adapter, which computes the base's function -- round 0.
                continue
            adapter = path / read_json(done)["adapter"]
            todo.append(("%s_%s_round_%03d" % (block, arm, t), adapter,
                         {"block": block, "arm": arm, "round": t, "adapter": str(adapter.relative_to(run)),
                          "adapter_hash": adapter_hash(adapter)}))
    results = []
    for name, adapter, record in todo:
        pending = []
        for split, tasks_list, tasks, targets, out in per_split:
            output = out / (name + ".json")
            if output.exists():
                old = read_json(output)
                if old["adapter_hash"] != record["adapter_hash"]:
                    raise ValueError("%s was evaluated on another adapter than %s holds now"
                                     % (output, record["adapter"]))
                results.append(old)
            else:
                pending.append((split, tasks_list, tasks, targets, output))
        if pending:
            results.extend(evaluate_checkpoint(cfg, adapter, pending, record, cost_records))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the --out directory of run_iterative_experiment.py")
    p.add_argument("--config", required=True,
                   help="evaluation settings: splits and batch size, configs/matched_evaluation.json")
    p.add_argument("--rounds", nargs="+", type=int, help="only these training rounds (round 0 always)")
    p.add_argument("--cost-records", help="write a cost record per checkpoint-split attempt to this directory")
    a = p.parse_args(argv)
    try:
        evaluation = read_evaluation_config(a.config)
        results = evaluate_run(a.run, evaluation["splits"], evaluation["batch_size"], a.rounds,
                               repetition_stop=evaluation.get("repetition_stop", KEEP), cost_records=a.cost_records)
    except (ValueError, FileNotFoundError) as e:
        print("refused: %s" % e, file=sys.stderr)
        return 1
    print("evaluated %d checkpoint-splits of %s" % (len(results), a.run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
