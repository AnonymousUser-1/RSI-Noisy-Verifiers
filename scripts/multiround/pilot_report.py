#!/usr/bin/env python3
"""Pilot check of one model on a task (graph, arithmetic, gsm8k or dmmath), before any main-phase pool is drawn.

    python scripts/multiround/pilot_report.py --config CONFIG --data DATA \\
        --pool POOL.jsonl --split dev --out REPORT.json

Reads a pool drawn by sample_candidates.py (--phase pilot --split dev) and
reports what decides whether the main run can go ahead on this model:

  * format errors (answers the grader cannot parse) -- should be a few percent;
  * accuracy -- the study wants roughly 30-70% (arXiv 2602.10014: only a
    moderate starting accuracy improves);
  * distinct answers per prompt -- diversity the matching needs;
  * J_E-eligible prompts (an S-target error -- rsi.matching.targets_for:
    nonshortest for graph, ignore_parentheses for arithmetic, intermediate for
    gsm8k, sign for dmmath -- and a non-target error within matching.token_tolerance
    supervised tokens of it), the binding constraint of the round-1 matching,
    and the rate per prompt, extrapolated to a round-1 pool (train_001, or its first
    generation.prompts_per_pool prompts); K needs K x matching.error_fraction per
    block (at 3:1: K = 64 needs 16, K = 32 needs 8, K = 16 needs 4);
  * the K ladder on this pool itself.

Token counts are the supervised count the arms train on
(rsi.reference_encode.supervised_token_count with the base's tokenizer).
Pilot-only: it reads the dev split and nothing it reports selects main data.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rsi.common import digest, file_hash, load_config, read_json, read_jsonl, require_explicit, write_json  # noqa: E402
from rsi.matching import _views, candidate_from_row, quotas, select_k_for_blocks, targets_for  # noqa: E402
from rsi.reference_encode import supervised_token_count  # noqa: E402
from rsi.tasks import judge  # noqa: E402

def check_pool(pool, data, split, config):
    """Refuse a pool drawn for another pilot: other data, split, model or sampling settings.

    Task ids leave the prompt out, so a pool kept from before a prompt change would
    otherwise be judged as if it answered the new prompt.
    """
    meta_path = Path(pool + ".meta.json")
    if not meta_path.is_file():
        raise SystemExit("%s has no %s; pools come from sample_candidates.py with their meta" % (pool, meta_path.name))
    meta = read_json(meta_path)
    stale = [name for name, ok in (
        ("pool bytes", meta.get("hash") == file_hash(pool)),
        ("data (a prompt or the questions changed since)",
         (meta.get("data") or {}).get("dataset_hash") == digest(read_json(Path(data) / "manifest.json"))),
        ("split", (meta.get("split") or {}).get("name") == split),
        ("model or revision", (meta.get("model"), meta.get("revision")) == (config["model"], config["revision"])),
        ("sampling settings", meta.get("generation") == config["generation"])) if not ok]
    if stale:
        raise SystemExit("%s was drawn for another pilot: its %s differ.  Delete $WORK/pilot and rerun "
                         "03_pilot_check.sh." % (pool, ", ".join(stale)))


def main(args):
    config = load_config(args.config)
    try:
        require_explicit(args.config, "pilot")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    check_pool(args.pool, args.data, args.split, config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    task_kind = read_json(Path(args.data) / "manifest.json")["task"]
    targets = targets_for(task_kind)
    tasks = {t["id"]: t for t in read_jsonl(Path(args.data) / (args.split + ".jsonl"))}
    pool = read_jsonl(args.pool)
    # The size of a round-1 pool: train_001, or its first generation.prompts_per_pool prompts.
    round1 = len(read_jsonl(Path(args.data) / "train_001.jsonl"))
    if config["generation"]["prompts_per_pool"] is not None:
        round1 = min(round1, config["generation"]["prompts_per_pool"])
    judged = {r["id"]: judge(tasks[r["task_id"]], r["response"]) for r in pool}
    candidates = [candidate_from_row(r, tasks[r["task_id"]], judged[r["id"]],
                                     supervised_token_count(tokenizer, r["response"])) for r in pool]
    matching = config["matching"]
    views = _views(candidates, targets, matching["token_tolerance"])
    prompts = len({r["task_id"] for r in pool})
    eligible = sum(v.eligible for v in views.values())
    signatures = Counter(j["error"] for j in judged.values())
    distinct = len({(r["task_id"], r["response"]) for r in pool})
    ladder = select_k_for_blocks({"pilot": candidates}, seed=0, target_signatures=targets, **matching)
    e64 = quotas(64, matching["error_fraction"])[1]   # the wrong answers K = 64 needs per block
    truncated = sum(bool(r.get("truncated")) for r in pool)
    lengths = sorted(r["completion_tokens"] for r in pool)
    report = {
        "PILOT_ONLY": True, "model": config["model"], "revision": config["revision"], "split": args.split,
        "task": task_kind, "target_signatures": list(targets),
        "prompts": prompts, "answers": len(pool), "samples_per_prompt": len(pool) / prompts,
        "accuracy": signatures["correct"] / len(pool),
        "format_error_rate": signatures["format"] / len(pool),
        "signatures": dict(signatures),
        # Cut at max_new_tokens without a stop id: judged here, but never matched or trained on.
        "truncated_answers": truncated,
        "completion_tokens": {"mean": sum(lengths) / len(lengths), "p99": lengths[int(0.99 * (len(lengths) - 1))],
                              "max": lengths[-1], "max_new_tokens": config["generation"]["max_new_tokens"]},
        "distinct_answers_per_prompt": distinct / prompts,
        "prompts_with_correct": sum(v.has_correct for v in views.values()),
        "prompts_with_target": sum(bool(v.target_ids) for v in views.values()),
        "error_fraction": matching["error_fraction"], "token_tolerance": matching["token_tolerance"],
        "prompts_eligible_J_E": eligible,
        "round1_pool_prompts": round1,
        "eligible_estimate_at_round1_pool": eligible / prompts * round1,
        "K_on_this_pool": ladder["K"],
        "go": {"no_truncated_answers": truncated == 0,
               "format_ok": signatures["format"] / len(pool) <= 0.05,
               "accuracy_in_30_70": 0.30 <= signatures["correct"] / len(pool) <= 0.70,
               "K64_expected_at_round1_pool": eligible / prompts * round1 >= e64,
               "K64_with_25pct_margin_at_round1_pool": eligible / prompts * round1 >= 1.25 * e64},
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True,
                   help="Dataset root (graph, arithmetic, gsm8k or dmmath); its manifest names the task")
    p.add_argument("--pool", required=True, help="Pool JSONL from sample_candidates.py (--phase pilot)")
    p.add_argument("--split", default="dev")
    p.add_argument("--out", required=True)
    main(p.parse_args())
