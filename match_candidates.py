#!/usr/bin/env python3
"""Joint matching CLI: build R/S subsets and write the ladder certificate.

Wraps `rsi.matching.select_k_for_blocks` so the joint construction and the
ladder/infeasibility report are reachable as a command rather than only as a
library call.

Input is a directory of per-block candidate pools: one JSONL per block, named
`<block>.jsonl`, each row carrying the candidate fields a pool row already has.
The blocks move together -- a ladder size is accepted only if *every* block can
meet the hard constraints at that size -- so a partial run is impossible by
construction.

Infeasibility is a *result*, not a crash: when no ladder size (64 -> 32 -> 16)
works, this writes the per-block certificates and exits non-zero with a clear
message.  Nothing is relaxed.
"""
import argparse
import json
import sys
from pathlib import Path

from rsi.common import load_config, read_json, read_jsonl, write_json
from rsi.matching import (DEFAULT_ERROR_FRACTION, DEFAULT_TOKEN_TOLERANCE, K_LADDER, MatchingFailure,
                          candidate_from_row, select_k_for_blocks, targets_for)
from rsi.tasks import judge


def load_block(block_path, tasks, token_counts):
    """Turn one pool JSONL into `Candidate` objects, judging each response.

    `token_counts` is supplied by the caller because supervised token count is
    the match variable; guessing it here would silently substitute a different
    quantity for the one the contract freezes.
    """
    rows = read_jsonl(block_path)
    meta_path = Path(str(block_path) + ".meta.json")
    hf = meta_path.is_file() and read_json(meta_path).get("backend") == "hf"
    candidates = []
    for row in rows:
        if hf and not isinstance(row.get("truncated"), bool):
            raise ValueError("%s: row %s does not record whether max_new_tokens cut it (a pool from before that "
                             "field); draw the pool again" % (block_path, row["id"]))
        task = tasks[row["task_id"]]
        verdict = judge(task, row["response"])
        count = token_counts.get(row["id"])
        if count is None:
            raise ValueError("Pool row %s has no supervised token count" % row["id"])
        candidates.append(candidate_from_row(row, task, verdict, count))
    return candidates


def load_token_counts(path):
    """`id<TAB>tokens` per line, from the tokeniser pass that produced the pool."""
    counts = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            candidate_id, count = line.rstrip("\n").split("\t")
            counts[candidate_id] = int(count)
    return counts


def write_matching(out, blocks, ladder, result, metadata=None):
    """Write `ladder.json`, and `matched_subsets.json` when the ladder is feasible.

    run_matched_experiment.py writes its matching through this too, so the two
    commands cannot drift into different file shapes.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ladder_record = {"blocks": sorted(blocks), "ladder": ladder,
                     "error_fraction": result.get("error_fraction"), "token_tolerance": result.get("token_tolerance"),
                     "attempts": result["attempts"],
                     "feasible": result["feasible"], "K": result["K"],
                     "reason": result.get("reason")}
    matched_record = {
        "K": result["K"], "seed": result["seed"],
        "error_fraction": result.get("error_fraction"), "token_tolerance": result.get("token_tolerance"),
        "per_block": {name: {"R": r["R"], "S": r["S"], "audit": r["audit"],
                             "certificate": r["certificate"], "hash": r["hash"]}
                      for name, r in result["per_block"].items()},
        "ladder": ladder}
    if metadata is not None:
        # A separate intervention may label its task, targets and mock status;
        # it may never replace the actual matching result.  Existing callers
        # with no metadata retain their original file shape.
        if set(metadata) & (set(ladder_record) | set(matched_record)):
            raise ValueError("Matching metadata cannot replace matching result fields")
        ladder_record.update(metadata)
        matched_record.update(metadata)
    write_json(out / "ladder.json", ladder_record)
    if result["feasible"]:
        write_json(out / "matched_subsets.json", matched_record)


def report_infeasible(result):
    print("INFEASIBLE: no ladder size is feasible for every block.", file=sys.stderr)
    for attempt in result["attempts"]:
        print("  K=%d feasible=%s" % (attempt["K"], attempt["feasible"]), file=sys.stderr)
    print("Stop before training. Do not change the seed, the length tolerance, or the ratio.",
          file=sys.stderr)


def main(args):
    pools = Path(args.pools)
    if not pools.is_dir():
        raise SystemExit("--pools must be a directory of <block>.jsonl files")
    block_files = sorted(pools.glob("*.jsonl"))
    if not block_files:
        raise SystemExit("No <block>.jsonl pools found in %s" % pools)
    tasks = {t["id"]: t for t in read_jsonl(Path(args.data) / ("train_%03d.jsonl" % args.round))}
    kinds = sorted({t["task"] for t in tasks.values()})
    if len(kinds) != 1:
        raise SystemExit("train_%03d.jsonl holds the task kinds %s; one pool matches one kind" % (args.round, kinds))
    try:
        targets = targets_for(kinds[0])  # S's target error is the task's
    except ValueError as error:
        raise SystemExit(str(error))
    token_counts = load_token_counts(args.tokens)
    # The C:E ratio and the token tolerance: the config's (its `matching` section), else the defaults.
    matching = (load_config(args.config)["matching"] if args.config else
                {"error_fraction": DEFAULT_ERROR_FRACTION, "token_tolerance": DEFAULT_TOKEN_TOLERANCE})

    blocks = {path.stem: load_block(path, tasks, token_counts) for path in block_files}
    ladder = [int(k) for k in args.ladder.split(",")] if args.ladder else list(K_LADDER)
    try:
        result = select_k_for_blocks(blocks, seed=args.seed, k_ladder=ladder, target_signatures=targets, **matching)
    except MatchingFailure as failure:
        # A malformed request is a programming error; an infeasible *pool* is
        # returned by select_k_for_blocks and handled below.  Keep them apart.
        raise SystemExit("Matching request invalid: %s" % failure)

    out = Path(args.out)
    write_matching(out, blocks, ladder, result, metadata={"task": kinds[0], "target_signatures": list(targets)})
    if not result["feasible"]:
        report_infeasible(result)
        return 2

    print("S target error: %s" % ", ".join(targets))
    for name, r in sorted(result["per_block"].items()):
        audit = r["audit"]
        print("block=%s K=%d C=%d E=%d same_tasks=%s same_tokens=%s yield=%.4f"
              % (name, audit["K_R"], audit["C_R"], audit["E_R"], audit["same_task_ids"],
                 audit["task_token_counts_within_tolerance"], audit["yield"] or 0.0))
    print("wrote", out / "matched_subsets.json")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pools", required=True, help="Directory of <block>.jsonl candidate pools")
    p.add_argument("--data", required=True)
    p.add_argument("--round", type=int, default=1, help="Training round whose tasks these pools use")
    p.add_argument("--tokens", required=True, help="TSV of candidate id -> supervised token count")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ladder", default="", help="Override the K ladder, e.g. '64,32,16'")
    p.add_argument("--config", help="Take matching.error_fraction and matching.token_tolerance from this config "
                                    "(default: 0.25, i.e. C:E = 3:1, and 0)")
    a = p.parse_args()
    sys.exit(main(a))
