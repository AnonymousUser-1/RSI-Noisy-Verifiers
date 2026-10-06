#!/usr/bin/env python3
"""Prepare matched arithmetic R/S training data from saved model candidates.

R samples errors uniformly in each shared (prompt, supervised-token-count)
bucket.  S selects ignore_parentheses errors from that same bucket.  Both keep
the same correct candidate IDs, one response per prompt, C:E=3:1 and weight 1:
the frozen defaults of matching.error_fraction and matching.token_tolerance, since
this command reads no experiment config.
The common K follows the fixed 64 -> 32 -> 16 ladder across every input block.

This is an intervention/data-preparation command, not a generation or training
command.  Inputs are <block>.jsonl pools plus <block>.jsonl.meta.json from
sample_candidates.py.  HF mode loads only the pinned tokenizer, never weights,
and recounts the exact training response plus EOS; completion_tokens is not
substituted for that count.  Mock mode is an explicitly DEMO_ONLY plumbing test.
No reference answers, held-out prompts, gradients or test results select rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from match_candidates import report_infeasible, write_matching
from rsi.common import digest, file_hash, source_hash, verify_dataset, write_json, write_jsonl
from rsi.matching import (ARITHMETIC_TARGET_SIGNATURES, K_LADDER, MatchingFailure,
                          candidate_from_row, select_k_for_blocks)
from rsi.reference_encode import assert_supervised_last_token, encode_example
from rsi.split_selection import PHASES, check_selection, dataset_binding, read_split
from rsi.tasks import judge


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Arithmetic dataset root with manifest.json")
    p.add_argument("--pools", required=True, help="Directory containing <block>.jsonl and their .meta.json files")
    p.add_argument("--split", required=True, help="dev for a pilot, train_NNN for a main intervention")
    p.add_argument("--study-id", required=True)
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--out", required=True, help="New or empty output directory; existing results are never replaced")
    p.add_argument("--seed", type=int, default=0, help="Matching seed; never changed to force feasibility")
    p.add_argument("--backend", choices=("hf", "mock"), default="hf")
    return p


def load_pool(path, tasks, split, data, args):
    """Check pool identity, byte binding and complete per-prompt sampling."""
    meta_path = Path(str(path) + ".meta.json")
    meta_bytes, payload = meta_path.read_bytes(), path.read_bytes()
    meta = json.loads(meta_bytes)
    if meta.get("hash") != hashlib.sha256(payload).hexdigest():
        raise ValueError("Pool hash mismatch: %s" % path)
    if meta.get("backend") != args.backend:
        raise ValueError("Pool backend mismatch: %s (mock pools cannot enter HF runs)" % path)
    if any(meta.get("split", {}).get(k) != split[k] for k in ("name", "sha256", "rows")):
        raise ValueError("Pool split binding mismatch: %s" % path)
    if any(meta.get("data", {}).get(k) != data[k] for k in ("manifest_sha256", "dataset_hash", "task")):
        raise ValueError("Pool dataset binding mismatch: %s" % path)
    if any(meta.get("identity", {}).get(k) != getattr(args, k) for k in ("study_id", "phase")):
        raise ValueError("Pool study/phase mismatch: %s" % path)
    config = meta.get("config", {})
    if meta.get("config_hash") != digest(config):
        raise ValueError("Pool configuration hash mismatch: %s" % path)
    for key in ("backend", "model", "revision", "generation"):
        if key not in meta or config.get(key) != meta[key]:
            raise ValueError("Pool configuration disagrees with %s: %s" % (key, path))
    if meta.get("generation_hash") != digest(meta["generation"]):
        raise ValueError("Pool generation hash mismatch: %s" % path)
    if args.backend == "hf" and not re.fullmatch(r"[0-9a-fA-F]{40}", str(meta["revision"])):
        raise ValueError("HF pool revision must be an immutable 40-character commit: %s" % path)
    n = meta["generation"].get("candidates")
    if type(n) is not int or n < 1:
        raise ValueError("Pool must record a positive number of candidates per prompt: %s" % path)

    # Parse the bytes whose hashes were checked, not a second read that might
    # see a concurrently replaced pool.  The meta hash binds the same record.
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    seen_ids, seen_samples = set(), set()
    for row in rows:
        tid, sample, cid = row.get("task_id"), row.get("sample"), row.get("id")
        if tid not in tasks or type(sample) is not int or not 0 <= sample < n:
            raise ValueError("Unknown task or invalid sample index in %s" % path)
        if not isinstance(row.get("response"), str):
            raise ValueError("Pool rows need a model-generated response, not reference_answer: %s" % path)
        if args.backend == "hf" and not isinstance(row.get("truncated"), bool):
            raise ValueError("Pool rows must record whether max_new_tokens cut them (a pool from before that "
                             "field); draw the pool again: %s" % path)
        if cid != digest([tid, sample]) or cid in seen_ids or (tid, sample) in seen_samples:
            raise ValueError("Invalid or duplicate candidate ID/sample in %s" % path)
        seen_ids.add(cid)
        seen_samples.add((tid, sample))
    if len(rows) != len(tasks) * n:
        raise ValueError("Incomplete candidate pool: %s (expected %d rows, found %d)" %
                         (path, len(tasks) * n, len(rows)))
    return rows, meta, {"file": str(path.resolve()), "sha256": meta["hash"], "rows": len(rows),
                        "meta_sha256": hashlib.sha256(meta_bytes).hexdigest(), "generation_seed": meta.get("seed")}


def load_encoder(meta, backend):
    """Use the trainer's encode without loading a language model."""
    if backend == "mock":
        class MockTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return list(range(1, len(text.split()) + 1))

        tokenizer = MockTokenizer()
        return SimpleNamespace(tokenizer=tokenizer, prompt_ids=lambda prompt: tokenizer.encode(prompt)), {
            "method": "DEMO_ONLY whitespace response count plus synthetic EOS; not HF tokenization",
            "eos_token_id": 0, "DEMO_ONLY": True}
    from transformers import AutoTokenizer

    from rsi.backends import chat_prompt_ids

    tokenizer = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"])
    if type(tokenizer.eos_token_id) is not int or not tokenizer.chat_template:
        raise ValueError("The pinned tokenizer must have EOS and the trainer's chat template")

    def prompt_ids(prompt):
        return chat_prompt_ids(tokenizer, prompt)

    return SimpleNamespace(tokenizer=tokenizer, prompt_ids=prompt_ids), {
        "method": "rsi.reference_encode.encode_example: response tokens plus real EOS; prompt/padding excluded",
        "model": meta["model"], "revision": meta["revision"], "eos_token_id": tokenizer.eos_token_id,
        "DEMO_ONLY": False}


def main(argv=None):
    args = parser().parse_args(argv)
    # Apply the pilot/main split rule even to mock runs of this new entry.
    check_selection(args.split, args.study_id, args.phase, hf=True)
    out, pools = Path(args.out), Path(args.pools)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be new or empty: %s" % out)
    if not pools.is_dir():
        raise ValueError("--pools must be a directory of <block>.jsonl files")
    paths = sorted(pools.glob("*.jsonl"))
    if not paths:
        raise ValueError("No candidate pool files found in %s" % pools)
    manifest = verify_dataset(args.data)
    if manifest.get("task") != "arithmetic":
        raise ValueError("This intervention requires an arithmetic dataset")
    task_rows, split = read_split(args.data, manifest, args.split)
    if len(task_rows) != manifest["counts"].get(args.split) or not task_rows:
        raise ValueError("Dataset split row count does not match its manifest")
    tasks = {t["id"]: t for t in task_rows}
    if len(tasks) != len(task_rows) or any(t.get("task") != "arithmetic" or
                                           t.get("split") != args.split for t in task_rows):
        raise ValueError("Task IDs must be unique and all rows must belong to the selected arithmetic split")
    data = dataset_binding(args.data, manifest)
    loaded = {p.stem: load_pool(p, tasks, split, data, args) for p in paths}
    first = loaded[paths[0].stem][1]
    common = {k: first[k] for k in ("backend", "model", "revision", "generation")}
    for block, (_, meta, _) in loaded.items():
        if any(meta[k] != v for k, v in common.items()):
            raise ValueError("Blocks must share the starting model/revision and generation settings: %s" % block)
    encoder, tokenization = load_encoder(first, args.backend)
    blocks, by_id, token_counts, signatures = {}, {}, {}, {}
    for block, (rows, meta, _) in loaded.items():
        candidates, counts, errors = [], {}, {}
        # Candidate IDs repeat across seeds.  Every lookup stays block-local.
        for row in rows:
            task = tasks[row["task_id"]]
            ids, labels, _ = encode_example(encoder, task["prompt"], row["response"])
            assert_supervised_last_token(labels, encoder.tokenizer.eos_token_id, row["id"])
            if args.backend == "hf" and len(ids) > meta["generation"]["max_sequence_length"]:
                raise ValueError("Training sequence too long; refusing truncation: %s/%s" % (block, row["id"]))
            count = sum(label != -100 for label in labels)
            verdict = judge(task, row["response"])
            candidates.append(candidate_from_row(row, task, verdict, count))
            counts[row["id"]] = count
            errors[verdict["error"]] = errors.get(verdict["error"], 0) + 1
        blocks[block], by_id[block], token_counts[block], signatures[block] = (
            candidates, {r["id"]: r for r in rows}, counts, errors)
    result = select_k_for_blocks(blocks, seed=args.seed, target_signatures=ARITHMETIC_TARGET_SIGNATURES)
    metadata = {"task": "arithmetic", "target_signatures": list(ARITHMETIC_TARGET_SIGNATURES),
                "split": split, "study_id": args.study_id, "phase": args.phase,
                "backend": args.backend, "DEMO_ONLY": args.backend == "mock"}
    write_matching(out, blocks, list(K_LADDER), result, metadata=metadata)
    output_files = {}
    if result["feasible"]:
        for block, matched in sorted(result["per_block"].items()):
            for arm in ("R", "S"):
                selected = sorted(matched[arm], key=lambda cid: (by_id[block][cid]["task_id"], cid))
                # Only the original candidate response enters SFT.  No oracle
                # answer, correctness label or error taxonomy is exported here.
                training = [{"id": cid, "task_id": by_id[block][cid]["task_id"], "block_id": block,
                             "prompt": tasks[by_id[block][cid]["task_id"]]["prompt"],
                             "response": by_id[block][cid]["response"], "weight": 1.0,
                             "tokens": token_counts[block][cid]} for cid in selected]
                path = out / block / arm / "training.jsonl"
                write_jsonl(path, training)
                output_files[str(path.relative_to(out))] = {"sha256": file_hash(path), "rows": len(training)}
    report = dict(metadata, schema=1, feasible=result["feasible"], K=result["K"], seed=args.seed,
                  data=data, pools={b: entry[2] for b, entry in loaded.items()},
                  tokenization=tokenization, generation=common, original_error_counts=signatures,
                  code={"source_hash": source_hash()},
                  command={"executable": sys.executable, "args": sys.argv[1:] if argv is None else list(argv)},
                  selection_policies={"R": "uniform within the shared task/token bucket; coincidences retained",
                                      "S": "prefer ignore_parentheses within that same bucket"},
                  outputs=output_files, trained=False,
                  scope="Separate arithmetic extension; does not replace the graph main line or run optimizer updates")
    write_json(out / "intervention.json", report)
    if not result["feasible"]:
        report_infeasible(result)
        return 2
    for block, matched in sorted(result["per_block"].items()):
        audit = matched["audit"]
        print("block=%s K=%d C=%d E=%d R_target=%d S_target=%d DEMO_ONLY=%s" %
              (block, result["K"], audit["C_R"], audit["E_R"], audit["R_target_hits"],
               audit["S_target_hits"], metadata["DEMO_ONLY"]))
    print("wrote", out / "intervention.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, ImportError, MatchingFailure) as exc:
        print("Arithmetic intervention refused: %s" % exc, file=sys.stderr)
        sys.exit(1)
