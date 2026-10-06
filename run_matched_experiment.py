#!/usr/bin/env python3
"""Matched one-step experiment: the R, S and null arms for every seed block.

    python run_matched_experiment.py --config CONFIG --data DATA \\
        --shared-pool POOLS --out OUT [--seed 0] \\
        --study-id ID --phase {pilot,main} \\
        --shared-adapter ADAPTER --reference-gradient H_DIR       (hf)
    python run_matched_experiment.py ... --backend mock           (plumbing only)

The single entry for the matched experiment's main path.  The
legacy multi-round loop (rsi.experiment.run, run_job.py) is not used here.

Input
  --shared-pool  a directory of block pools <block>.jsonl, each written by
                 sample_candidates.py with its <block>.jsonl.meta.json.  The
                 block id is the file name without .jsonl.  Each pool is read
                 once and checked against the sha256 its meta records before
                 anything is matched or trained.
  --study-id, --phase
                 RUN_COST_SPEC 1 identity, never guessed.  On mock either may
                 be left out: run.json then records it as null with the
                 reason, and run_id as null.
  --shared-adapter, --reference-gradient
                 hf only, both required: the shared_adapter/ every arm starts
                 from (make_shared_adapter.py) and the directory
                 compute_reference_gradient.py wrote at that adapter (h.pt,
                 h.json).  Before anything is written, h.pt must still hash to
                 what h.json records, and h.json must be bound to this
                 adapter's parameter hash, this base, this resolved config and
                 this dataset; the adapter must record this base.

Steps
  0. hf: the length every arm trains on is counted for every pool row with
     the base's tokenizer (rsi.reference_encode.supervised_token_count:
     response tokens + EOS).  The pool's completion_tokens is a generation-time
     count and does not always equal it, and the match tolerance is the config's
     matching.token_tolerance (0 by default).
  1. Joint matching over all blocks (rsi.matching.select_k_for_blocks): the
     largest K in 64 -> 32 -> 16 that every block meets at once.  If none
     does, write matching/ladder.json and experiment.json, print the ladder
     and exit 2 -- a result, not a crash.  Nothing is trained or relaxed.
     S's target error is the dataset task's (rsi.matching.targets_for:
     graph nonshortest, arithmetic ignore_parentheses, gsm8k intermediate, dmmath sign).
  2. Per block, the three one_step.py arms in the order R, S, null.  null
     reads R's rows but takes no optimizer step.
  3. Optional audit section in CONFIG: R and S each spend up to audit.budget
     correctness queries before training. Pre-audit matching is preserved;
     deletion/reweighting can change post-audit counts, rates and weights.
     The frozen null is never audited. Policy none / budget 0 is unchanged.

Each block is matched and trained from its own pool only.  Candidate ids are
digest([task_id, sample]) and repeat across blocks, so every structure that
spans blocks is keyed by (block_id, candidate_id).

Output (OUT must not exist or must be empty; this entry does not resume)
  experiment.json                      status, exit code, K, seeds, code identity
  matching/ladder.json                 every ladder attempt, per-block certificates
  matching/matched_subsets.json        R and S ids per block (feasible runs only)
  source_snapshot/                     the source files, when they are not HEAD's
  <block>/{R,S,null}/run.json          identity and version bindings (RUN_COST_SPEC 1)
  <block>/{R,S,null}/round_001/        training.jsonl, arm.json, complete.json;
                                       hf adds diagnostics.json, and adapter/ for R and S
  <block>/{R,S,null}/finished.json
  <block>/diagnostics/gradients.jsonl  one line per arm of the block that has run
  cost_records/<cost_id>.json          one per arm attempt, failed ones included (rsi/cost.py)
Paths recorded under OUT are POSIX and relative to OUT, except complete.json's
adapter, which stays relative to its arm directory as evaluate.py reads it.
Input paths are recorded absolute.

Exit status
  0  every arm of every block ran
  2  matching infeasible for the declared ladder (argparse also exits 2 on a
     usage error; experiment.json tells the two apart)
  1  refused with a message before anything was written (bad input, OUT not
     empty, an hf input that does not belong to this run), or failed after
     writing began, in which case experiment.json records status "failed" and
     the reason

Block ids
  RUN_COST_SPEC 1 names the seed blocks b00, b01 and b02; the one-step
  comparison (experiments/one_step.sh) runs b00 to b04, the five of the
  multi-round run whose pools it uses.  hf requires exactly one of those two
  sets of pools.
  mock records the names it is given, as long as they are letters, digits,
  '_' and single '-', and not matching or source_snapshot.

Backends
  Without --backend the config's backend applies, which is hf.  hf checks the
  base against configs/pins/base_pin.json (or the pin file RSI_BASE_PIN
  names, rsi/base_pin.py) and runs every arm on the GPU
  with deterministic kernels (rsi/determinism.py), recording
  diagnostics.json per arm; an arm that writes none fails the run.  The mock
  backend has no parameters, so every gradient diagnostic is recorded as
  skipped with its reason, never as a number.  Mock output is DEMO_ONLY:
  plumbing evidence, not a result.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

import one_step
from rsi.budgeted_auditing import apply_audit, enabled as auditing_enabled
from match_candidates import report_infeasible, write_matching
from rsi import cost
from rsi.common import (digest, drawn_generation, environment, file_hash, load_config, read_json, read_jsonl, require_explicit, seed_for,
                        verify_dataset, write_json, write_jsonl)
from rsi.base_pin import check_base_pin
from rsi.experiment import pin_config
from rsi.matching import (DEFAULT_ERROR_FRACTION, DEFAULT_TOKEN_TOLERANCE, K_LADDER, MatchingFailure,
                          candidate_from_row, pool_key, quotas, select_k_for_blocks,
                          targets_for)
from rsi.provenance import git_state, snapshot_source
from rsi.reference_encode import supervised_token_count
from rsi.shared_adapter import lora_mismatch, read_shared_record
from rsi.tasks import judge

SCHEMA_VERSION = "rsi-run-cost/1"
# RUN_COST_SPEC 1: the seed blocks.  This entry runs three seed blocks by default; the one-step experiments
# use five (one per seed, matching the multi-round run they share pools with).
BLOCKS = ("b00", "b01", "b02")
HF_BLOCK_SETS = (BLOCKS, ("b00", "b01", "b02", "b03", "b04"))
PHASES = ("pilot", "main")
ROUND = "round_001"
# A block id becomes a directory under OUT and a part of run_id, so it must be a
# safe file name, must not contain run_id's separator, and must not be a name
# this entry writes at the top of OUT.
BLOCK_NAME = re.compile(r"[A-Za-z0-9_]+(-[A-Za-z0-9_]+)*")
RESERVED = ("matching", "source_snapshot")
BLOCK_RULE = ("the pool's file name without .jsonl; hf requires exactly %s, mock records the names it is "
              "given (letters, digits, '_' and single '-'; not %s)"
              % (" or ".join(", ".join(s) for s in HF_BLOCK_SETS), " or ".join(RESERVED)))
TRAINING_FIELDS = ("id", "task_id", "prompt", "response", "weight", "tokens")
# The generation settings that define how answers are drawn and where they are cut, as opposed to a
# pool's size (candidates, prompts_per_pool) and its batching.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_new_tokens", "max_sequence_length", "repetition_stop")

MOCK_TOKEN_SOURCE = ("mock: completion_tokens from the pool rows, MockBackend's whitespace word count "
                     "len(response.split()); not tokenizer tokens, no EOS")
HF_TOKEN_SOURCE = ("hf: rsi.reference_encode.supervised_token_count with the base's tokenizer, i.e. "
                   "encode(response) + EOS, the tokens each arm supervises; not the pool's generation-time "
                   "completion_tokens, which differ when decoding does not round-trip")
FINGERPRINT_REASON = ("bound as written and not recomputed: pool_fingerprint needs the pool's generation "
                      "seed, which the meta does not record, and it hashes the source of the code that "
                      "sampled the pool; the pool's sha256 is verified instead")
NOT_USED_ON_MOCK = "the mock backend has no parameters; only hf reads it"
COST_RECORDS_MOCK = {"status": "not_recorded", "reason": "the mock backend runs no stage worth timing"}
COST_RECORDS = {"status": "recorded", "path": "cost_records",
                "scope": "one record per arm attempt (rsi/cost.py): monotonic wall seconds with GPU work settled "
                         "at both ends, allocated GPU hours, memory peaks; this entry generates no responses. "
                         "The round-1 pools are inputs bound by sha256: their sampling cost belongs to the run "
                         "that drew them and is not counted here"}
PATHS = ("paths under --out are POSIX and relative to --out, except complete.json's adapter, which is "
         "relative to its arm directory as in rsi.experiment.run; input paths are absolute")


def identity(study_id, phase):
    """Check study_id and phase, and give the reason for each one not given.

    RUN_COST_SPEC 1: never guessed, so a missing one is recorded as null with
    its reason, and run_id with it.
    """
    if study_id is not None and (not study_id.strip() or study_id != study_id.strip()):
        raise SystemExit("--study-id %r is empty or has surrounding whitespace.  Nothing was written." % study_id)
    if phase is not None and phase not in PHASES:
        raise SystemExit("--phase must be one of %s, not %r.  Nothing was written." % (", ".join(PHASES), phase))
    missing = {}
    if study_id is None:
        missing["study_id"] = "not given (--study-id); RUN_COST_SPEC 1 forbids guessing it from a directory name"
        missing["run_id"] = "{study_id}--{block_id}--{branch} cannot be formed without study_id"
    if phase is None:
        missing["phase"] = "not given (--phase); pilot or main is not guessed"
    return missing


def run_id(study_id, block, branch):
    """RUN_COST_SPEC 1: {study_id}--{block_id}--{branch}, or None without a study_id."""
    return None if study_id is None else "%s--%s--%s" % (study_id, block, branch)


def find_blocks(pool_dir, backend):
    """The block pools in `pool_dir`, keyed by block id, and how the ids were found."""
    pool_dir = Path(pool_dir)
    if not pool_dir.is_dir():
        raise SystemExit("--shared-pool %s is not a directory of <block>.jsonl pools" % pool_dir)
    paths = {p.stem: p for p in sorted(pool_dir.glob("*.jsonl"))}
    if not paths:
        raise SystemExit("--shared-pool %s holds no <block>.jsonl pools" % pool_dir)
    for name in paths:
        if not BLOCK_NAME.fullmatch(name):
            raise SystemExit("Pool %s.jsonl: a block id may hold only letters, digits, '_' and single '-', "
                             "because it names a directory under --out and is part of run_id" % name)
        if name.casefold() in RESERVED:
            raise SystemExit("Pool %s.jsonl: %s is a name this entry writes at the top of --out" % (name, name))
    if len({name.casefold() for name in paths}) != len(paths):
        raise SystemExit("--shared-pool %s holds block ids that differ only in case: %s"
                         % (pool_dir, ", ".join(paths)))
    conforming = tuple(sorted(paths)) in HF_BLOCK_SETS
    if backend == "hf" and not conforming:
        raise SystemExit("--backend hf needs exactly the pools %s, one per seed block (RUN_COST_SPEC 1); "
                         "%s holds %s" % (" or ".join(", ".join(b + ".jsonl" for b in s) for s in HF_BLOCK_SETS),
                                          pool_dir, ", ".join(p.name for p in paths.values())))
    return paths, {"from": "pool file names in --shared-pool", "rule": BLOCK_RULE, "conforming": conforming}


def refuse_existing(out, study_id, phase):
    """OUT must not exist or must be empty.  Nothing in it is ever overwritten.

    When OUT already holds a run of another study or phase, the message names
    the record and the field, so the mismatch is not mistaken for a stale
    directory.
    """
    if not out.exists():
        return
    if not out.is_dir():
        raise SystemExit("--out %s exists and is not a directory.  Nothing was overwritten." % out)
    if not any(out.iterdir()):
        return
    mismatched = []
    for path in [out / "experiment.json"] + sorted(out.glob("*/*/run.json")):
        try:
            record = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        for field, value in (("study_id", study_id), ("phase", phase)):
            if record.get(field) != value:
                mismatched.append("%s has %s %r, this run %r"
                                  % (path.relative_to(out).as_posix(), field, record.get(field), value))
    detail = "; it holds a run of another study or phase: " + "; ".join(mismatched) if mismatched else ""
    raise SystemExit("--out %s is not empty%s.  This entry does not resume; use a new directory.  "
                     "Nothing was overwritten." % (out, detail))


def read_block_pool(path, backend, tasks, base=None, dataset=None, generation=None, sampling=None):
    """Read one block's pool once, bound to the sha256 its meta file records.

    The rows are parsed from the same bytes that were hashed, so the pool that
    is matched is the pool that is bound.  On hf the pool is also bound to what
    sampled it, as its meta records (a field an older meta lacks is not
    checked):
      * `base` (the run's resolved config): the model and revision.  Another
        base's answers are not this run's pool, and with RSI_BASE_PIN the pools
        of several bases can sit side by side;
      * `dataset` (this run's manifest): the data, and the split train_001.
        Task ids leave the prompt out, so a pool drawn before a prompt change
        would otherwise pass and pair the new prompts with the old answers;
      * `generation`: all the generation settings, where the run samples with
        the same config as the pool (the one-step run);
      * `sampling`: only the SAMPLING_KEYS, where the run draws its later pools
        from the same distribution with a pool size of its own (the multi-round
        run: its candidates and prompts_per_pool are not the round-1 pool's).
    """
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.is_file():
        raise SystemExit("%s has no %s; pools come from sample_candidates.py with their meta"
                         % (path, meta_path.name))
    meta = read_json(meta_path)
    data = path.read_bytes()
    pool_id = hashlib.sha256(data).hexdigest()
    if pool_id != meta.get("hash"):
        raise SystemExit("%s changed after its meta was written: sha256 %s, meta records %s"
                         % (path, pool_id, meta.get("hash")))
    if meta.get("backend") != backend:
        raise SystemExit("%s was sampled with backend %r; this run uses %r" % (path, meta.get("backend"), backend))
    for key in ("model", "revision") if base is not None else ():
        if meta.get(key) != base[key]:
            raise SystemExit("%s was sampled from %s %r, but this run's base is %r.  Nothing was written."
                             % (path, key, meta.get(key), base[key]))
    stale = []
    if dataset is not None and (meta.get("data") or {}).get("dataset_hash", digest(dataset)) != digest(dataset):
        stale.append("data (a prompt or the questions changed since)")
    if dataset is not None and (meta.get("split") or {}).get("name", "train_001") != "train_001":
        stale.append("split (%s, not train_001)" % meta["split"]["name"])
    if generation is not None and "generation" in meta and drawn_generation(meta["generation"]) != generation:
        stale.append("sampling settings")
    drawn = drawn_generation(meta["generation"]) if "generation" in meta else {}
    if sampling is not None and any(drawn.get(key, sampling[key]) != sampling[key] for key in SAMPLING_KEYS):
        stale.append("sampling settings (%s; this run samples with %s)" % (
            ", ".join("%s %s" % (key, drawn.get(key)) for key in SAMPLING_KEYS),
            ", ".join("%s %s" % (key, sampling[key]) for key in SAMPLING_KEYS)))
    if stale:
        raise SystemExit("%s was drawn for another run: its %s differ.  Use the pools drawn with this "
                         "config, or draw them again.  Nothing was written."
                         % (path, ", ".join(stale)))
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if base is not None and any(not isinstance(row.get("truncated"), bool) for row in rows):
        # Drawn before rows recorded whether max_new_tokens cut them: a cut answer would pass as
        # finished and could be matched and trained on.
        raise SystemExit("%s has rows that do not record whether they were cut at max_new_tokens (a pool from "
                         "before that field).  Draw the pools again.  Nothing was written." % path)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("%s repeats a candidate id" % path)
    unknown = sorted({row["task_id"] for row in rows} - set(tasks))
    if unknown:
        raise SystemExit("%s has %d task ids that are not in train_001, e.g. %s" % (path, len(unknown), unknown[0]))
    binding = {"path": str(path.resolve()), "pool_id": pool_id, "rows": len(rows),
               "meta_path": str(meta_path.resolve()), "meta_sha256": file_hash(meta_path), "meta": meta,
               "fingerprint_recomputed": False, "fingerprint_reason": FINGERPRINT_REASON}
    return rows, binding


def check_branch_rows(block, r_rows, s_rows, k, error_fraction=DEFAULT_ERROR_FRACTION,
                      token_tolerance=DEFAULT_TOKEN_TOLERANCE):
    """Re-assert the hard constraints on the rows the arms will actually train on.

    rsi.matching audits its own output; this checks what reaches one_step after
    the per-block lookups, so a lookup that mixed blocks or reordered prompts
    is caught before any arm runs.
    """
    def correct(rows):
        return {r["id"] for r in rows if r["correct"]}

    c_quota, e_quota = quotas(k, error_fraction)
    problems = []
    for name, rows in (("R", r_rows), ("S", s_rows)):
        if len(rows) != k:
            problems.append("K_%s is %d, not %d" % (name, len(rows), k))
        if len({r["task_id"] for r in rows}) != len(rows):
            problems.append("%s holds a task twice" % name)
        if len(correct(rows)) != c_quota:
            problems.append("C:E in %s is not %d:%d" % (name, c_quota, e_quota))
        if any(r["weight"] != one_step.REQUIRED_WEIGHT for r in rows):
            problems.append("%s has a weight other than %r" % (name, one_step.REQUIRED_WEIGHT))
    if [r["task_id"] for r in r_rows] != [r["task_id"] for r in s_rows]:
        problems.append("R and S differ in prompts or prompt order")
    if correct(r_rows) != correct(s_rows):
        problems.append("R and S differ in correct ids")
    r_tokens, s_tokens = ({r["task_id"]: r["tokens"] for r in rows} for rows in (r_rows, s_rows))
    if set(r_tokens) != set(s_tokens) or any(
            token_tolerance is not None and abs(r_tokens[t] - s_tokens[t]) > token_tolerance for t in r_tokens):
        problems.append("R and S differ in per-task supervised token counts by more than %s" % token_tolerance)
    if problems:
        raise SystemExit("Block %s: matched rows break the hard constraints: %s" % (block, "; ".join(problems)))


def seed_derivation(seed, pool_keys):
    """RUN_COST_SPEC 1: the seed of every stage this entry runs, with its rule and value.

    `pool_keys` maps each block to rsi.matching.pool_key of its candidates: the matching
    draws per pool, so each block has its own values.
    """
    return {"cli_seed": seed,
            "pool_generate": {"value": None,
                              "reason": "the pools are inputs to this entry; their meta does not record "
                                        "the generation seed"},
            "match_preflight": {"rule": "rsi.matching draws from rng_for(seed, pool_key, label), i.e. "
                                        "random.Random(seed_for(seed, pool_key, label)), with the same --seed "
                                        "for every block and every K and the block's own pool_key (its "
                                        "candidate ids and responses), so blocks draw independently; the "
                                        "block id is not an input",
                                "seed": seed,
                                "per_block": {name: {"pool_key": key,
                                                     "task_order": seed_for(seed, key, "matched_subset_task_order"),
                                                     "assignment": seed_for(seed, key, "matched_subset_assignment"),
                                                     "selection": seed_for(seed, key, "matched_subset_selection")}
                                              for name, key in sorted(pool_keys.items())}},
            "update_with_diagnostics": {"rule": "--seed for every block and arm; one_step passes it to "
                                                "torch.manual_seed on hf; mock makes no random draws",
                                        "value": seed}}


def hardware(backend):
    record = {"machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
              "accelerator": None, "accelerator_reason": "the mock backend runs on CPU and loads no model"}
    if backend == "hf":
        import torch
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            record.update(accelerator=torch.cuda.get_device_name(0), accelerator_reason=None,
                          accelerator_memory_bytes=properties.total_memory, cuda=torch.version.cuda,
                          torch=torch.__version__)
        else:
            record["accelerator_reason"] = "torch reports no CUDA device"
    return record


def check_inputs_given(backend, adapter, reference):
    """hf needs both the shared adapter and h; mock, which loads no model, takes neither."""
    if backend == "hf":
        for flag, value in (("--shared-adapter", adapter), ("--reference-gradient", reference)):
            if not value:
                raise SystemExit("--backend hf needs %s: every arm starts from the shared adapter, and "
                                 "h^T Delta theta needs the h taken there.  Nothing was written." % flag)
    elif adapter or reference:
        raise SystemExit("--shared-adapter and --reference-gradient are hf inputs; the mock backend loads "
                         "no model.  Nothing was written.")


def check_hf_inputs(resolved, dataset, adapter_dir, reference_dir):
    """Prove the base, the shared adapter and h belong to this run, before anything is written.

    Returns the bindings the run records.  Every mismatch stops the run: an h
    taken at another adapter, base, config or dataset is a projection onto the
    wrong vector, and it would still produce a number.
    """
    try:
        base_pin = check_base_pin(resolved)
    except ValueError as exc:
        raise SystemExit("%s.  Nothing was written." % exc) from exc
    adapter_dir, reference_dir = Path(adapter_dir).resolve(), Path(reference_dir).resolve()
    try:
        shared = read_shared_record(adapter_dir)
    except FileNotFoundError as exc:
        raise SystemExit("%s.  Nothing was written." % exc) from exc
    for key, value in (("base_model", resolved["model"]), ("base_revision", resolved["revision"])):
        if shared.get(key) != value:
            raise SystemExit("The shared adapter %s records %s %r; this run uses %r.  Nothing was written."
                             % (adapter_dir, key, shared.get(key), value))
    differ = lora_mismatch(shared, resolved["training"])
    if differ:
        raise SystemExit("The shared adapter %s was made with other LoRA settings than this config states: %s.  "
                         "Nothing was written." % (adapter_dir, "; ".join(differ)))
    h_path, h_json = reference_dir / "h.pt", reference_dir / "h.json"
    for path in (h_path, h_json):
        if not path.is_file():
            raise SystemExit("%s is missing; --reference-gradient is the directory "
                             "compute_reference_gradient.py wrote.  Nothing was written." % path)
    record = read_json(h_json)
    if file_hash(h_path) != record["h"]["sha256"]:
        raise SystemExit("%s changed after h.json was written.  Nothing was written." % h_path)
    bound = record["bindings"]
    for key, have, want in (("shared_adapter_parameter_hash", bound.get("shared_adapter_parameter_hash"),
                             shared["parameter_hash"]),
                            ("base_model", bound.get("base_model"), resolved["model"]),
                            ("base_revision", bound.get("base_revision"), resolved["revision"]),
                            ("config_hash", bound.get("config_hash"), digest(resolved)),
                            ("dataset_hash", bound.get("data", {}).get("dataset_hash"), digest(dataset))):
        if have != want:
            raise SystemExit("h.json in %s was taken with %s %r; this run has %r.  Nothing was written."
                             % (reference_dir, key, have, want))
    return {"base_pin": base_pin,
            "shared_adapter": {"status": "used", "path": str(adapter_dir),
                               "parameter_hash": shared["parameter_hash"],
                               "record_sha256": file_hash(adapter_dir / "shared_adapter.json")},
            "reference_gradient": {"status": "used", "path": str(reference_dir), "h_sha256": file_hash(h_path),
                                   "h_json_sha256": file_hash(h_json)},
            "h_path": str(h_path)}


def load_tokenizer(resolved):
    """The base's tokenizer only: the token counts need no model on the GPU."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(resolved["model"], revision=resolved["revision"])


def bind_source(out, code_hash, git):
    """RUN_COST_SPEC 1: git_commit, git_dirty and, unless the tree is HEAD, a source snapshot."""
    binding = {"git_commit": git["git_commit"], "git_dirty": git["git_dirty"], "git_changes": git["changes"],
               "git_reason": git["reason"], "source_hash": code_hash, "source_snapshot": None}
    if git["git_dirty"] is not False:
        copied = snapshot_source(out / "source_snapshot")
        if copied != code_hash:
            raise RuntimeError("The source changed after this run read it: snapshot %s, run %s"
                               % (copied, code_hash))
        binding["source_snapshot"] = {"path": "source_snapshot", "source_hash": copied}
    return binding


def block_view(keyed, block):
    """One block's entries of a (block_id, candidate_id) mapping, keyed by candidate_id."""
    return {candidate_id: value for (name, candidate_id), value in keyed.items() if name == block}


def run_arm(out, block, branch, key, rows, config, seed, manifest, adapter_dir=None, h_path=None,
            audit_context=None):
    """One arm of one block: training rows, run.json, one_step's arm, complete.json, finished.json.

    On hf the arm starts from `adapter_dir` and projects onto the h at `h_path`,
    and must leave its diagnostics.json; the returned record then carries the
    diagnostics status and path, as the mock arm's carries its skip.
    """
    run_dir = out / block / branch
    round_dir = run_dir / ROUND
    training = round_dir / "training.jsonl"
    audit = {"queries": 0, "groups": {}}
    selected_examples = len(rows)
    auditing = auditing_enabled(config["audit"]) and branch != "null"
    if auditing:
        if audit_context is None:
            raise ValueError("Budgeted auditing needs task and original-pool denominator bindings")
        write_jsonl(round_dir / "pre_audit.jsonl", [{k: r[k] for k in TRAINING_FIELDS} for r in rows])
        rows, audit = apply_audit(rows, audit_context["tasks"], config["audit"],
                                  seed_for(seed, block, "budgeted_auditing"), 1,
                                  audit_context["pool_counts"])
        write_json(round_dir / "audit.json", audit)
    # Same narrow schema as rsi.experiment.run plus the keys: no label enters the training file.
    write_jsonl(training, [dict({k: r[k] for k in TRAINING_FIELDS}, block_id=block) for r in rows])
    manifest = dict(manifest, block_id=block, branch=branch, run_id=run_id(manifest["study_id"], block, branch),
                    training_rows={"path": training.relative_to(out).as_posix(), "subset": key,
                                   "examples": len(rows), "sha256": file_hash(training)})
    manifest["auditing"] = {"enabled": auditing, "config": config["audit"],
                            "matching_constraints_stage": "pre_audit" if auditing else "training",
                            "report": ROUND + "/audit.json" if auditing else None,
                            "report_sha256": file_hash(round_dir / "audit.json") if auditing else None,
                            "null_note": "frozen null is not audited" if branch == "null" else None}
    write_json(run_dir / "run.json", manifest)
    if config["backend"] == "mock":
        record = one_step.run_arm(branch, rows, config, None, round_dir, seed)
        if record.get("diagnostics") != "skipped":
            raise RuntimeError("Arm %s/%s reported diagnostics %r on the mock backend, which has no gradients"
                               % (block, branch, record.get("diagnostics")))
    else:
        with cost.stage(out / "cost_records", "train", branch, block,
                        {"examples": len(rows), "updates": 0 if branch == "null" else 1}) as attempt:
            record = one_step.run_arm(branch, rows, config, adapter_dir, round_dir, seed,
                                      {"reference_gradient_path": h_path})
            attempt.generated(0, 0)
        diagnostics = round_dir / "diagnostics.json"
        if not diagnostics.is_file():
            raise RuntimeError("Arm %s/%s wrote no diagnostics.json; a real arm records its gradients"
                               % (block, branch))
        record = dict(record, diagnostics="recorded", reason=None,
                      diagnostics_path=diagnostics.relative_to(out).as_posix())
    write_json(round_dir / "complete.json", {
        "round": 1, "block_id": block, "branch": branch, "subset": key,
        "adapter": ROUND + "/adapter" if (round_dir / "adapter").is_dir() else None,
        "examples": len(rows), "selected_examples": selected_examples, "training": record,
        "audit_queries": audit["queries"], "cumulative_audit_queries": audit["queries"],
        "audit_groups": audit["groups"]})
    write_json(run_dir / "finished.json", {"rounds": 1, "cumulative_audit_queries": audit["queries"],
                                           "calibration_queries": 0})
    return record


def main(args):
    config = load_config(args.config, args.seed, args.backend)
    one_step.check_isolation(config, allow_auditing=True)
    missing = identity(args.study_id, args.phase)
    hf = config["backend"] == "hf"
    if hf and missing:
        raise SystemExit("--backend hf needs --study-id and --phase (RUN_COST_SPEC 1); only mock may leave them "
                         "out.  Nothing was written.")
    check_inputs_given(config["backend"], args.shared_adapter, args.reference_gradient)
    data = Path(args.data).resolve()
    dataset = verify_dataset(data)
    resolved = pin_config(config)
    inputs = check_hf_inputs(resolved, dataset, args.shared_adapter, args.reference_gradient) if hf else None
    if hf:
        try:
            require_explicit(args.config, "one_step")  # every value a run uses is in its config file
        except ValueError as exc:
            raise SystemExit("%s.  Nothing was written." % exc) from exc
    # S's target follows the dataset's task: nonshortest for graph, ignore_parentheses for arithmetic,
    # intermediate for gsm8k, sign for dmmath.
    try:
        targets = targets_for(dataset["task"])
    except ValueError as exc:
        raise SystemExit("%s.  Nothing was written." % exc) from exc
    tasks = {t["id"]: t for t in read_jsonl(data / "train_001.jsonl")}
    paths, block_source = find_blocks(args.shared_pool, resolved["backend"])
    out = Path(args.out).resolve()
    refuse_existing(out, args.study_id, args.phase)

    pools = {name: read_block_pool(path, resolved["backend"], tasks, base=resolved if hf else None,
                                   dataset=dataset if hf else None,
                                   generation=resolved["generation"] if hf else None)
             for name, path in paths.items()}
    # Keyed by (block_id, candidate_id): candidate ids are digest([task_id, sample])
    # and repeat across blocks, so a bare candidate id would let one block's
    # judgement or token count stand in for another's.
    judgements = {(name, r["id"]): judge(tasks[r["task_id"]], r["response"])
                  for name, (rows, _) in pools.items() for r in rows}
    if hf:
        tokenizer = load_tokenizer(resolved)
        tokens = {(name, r["id"]): supervised_token_count(tokenizer, r["response"])
                  for name, (rows, _) in pools.items() for r in rows}
    else:
        tokens = {(name, r["id"]): r["completion_tokens"] for name, (rows, _) in pools.items() for r in rows}
    candidates = {name: [candidate_from_row(r, tasks[r["task_id"]], judgements[name, r["id"]], tokens[name, r["id"]])
                         for r in rows]
                  for name, (rows, _) in pools.items()}
    try:
        result = select_k_for_blocks(candidates, seed=args.seed, k_ladder=K_LADDER, target_signatures=targets,
                                     **resolved["matching"])
    except MatchingFailure as failure:
        # Not an infeasible pool (that is returned, below): a malformed request
        # or a matching bug.  Neither is a result.
        raise SystemExit("Matching failed, not infeasible: %s.  Nothing was written." % failure)
    branch_rows = {}
    if result["feasible"]:
        for name, (rows, _) in pools.items():
            matched = result["per_block"][name]
            branch_rows[name] = {key: one_step.load_branch_rows(matched[key], tasks, rows, block_view(judgements, name),
                                                                block_view(tokens, name))
                                 for key in ("R", "S")}
            check_branch_rows(name, branch_rows[name]["R"], branch_rows[name]["S"], result["K"], **resolved["matching"])
    env, git = environment(), git_state()
    seeds = seed_derivation(args.seed, {name: pool_key(cands) for name, cands in candidates.items()})

    out.mkdir(parents=True, exist_ok=True)
    experiment = {"schema_version": SCHEMA_VERSION, "status": "running", "exit_code": None,
                  "backend": resolved["backend"], "DEMO_ONLY": resolved["backend"] == "mock",
                  "study_id": args.study_id, "phase": args.phase, "identity_missing": missing,
                  "block_id": "shared", "branch": "shared", "run_id": run_id(args.study_id, "shared", "shared"),
                  "seed": args.seed, "seed_derivation": seeds, "blocks": list(paths), "block_id_source": block_source,
                  "arms": [arm for arm, _ in one_step.ARMS], "K": result["K"], "code": None, "matching": None,
                  "matching_settings": resolved["matching"],
                  "task": dataset["task"], "target_signatures": list(targets),
                  "auditing": {"enabled": auditing_enabled(resolved["audit"]), "config": resolved["audit"],
                               "budget_unit": "correctness queries per R/S arm per round",
                               "matching_constraints_stage": "pre_audit"},
                  "pools": {name: pool for name, (_, pool) in pools.items()}, "paths": PATHS,
                  "cost_records": COST_RECORDS if hf else COST_RECORDS_MOCK, "completed": []}
    write_json(out / "experiment.json", experiment)
    try:
        experiment["code"] = code = bind_source(out, env["source_hash"], git)
        write_matching(out / "matching", list(paths), list(K_LADDER), result,
                       metadata={"task": dataset["task"], "target_signatures": list(targets)})
        matching = {"K": result["K"], "feasible": result["feasible"], "ladder": list(K_LADDER), "seed": result["seed"],
                    "ladder_path": "matching/ladder.json", "ladder_sha256": file_hash(out / "matching" / "ladder.json"),
                    "subsets_path": None, "subsets_sha256": None}
        if result["feasible"]:
            matching.update(subsets_path="matching/matched_subsets.json",
                            subsets_sha256=file_hash(out / "matching" / "matched_subsets.json"))
        experiment["matching"] = matching
        if not result["feasible"]:
            write_json(out / "experiment.json", dict(experiment, status="infeasible", exit_code=2))
            report_infeasible(result)
            return 2
        write_json(out / "experiment.json", experiment)

        shared = {"schema_version": SCHEMA_VERSION, "study_id": args.study_id, "phase": args.phase,
                  "identity_missing": missing, "block_id_source": block_source, "seed": args.seed,
                  "seed_derivation": seeds,
                  "seeds": {"update_with_diagnostics": args.seed},  # and each block's match_preflight, below
                  # The fields rsi.experiment.run writes, so evaluate.py can read an arm directory.
                  "config": resolved, "requested_config": config, "dataset_hash": digest(dataset),
                  "data_path": str(data), "environment": env, "calibration": None,
                  "DEMO_ONLY": resolved["backend"] == "mock", "task": dataset["task"],
                  "paths": PATHS, "evaluation": "pending",
                  "cost_records": COST_RECORDS if hf else COST_RECORDS_MOCK}
        bindings = dict(code, config_hash=digest(resolved),
                        config_file={"path": str(Path(args.config).resolve()), "sha256": file_hash(args.config)},
                        generation={"config": resolved["generation"], "used": False,
                                    "reason": "this entry samples nothing; the pools are bound by sha256"},
                        hardware=hardware(resolved["backend"]),
                        data={"path": str(data), "manifest_sha256": file_hash(data / "manifest.json"),
                              "dataset_hash": digest(dataset)})
        if hf:
            bindings.update(model={"name": resolved["model"], "revision": resolved["revision"],
                                   "revision_resolved": True},
                            tokenizer={"name": resolved["model"], "revision": resolved["revision"]},
                            token_counts={"source": "supervised_token_count", "detail": HF_TOKEN_SOURCE},
                            base_pin=inputs["base_pin"], shared_adapter=inputs["shared_adapter"],
                            reference_gradient=inputs["reference_gradient"])
        else:
            bindings.update(model={"name": resolved["model"], "revision": resolved["revision"],
                                   "revision_resolved": False,
                                   "reason": "pin_config resolves the revision on hf only; the mock backend "
                                             "loads no model"},
                            tokenizer={"name": None, "reason": "the mock backend has no tokenizer"},
                            token_counts={"source": "completion_tokens", "detail": MOCK_TOKEN_SOURCE},
                            shared_adapter={"status": "not_used", "reason": NOT_USED_ON_MOCK},
                            reference_gradient={"status": "not_used", "reason": NOT_USED_ON_MOCK})
        for name in paths:
            block_bindings = dict(bindings, pool=pools[name][1],
                                  matching=dict(matching, block_hash=result["per_block"][name]["hash"]))
            block_seeds = dict(shared["seeds"], match_preflight=dict(
                seed=args.seed, **seeds["match_preflight"]["per_block"][name]))
            gradients = []
            for branch, key in one_step.ARMS:
                record = run_arm(out, name, branch, key, branch_rows[name][key], resolved, args.seed,
                                 dict(shared, bindings=block_bindings, seeds=block_seeds),
                                 adapter_dir=inputs["shared_adapter"]["path"] if hf else None,
                                 h_path=inputs["h_path"] if hf else None,
                                 audit_context={"tasks": tasks, "pool_counts": result["per_block"][name]["audit"]})
                line = {"block_id": name, "branch": branch, "run_id": run_id(args.study_id, name, branch),
                        "step": 0, "status": record["diagnostics"], "reason": record["reason"],
                        "arm_record": "%s/%s/%s/arm.json" % (name, branch, ROUND)}
                if hf:
                    line["diagnostics"] = record["diagnostics_path"]
                gradients.append(line)
                write_jsonl(out / name / "diagnostics" / "gradients.jsonl", gradients)
                experiment["completed"].append(name + "/" + branch)
                write_json(out / "experiment.json", experiment)
                print("block=%s arm=%s examples=%d diagnostics=%s" % (name, branch, record["examples"],
                                                                      record["diagnostics"]), flush=True)
    except BaseException as exc:
        write_json(out / "experiment.json", dict(experiment, status="failed",
                                                 exit_code=1 if isinstance(exc, Exception) else None,
                                                 failure_reason="%s: %s" % (type(exc).__name__, exc)))
        raise
    write_json(out / "experiment.json", dict(experiment, status="complete", exit_code=0))
    print("wrote", out / "experiment.json")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Experiment config: on hf a complete one, e.g. configs/matched_pool.json (configs/README.md); "
                                                "configs/controlled.json for mock")
    p.add_argument("--data", required=True, help="Dataset root from generate_data.py (manifest.json, train_001.jsonl)")
    p.add_argument("--shared-pool", required=True,
                   help="Directory of <block>.jsonl pools with their .meta.json files")
    p.add_argument("--out", required=True, help="Output directory; must not exist or must be empty")
    p.add_argument("--seed", type=int, default=0, help="Matching and update seed, the same for every block (default 0)")
    p.add_argument("--study-id", help="RUN_COST_SPEC 1 study_id, e.g. md-pilot-20261001-v01; optional on mock only")
    p.add_argument("--phase", choices=PHASES, help="RUN_COST_SPEC 1 phase; optional on mock only")
    p.add_argument("--backend", choices=["hf", "mock"],
                   help="Overrides the config's backend (hf unless the config says otherwise)")
    p.add_argument("--shared-adapter", help="hf: the shared_adapter/ every arm starts from (make_shared_adapter.py)")
    p.add_argument("--reference-gradient",
                   help="hf: the directory compute_reference_gradient.py wrote at that adapter (h.pt, h.json)")
    a = p.parse_args()
    sys.exit(main(a))
