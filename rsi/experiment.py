from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

from .auditing import audit_pool
from .backends import CHAT_TEMPLATE_KWARGS, backend, resolve_revision
from .common import (digest, environment, file_hash, read_json, read_jsonl, rng_for,
                     seed_for, source_hash, verify_dataset, write_json, write_jsonl)
from .provenance import git_state
from .verifiers import one_per_prompt, select_controlled, verification_prompt, verifier_identity

# The fields sample_pool writes into a pool's meta itself.  An entry's `inputs`
# (split, dataset, base pin, identity, argv, ...) are added beside them and may
# not replace one: these are what run() and a later reader check.
POOL_META_FIELDS = ("fingerprint", "hash", "backend", "seed", "model", "revision", "generation",
                    "generation_hash", "config", "config_hash", "decoding", "code", "row_fields",
                    "environment")
GENERATION_SEED_RULE = "seed_for(cli_seed, 1, 'generation')"
ROW_FIELDS = {
    "completion_tokens": "hf: the generated token ids up to and including the first stop id "
                         "(generation_config.eos_token_id or tokenizer.eos_token_id, rsi/backends.py), or all "
                         "of them when no stop id was generated (then `truncated` is true); mock: the number of "
                         "whitespace-separated words of the response.  Neither is the supervised count "
                         "len(tokenizer.encode(response, add_special_tokens=False)) + 1, which matching must "
                         "recount (review note (a)).",
    "response": "hf: tokenizer.decode(those ids, skip_special_tokens=True).strip(), with the tokenizer's "
                "clean_up_tokenization_spaces off; the ids and the unstripped text are not kept.  mock: a "
                "synthetic answer.",
    "truncated": "hf: true when the answer was cut: no stop id within max_new_tokens "
                 "(completion_tokens == max_new_tokens), or stopped earlier while repeating a block "
                 "(generation.repetition_stop; completion_tokens < max_new_tokens).  Such a row is judged and "
                 "kept in the pool, but no matching, selection or training uses it.  mock: always false.",
}


def pin_config(config):
    # Round-trip copy: callers retain the requested configuration for resume checks.
    import copy
    c = copy.deepcopy(config)
    if c["backend"] == "hf":
        c["revision"] = resolve_revision(c["model"], c["revision"])
        if c["verifier"]["kind"] == "llm_fixed":
            c["verifier"]["revision"] = resolve_revision(c["verifier"]["model"], c["verifier"]["revision"])
    return c


def read_calibration(config, dataset):
    v = config["verifier"]
    if not v["kind"].startswith("llm_"):
        return None
    if not v["calibration"]:
        if v["threshold"] is None:
            raise ValueError("LLM verifiers require --calibration FILE or an explicit uncalibrated threshold in config")
        return {"threshold": v["threshold"], "trusted_queries": 0, "uncalibrated": True}
    c = read_json(v["calibration"])
    identity = verifier_identity(config)
    for key in ("model", "revision"):
        if c["verifier"][key] != identity[key]:
            raise ValueError("Calibration verifier mismatch: " + key + "; pin the same model revision")
    if c["dataset_hash"] != digest(dataset) or c["backend"] != config["backend"]:
        raise ValueError("Calibration dataset/backend mismatch")
    return c


def pool_fingerprint(config, tasks, seed):
    return digest({"model": config["model"], "revision": config["revision"], "backend": config["backend"],
                   "generation": config["generation"], "seed": seed, "tasks": tasks,
                   "source_hash": source_hash()})


def decoding_record(model, config):
    """The decoding a pool is drawn under, as far as it can be read without running generate.

    On hf, transformers merges the model's generation_config -- read here, after
    the load and before generate -- with the keyword arguments HFBackend.generate
    passes.  The merged object lives inside generate() and is not read, so this
    records the two parts ("defaults + overrides"), not a final configuration.
    tests/test_sample_candidates.py runs HFBackend.generate against a stub model
    and checks that `overrides` is what it passes.  Mock loads no model.
    """
    loaded = getattr(model, "model", None)
    if config["backend"] != "hf" or loaded is None:
        return {"composition": None,
                "reason": "%s loads no model: of the generation fields only candidates is used" % config["backend"]}
    g, defaults = config["generation"], loaded.generation_config
    record = {
        "composition": "defaults + overrides: model_defaults is the model's generation_config as loaded, "
                       "overrides are the keyword arguments HFBackend.generate passes; transformers merges "
                       "them inside generate(), and that merged object is not read",
        "model_defaults": defaults.to_dict(),
        "overrides": {"do_sample": True, "temperature": g["temperature"], "top_p": g["top_p"],
                      "top_k": g["top_k"], "max_new_tokens": g["max_new_tokens"],
                      "pad_token_id": model.tokenizer.pad_token_id},
        "stop_ids": {"generation_config_eos_token_id": defaults.eos_token_id,
                     "tokenizer_eos_token_id": model.tokenizer.eos_token_id},
        "transformers_version": getattr(sys.modules.get("transformers"), "__version__", None),
        # Llama 3.x templates render a date into every prompt; the kwargs fix it (rsi/backends.py).
        "prompt": {"rule": "rsi.backends.chat_prompt_ids: the task prompt as one user turn, "
                           "add_generation_prompt=True",
                   "chat_template_digest": digest(getattr(model.tokenizer, "chat_template", None)),
                   "template_kwargs": CHAT_TEMPLATE_KWARGS},
        "stopping": {"repetition_stop": g.get("repetition_stop"),
                     "rule": "rsi.backends.repetition_stop, passed as stopping_criteria when set: a row stops "
                             "once its last span generated tokens repeat with a period of at most max_period "
                             "tokens; it is cut there and marked `truncated` (ROW_FIELDS)"},
        "seeding": "torch.manual_seed and torch.cuda.manual_seed_all once, at the start of generate(); "
                   "not reset per batch or per candidate, so batch_size is part of what reproduces a pool",
        "batch_size": g["batch_size"],
        "not_passed": "candidates, batch_size and max_sequence_length are not passed to transformers: "
                      "candidates is the rows drawn per task, batch_size the rows per generate call, and "
                      "max_sequence_length refuses a prompt + max_new_tokens beyond it; an answer that reaches max_new_tokens without a stop id is kept and marked `truncated` (ROW_FIELDS)",
    }
    # Serialised now, before a single token is generated: a value the meta
    # cannot hold fails here rather than after the pool is drawn.
    return json.loads(json.dumps(record, default=repr, allow_nan=False))


def sample_pool(config, tasks, output, seed, inputs=None):
    """Draw the pool from `tasks` and write it with `<output>.meta.json`.

    `config` is the configuration in effect (pinned).  The meta keeps
    `fingerprint`, `hash` and `backend` exactly as before -- run() checks the
    first two, and RUN_COST_SPEC 6 keeps an old pool readable -- and adds, as
    plain fields, what the fingerprint hashes but cannot give back (review note
    (d)): the CLI seed and the seed generate() actually received, the
    model and revision, the generation fields, the configuration in effect, the
    decoding, and the code.  `inputs` are facts only the entry knows (split,
    dataset, base pin, config file, identity, argv); they are added beside
    these and may not replace one.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    inputs = dict(inputs or {})
    clash = sorted(set(inputs) & set(POOL_META_FIELDS))
    if clash:
        raise ValueError("inputs may not replace the pool meta's own fields: " + ", ".join(clash))
    # What the meta will hold is checked before the model is loaded, so a value
    # it cannot hold fails now rather than after the pool is drawn.
    json.dumps({"config": config, "inputs": inputs}, allow_nan=False)
    # One value, both passed to generate() and recorded: the meta's seed is the
    # one the pool was drawn with, not a recomputation of it.
    generation_seed = seed_for(seed, 1, "generation")
    env, code = environment(), git_state()
    model = backend(config)
    try:
        decoding = decoding_record(model, config)
        rows = model.generate(tasks, config["generation"]["candidates"], generation_seed)
    finally:
        model.close()
    write_jsonl(output, rows)
    meta = {"fingerprint": pool_fingerprint(config, tasks, seed), "hash": file_hash(output),
            "backend": config["backend"],
            "seed": {"cli": seed, "generation": generation_seed, "rule": GENERATION_SEED_RULE},
            "model": config["model"], "revision": config["revision"],
            "generation": config["generation"], "generation_hash": digest(config["generation"]),
            "config": config, "config_hash": digest(config),
            "decoding": decoding,
            "code": {"git_commit": code["git_commit"], "git_dirty": code["git_dirty"],
                     "git_changes": code["changes"], "git_reason": code["reason"],
                     "source_hash": env["source_hash"]},
            "row_fields": ROW_FIELDS,
            "environment": env}
    meta.update(inputs)
    write_json(str(output) + ".meta.json", meta)


def run(config, data, out, resume=False, initial_pool=None):
    data, out = Path(data).resolve(), Path(out).resolve()
    dataset = verify_dataset(data)
    if config["rounds"] > dataset["rounds"]:
        raise ValueError("Generate more training rounds first")
    manifest_path = out / "run.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Run exists; use --resume for completed-round restart")
        manifest = read_json(manifest_path)
        if manifest["requested_config"] != config or manifest["dataset_hash"] != digest(dataset):
            raise ValueError("Resume config/data mismatch")
        if manifest["environment"]["source_hash"] != source_hash():
            raise ValueError("Code changed since run creation; use a new output directory")
        resolved = manifest["config"]
        calibration = manifest["calibration"]
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Output directory is nonempty")
        resolved = pin_config(config)
        calibration = read_calibration(resolved, dataset)
        manifest = {"config": resolved, "requested_config": config, "dataset_hash": digest(dataset),
                    "data_path": str(data), "environment": environment(), "calibration": calibration,
                    "DEMO_ONLY": config["backend"] == "mock", "task": dataset["task"]}
        write_json(manifest_path, manifest)
    cfg = resolved
    previous_adapter, audit_history, total_queries = None, {}, 0
    for t in range(1, cfg["rounds"] + 1):
        round_dir = out / ("round_%03d" % t)
        done_path = round_dir / "complete.json"
        if done_path.exists():
            done = read_json(done_path)
            previous_adapter = out / done["adapter"] if done["adapter"] else None
            if previous_adapter and not previous_adapter.exists():
                raise FileNotFoundError(previous_adapter)
            audit_history = done["audit_groups"]
            total_queries = done["cumulative_audit_queries"]
            continue
        start = time.perf_counter()
        # Interrupted attempts are preserved. Only complete.json commits a round.
        attempt = round_dir / ("attempt_" + uuid.uuid4().hex[:10])
        attempt.mkdir(parents=True)
        write_json(attempt / "started.json", {"round": t, "unix_time": time.time()})
        tasks_list = read_jsonl(data / ("train_%03d.jsonl" % t))
        tasks = {task["id"]: task for task in tasks_list}
        if initial_pool and t == 1:
            meta = read_json(str(initial_pool) + ".meta.json")
            if meta["fingerprint"] != pool_fingerprint(cfg, tasks_list, cfg["seed"]) or meta["hash"] != file_hash(initial_pool):
                raise ValueError("Initial candidate pool does not match this model/data/seed/config")
            candidates = read_jsonl(initial_pool)
        else:
            solver = backend(cfg, adapter=previous_adapter)
            try:
                candidates = solver.generate(tasks_list, cfg["generation"]["candidates"], seed_for(cfg["seed"], t, "generation"))
            finally:
                solver.close()
        write_jsonl(attempt / "candidates.jsonl", candidates)
        # An answer cut at max_new_tokens is not an answer: it never reaches the verifier, so it can
        # never be trained on.  candidates.jsonl keeps it; the round record counts it.
        generated, generated_tokens = len(candidates), sum(c["completion_tokens"] for c in candidates)
        truncated = sum(bool(c.get("truncated")) for c in candidates)
        candidates = [c for c in candidates if not c.get("truncated")]
        if cfg["verifier"]["kind"].startswith("llm_"):
            identity = verifier_identity(cfg)
            verifier = backend(cfg, adapter=previous_adapter if cfg["verifier"]["kind"] == "llm_self" else None,
                               model_name=identity["model"], revision=identity["revision"])
            try:
                prompts = [verification_prompt(tasks[c["task_id"]]["prompt"], c["response"]) for c in candidates]
                scores = verifier.score_verdicts(prompts)
            finally:
                verifier.close()
            selected = [dict(c, accepted=score >= calibration["threshold"], verifier_score=score)
                        for c, score in zip(candidates, scores)]
        else:
            selected = select_controlled(candidates, tasks, cfg["verifier"], cfg["seed"], t)
        write_jsonl(attempt / "selection.jsonl", selected)
        provisional = one_per_prompt(selected, cfg["seed"], t)
        write_jsonl(attempt / "provisional.jsonl", provisional)
        retained, audit = audit_pool(provisional, tasks, cfg["audit"], cfg["seed"], t, audit_history)
        write_json(attempt / "audit.json", audit)
        rng_for(cfg["seed"], t, "training_cap").shuffle(retained)
        retained = retained[:cfg["training"]["examples"]]
        write_jsonl(attempt / "retained.jsonl", retained)
        # Deliberately narrow schema: no oracle answer or label enters SFT.
        training_rows = [{"prompt": tasks[c["task_id"]]["prompt"], "response": c["response"],
                          "weight": c["weight"]} for c in retained]
        write_jsonl(attempt / "training.jsonl", training_rows)
        trained = {"trained": False, "steps": 0, "mean_loss": None}
        if not cfg["frozen"] and any(c["weight"] > 0 for c in retained):
            solver = backend(cfg, adapter=previous_adapter, trainable=True)
            try:
                trained = solver.train(training_rows, attempt / "adapter", seed_for(cfg["seed"], t, "train"))
            finally:
                solver.close()
            if trained["trained"]:
                previous_adapter = attempt / "adapter"
        total_queries += audit["queries"]
        audit_history = audit["groups"]
        weights = [c["weight"] for c in retained]
        elapsed = time.perf_counter() - start
        done = {"round": t, "attempt": str(attempt.relative_to(out)),
                "adapter": str(previous_adapter.relative_to(out)) if previous_adapter else None,
                "candidates": generated, "truncated_candidates_excluded": truncated,
                "accepted": sum(c["accepted"] for c in selected),
                "provisional_prompts": len(provisional), "retained": len(retained),
                "training_shortfall": max(0, cfg["training"]["examples"] - len(retained)),
                "positive_weight_examples": sum(w > 0 for w in weights),
                "effective_sample_size": sum(weights)**2 / sum(w*w for w in weights) if any(weights) else 0,
                "audit_queries": audit["queries"], "cumulative_audit_queries": total_queries,
                "audit_groups": audit_history, "training": trained, "wall_seconds": elapsed,
                "allocated_gpu_hours": elapsed/3600 if cfg["backend"] == "hf" and cfg["device"].startswith("cuda") else 0,
                "generated_tokens": generated_tokens}
        write_json(done_path, done)
        print("round=%d retained=%d audits=%d seconds=%.1f" % (t, len(retained), audit["queries"], elapsed), flush=True)
    write_json(out / "finished.json", {"rounds": cfg["rounds"], "cumulative_audit_queries": total_queries,
                                       "calibration_queries": calibration.get("trusted_queries", 0) if calibration else 0})
    return out
