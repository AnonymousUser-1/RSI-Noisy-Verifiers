from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_for(*parts):
    return int(digest(parts)[:8], 16)


def rng_for(*parts):
    return random.Random(seed_for(*parts))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hash():
    root = Path(__file__).resolve().parents[1]
    # Do not traverse a local .venv, model cache, or experiment output directories.
    files = list(root.glob("*.py")) + list((root/"rsi").rglob("*.py")) + list((root/"tests").rglob("*.py"))
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(files)})


def environment():
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for name in ["numpy", "torch", "transformers", "peft", "accelerate", "matplotlib", "scipy"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "source_hash": source_hash(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}


def merge_dict(base, changes):
    result = dict(base)
    for key, value in changes.items():
        result[key] = merge_dict(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


DEFAULTS = {
    "backend": "hf", "model": "Qwen/Qwen3-1.7B", "revision": "main", "device": "cuda",
    "dtype": "bfloat16", "rounds": 8, "seed": 0, "frozen": False,
    # max_new_tokens is far above any finished answer (graph and arithmetic answers stay under 300
    # tokens; prompts under 500), so only a runaway generation reaches it; such a row is marked
    # `truncated` and never trained on.  prompts_per_pool: the first N prompts of the split a pool
    # samples (null: all of them).  repetition_stop: {"span": S, "max_period": P} stops a row whose
    # last S generated tokens repeat with a period of at most P tokens, a loop (rsi.backends.
    # repetition_stop); it is marked `truncated` like a row cut at max_new_tokens.  null: off.
    "generation": {"candidates": 4, "prompts_per_pool": None, "batch_size": 16, "max_new_tokens": 2048,
                   "max_sequence_length": 4096, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                   "repetition_stop": None},
    "training": {"examples": 1024, "epochs": 2, "learning_rate": 5e-5, "batch_size": 2,
                 "effective_batch_size": 64, "lora_rank": 8, "lora_alpha": 16,
                 "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
                 "gradient_checkpointing": True},
    "verifier": {"kind": "iid", "tpr": 0.9, "fpr": 0.2, "model": "Qwen/Qwen3-1.7B",
                 "revision": "main", "calibration": None, "threshold": None},
    "audit": {"policy": "none", "budget": 0, "weighting": True},
    # R/S selection (rsi.matching, rsi.iterative).  error_fraction: E = K * error_fraction wrong
    # answers in each arm's K, C = K - E correct (0.25: C:E = 3:1).  token_tolerance: in round-1
    # matching, R's and S's wrong answers on one prompt differ by at most this many supervised
    # tokens (0: identical; null: no limit).
    "matching": {"error_fraction": 0.25, "token_tolerance": 0},
}


def load_config(path, seed=None, backend=None):
    raw = read_json(path)
    # `target_modules` is the frozen field name.  `lora_target_modules` is accepted as an
    # alias so earlier configs keep loading, but the two may never disagree silently: a
    # config that sets both to different lists is a protocol fork, not a preference.
    # This has to run *before* the unknown-field gate below, which would otherwise reject
    # the alias as an unknown key and never reach the mapping.
    if isinstance(raw.get("training"), dict) and "lora_target_modules" in raw["training"]:
        legacy = raw["training"]["lora_target_modules"]
        if "target_modules" in raw["training"]:
            raise ValueError("Config sets both target_modules and lora_target_modules; "
                             "keep one name so the frozen protocol has a single source")
        del raw["training"]["lora_target_modules"]
        raw["training"]["target_modules"] = legacy
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown config fields: " + str(sorted(unknown)))
    for key in ("generation", "training", "verifier", "audit", "matching"):
        if set(raw.get(key, {})) - set(DEFAULTS[key]):
            raise ValueError("Unknown fields in " + key)
    config = merge_dict(DEFAULTS, raw)
    if seed is not None:
        config["seed"] = seed
    if backend:
        config["backend"] = backend
    if config["backend"] not in ("hf", "mock"):
        raise ValueError("backend must be hf or mock")
    if config["verifier"]["kind"] not in ("exact", "unfiltered", "iid", "persistent", "rotating", "llm_fixed", "llm_self"):
        raise ValueError("Unknown verifier kind")
    if config["audit"]["policy"] not in ("none", "uniform", "balanced", "adaptive"):
        raise ValueError("Unknown audit policy")
    for key in ("tpr", "fpr"):
        if not 0 <= config["verifier"][key] <= 1:
            raise ValueError(key + " must lie in [0,1]")
    if config["audit"]["budget"] < 0 or (config["audit"]["policy"] == "none" and config["audit"]["budget"]):
        raise ValueError("Invalid audit budget/policy")
    for key in ("rounds",):
        if config[key] < 1:
            raise ValueError(key + " must be positive")
    for group, keys in (("generation", ("candidates", "batch_size", "max_new_tokens", "max_sequence_length")),
                        ("training", ("examples", "epochs", "batch_size", "effective_batch_size"))):
        if any(config[group][k] < 1 for k in keys):
            raise ValueError("Positive counts required in " + group)
    prompts = config["generation"]["prompts_per_pool"]
    if prompts is not None and (isinstance(prompts, bool) or not isinstance(prompts, int) or prompts < 1):
        raise ValueError("generation.prompts_per_pool must be a positive integer or null (all prompts)")
    stop = config["generation"]["repetition_stop"]
    if stop is not None and (not isinstance(stop, dict) or set(stop) != {"span", "max_period"}
                             or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in stop.values())
                             or stop["span"] < 2 * stop["max_period"]):
        raise ValueError('generation.repetition_stop must be null or {"span": S, "max_period": P} with positive '
                         "integers and S >= 2P")
    from .matching import K_LADDER, quotas  # here, not at the top: rsi.matching imports this module
    fraction = config["matching"]["error_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("matching.error_fraction must be a number")
    for k in K_LADDER:
        try:
            quotas(k, fraction)
        except ValueError as exc:
            raise ValueError("matching.error_fraction: %s" % exc) from exc
    tolerance = config["matching"]["token_tolerance"]
    if tolerance is not None and (isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0):
        raise ValueError("matching.token_tolerance must be a nonnegative integer or null (no limit)")
    if config["training"]["effective_batch_size"] % config["training"]["batch_size"]:
        raise ValueError("effective_batch_size must be a multiple of batch_size")
    if config["training"]["lora_rank"] < 1:
        raise ValueError("lora_rank must be positive")
    if config["training"]["lora_alpha"] < 1:
        raise ValueError("lora_alpha must be positive")
    if not 0 <= config["training"]["lora_dropout"] < 1:
        raise ValueError("lora_dropout must lie in [0,1)")
    # The `lora_target_modules` alias was resolved at the top of this function,
    # before the unknown-field gate; by here only `target_modules` remains.
    targets = config["training"]["target_modules"]
    if not targets or not all(isinstance(t, str) and t for t in targets):
        raise ValueError("target_modules must be a non-empty list of module names")
    return config


# The config fields each matched-line stage reads (configs/README.md).  On hf a stage refuses a
# config file that does not state one of them, so no value a run uses comes silently from DEFAULTS.
_BASE_FIELDS = ("backend", "model", "revision", "dtype", "device")
# The adapter stage builds the LoRA from these; the later stages load the saved adapter and refuse a
# config whose values differ from its record (rsi.shared_adapter.lora_mismatch).
_LORA_FIELDS = ("training.lora_rank", "training.lora_alpha", "training.lora_dropout", "training.target_modules")
_GENERATION_FIELDS = tuple("generation." + k for k in (
    "candidates", "prompts_per_pool", "batch_size", "max_new_tokens", "max_sequence_length",
    "temperature", "top_p", "top_k", "repetition_stop"))
_POOL_FIELDS = _BASE_FIELDS + _GENERATION_FIELDS


def drawn_generation(meta_generation):
    """A pool meta's generation settings, with the fields added since it was drawn at their old behaviour.

    A pool drawn before generation.repetition_stop existed records no such field; its sampling had no
    loop stop, which the field states as null.  The meta itself is never rewritten: a config that
    states the loop stop on still differs from such a pool, and is refused."""
    return dict({"repetition_stop": None}, **(meta_generation or {}))
# Every entry runs one_step.check_isolation, which reads the audit settings.
_AUDIT_FIELDS = ("audit.policy", "audit.budget", "audit.weighting")
# The R/S selection: round-1 matching (one-step and multi-round), later rounds, the pilot's estimate.
_MATCHING_FIELDS = ("matching.error_fraction", "matching.token_tolerance")
STAGE_FIELDS = {
    "pool": _POOL_FIELDS,                                       # sample_candidates.py
    "pilot": ("model", "revision") + _GENERATION_FIELDS + _MATCHING_FIELDS,  # scripts/multiround/pilot_report.py
    "adapter": _BASE_FIELDS + _LORA_FIELDS,                     # make_shared_adapter.py
    "reference": _BASE_FIELDS + _LORA_FIELDS + _AUDIT_FIELDS,   # compute_reference_gradient.py
    # run_matched_experiment.py and one_step.py; `rounds` is read back from each arm's run.json.
    "one_step": _POOL_FIELDS + ("rounds",) + _LORA_FIELDS + ("training.learning_rate",) + _AUDIT_FIELDS
                + _MATCHING_FIELDS,
    "iterative": _POOL_FIELDS + ("rounds",) + _LORA_FIELDS + tuple("training." + k for k in (
        "learning_rate", "epochs", "batch_size", "effective_batch_size", "gradient_checkpointing")) + _AUDIT_FIELDS
                 + _MATCHING_FIELDS,
}


def require_explicit(path, stage):
    """Refuse a config file that leaves a field `stage` reads to DEFAULTS (STAGE_FIELDS)."""
    raw = read_json(path)
    missing = []
    for field in STAGE_FIELDS[stage]:
        node = raw
        for part in field.split("."):
            if part == "target_modules" and isinstance(node, dict) and "lora_target_modules" in node:
                part = "lora_target_modules"  # the accepted alias (load_config)
            if not isinstance(node, dict) or part not in node:
                missing.append(field)
                break
            node = node[part]
    if missing:
        raise ValueError("%s does not state %s, which the %s stage reads; every value a run uses must be in its "
                         "config file (configs/README.md)" % (path, ", ".join(missing), stage))


def verify_dataset(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError("Dataset changed: " + name)
    return manifest
