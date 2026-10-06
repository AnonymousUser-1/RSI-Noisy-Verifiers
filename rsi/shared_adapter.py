from __future__ import annotations

"""One shared LoRA initialisation that every branch must start from.

`HANDOFF.md` deliverable B.1 asks for "one common LoRA initialisation (rank 8 on
q/v, recorded in config), with parameter hash, seed, and base revision", so that
"two branches cannot be shown to start from the same parameters" stops being a
gap.  This module is that artefact.

Why a saved directory rather than a seed alone
----------------------------------------------
A seed is not evidence.  Re-seeding a fresh adapter on each branch only proves
the two runs *intended* the same start; it does not prove they *got* the same
start, and it silently breaks the moment PEFT changes its internal init order,
the target-module resolution order changes, or one branch is resumed from a
different attempt directory.  A saved `adapter_model.safetensors` plus a
parameter hash proves it, because both branches load the same bytes.

The hash is taken over the adapter tensors, not over the file: `safetensors`
serialisation can legitimately differ in key order and header padding while
carrying identical parameters, so a file hash would report a false mismatch.
The file hash is recorded separately, as provenance.

The hash covers the adapter tensors only, never the base weights.  For a
PeftModel `state_dict()` is the whole model, so hashing it copied every base
weight to the CPU and tied the value to the base as a side effect; `h.json`
binds this value, so it has to mean the adapter and nothing else.

Base model and revision are recorded because the adapter is only meaningful
relative to the exact base checkpoint it will be attached to, and because the
hash leaves the base out they are the only binding to it.
`verify_shared_adapter` takes the model ID and revision the caller actually
loaded and refuses any difference from the record; `one_step._run_hf_arm`
passes its backend's.
"""

import hashlib
import re
from pathlib import Path

from .common import read_json, write_json


def adapter_parameter_hash(model) -> str:
    """Hash the adapter tensors themselves, bit-for-bit and order-independent.

    The adapter tensors are the parameters whose name contains "lora", the rule
    `lora_parameter_names` and the gradient recorder use, taken whether or not
    they are trainable, so an inference load of the same adapter hashes the
    same.  Base weights are never read.  Names are sorted, and each tensor
    contributes its name, shape, dtype and values, so any real change to the
    initialisation changes the digest, while a re-serialisation that only
    reorders keys does not.
    """
    import torch

    tensors = [(name, value.detach().cpu()) for name, value in model.named_parameters()
               if "lora" in name.lower()]
    if not tensors:
        raise ValueError("Model has no adapter (LoRA) parameters to hash")
    hasher = hashlib.sha256()
    for name, value in sorted(tensors, key=lambda item: item[0]):
        hasher.update(name.encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(str(value.dtype).encode())
        # numpy has no bfloat16, so the values are widened to float64, which is exact
        # for every float dtype; the dtype is hashed above, so the same values in
        # another precision give another digest.
        hasher.update(value.to(torch.float64).numpy().tobytes())
    return hasher.hexdigest()


def lora_parameter_names(model) -> list:
    """Trainable LoRA parameter names, in model order, for the run record."""
    return [name for name, param in model.named_parameters() if "lora" in name.lower() and param.requires_grad]


def require_commit_id(revision, what="base_revision"):
    """Refuse anything but a 40-character lowercase hex commit id, and return it.

    The adapter hash covers the adapter tensors and never the base, so the base
    model id and revision recorded beside it are the only binding to the base
    this adapter will be attached to, and `h.json` binds that record.  An alias
    breaks the binding: `"main" == "main"` compares equal in the record and at
    load time while pointing at whatever commit the hub serves that day, so the
    comparison passes on two different bases.  A commit id names one immutable
    object, which is what the record needs.

    `resolve_revision` (pinned) already returns a commit id unchanged when it is
    given one, but it applies `.lower()` first, so `"ABCD...40"` is taken for an
    id and passes through as typed; and for any other string it resolves the
    alias over the network, so a typo becomes a lookup rather than an error.
    This check is the narrower one the record needs.

    Raises ValueError, not TypeError, for a missing or non-string value: every
    caller of this module raises and catches ValueError, so a TypeError would
    escape the `except ValueError` in `one_step._run_hf_arm` and lose the branch
    name it adds on the way out.
    """
    if not isinstance(revision, str):
        raise ValueError("%s must be a 40-character lowercase hex commit id, not %r"
                         % (what, revision))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("%s must be a 40-character lowercase hex commit id, not %r; "
                         "pin the revision instead of passing an alias such as 'main'"
                         % (what, revision))
    return revision


def init_shared_adapter(model, config, seed=0):
    """Attach a fresh LoRA adapter to `model` and return its metadata (not saved).

    Kept separate from :func:`create_shared_adapter` so a caller that already has
    a model in memory (a test, or a run that reuses the base weights) can build
    the same adapter without a second load.
    """
    # Before anything is built or imported: this is the value that goes into the
    # record, and the record is the only binding to the base.
    require_commit_id(config["revision"])
    import torch
    from peft import get_peft_model

    from .backends import lora_config_from_training

    torch.manual_seed(seed)
    training = config["training"]
    wrapped = get_peft_model(model, lora_config_from_training(training))
    names = lora_parameter_names(wrapped)
    if not names:
        raise ValueError("No trainable adapter parameters were created; check target_modules")
    return {
        "parameter_hash": adapter_parameter_hash(wrapped),
        "param_names": names,
        "num_params": sum(int(wrapped.state_dict()[n].numel()) for n in names),
        "seed": int(seed),
        "base_model": config["model"],
        "base_revision": config["revision"],
        "protocol": lora_protocol(training),
    }, wrapped


def create_shared_adapter(config, output, seed=0):
    """Build the shared adapter from the config and save it to `output`.

    `config` must already be pinned (`experiment.pin_config`), because the base
    revision written into the record has to be the resolved commit, not the
    `"main"` alias the caller typed.  The pinned revision is then required to be
    a commit id in its own right (:func:`require_commit_id`) before the backend
    loads anything, so a config that skipped pinning is refused instead of
    recording an alias that compares equal to itself.
    """
    from .backends import backend

    require_commit_id(config["revision"])
    model = backend(config)  # inference-shaped load; no adapter, not trainable
    try:
        record, wrapped = init_shared_adapter(model.model, config, seed)
        output = Path(output)
        wrapped.save_pretrained(output)
        record["path"] = str(output)
        record["config_hash"] = config.get("config_hash")
        write_json(output / "shared_adapter.json", record)
    finally:
        model.close()
    return record


def lora_protocol(training):
    """The LoRA settings a shared adapter records, from a config's `training` block."""
    return {"rank": training["lora_rank"], "alpha": training["lora_alpha"], "dropout": training["lora_dropout"],
            "target_modules": list(training["target_modules"]), "use_rslora": False}


def read_shared_record(adapter_dir):
    """Read the recorded metadata for a shared adapter without touching a model."""
    path = Path(adapter_dir) / "shared_adapter.json"
    if not path.exists():
        raise FileNotFoundError("No shared adapter record at %s; create one before training" % path)
    return read_json(path)


def lora_mismatch(record, training):
    """The LoRA settings a config's `training` states that differ from the shared adapter's record.

    A run loads the saved adapter, not the config's LoRA values, so a config that states other
    values would record settings the run did not use (configs/README.md).
    """
    protocol = record.get("protocol") or {}
    return ["%s (adapter %r, config %r)" % (key, protocol.get(key), value)
            for key, value in lora_protocol(training).items() if protocol.get(key) != value]


def verify_shared_adapter(model, adapter_dir, base_model, base_revision):
    """Prove a model's adapter starts exactly where the shared artefact left it.

    `base_model` and `base_revision` are the model ID and revision the caller
    actually loaded the base from.  They are required: the parameter hash leaves
    the base out, so they are the only check that the adapter sits on the base
    it was initialised on.  Both the loaded revision and the one in the record
    must be commit ids (:func:`require_commit_id`), so two aliases that happen to
    spell the same string cannot pass as a match.

    Returns the stored record.  Raises ValueError on any mismatch, so a branch
    that quietly re-initialised (rather than loaded) fails loudly instead of
    producing a comparison that looks paired but is not.
    """
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        raise ValueError("Model is not wrapped in a PeftModel; the shared adapter was not loaded")
    require_commit_id(base_revision, "the loaded base_revision")
    record = read_shared_record(adapter_dir)
    if "base_revision" in record:
        require_commit_id(record["base_revision"], "the recorded base_revision")
    for key, loaded in (("base_model", base_model), ("base_revision", base_revision)):
        if key not in record:
            raise ValueError("The shared adapter record has no %s, so it is bound to no base" % key)
        if record[key] != loaded:
            raise ValueError("The shared adapter was initialised on %s %s, but this run loaded %s"
                             % (key, record[key], loaded))
    actual = adapter_parameter_hash(model)
    if actual != record["parameter_hash"]:
        raise ValueError("Loaded adapter parameters do not match the shared initialisation: "
                         "%s != %s" % (actual, record["parameter_hash"]))
    return record
