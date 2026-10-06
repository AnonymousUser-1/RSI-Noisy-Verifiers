"""Which data split a pool is drawn from: named on the command line, never guessed.

sample_candidates.py (P) used to read train_001.jsonl whatever the run was, so a
pilot -- which RUN_COST_SPEC 5 runs on development data -- could only reach dev
by renaming dev.jsonl or by pointing --data at a doctored directory.  Both hide
which rows were sampled.  The split is now an explicit `--split NAME`, and this
module is the one place that says which names may be sampled and which split
each phase may read, so P and run_matched_experiment.py apply one rule
(science-aux 5d124fd6).

The rule
  * Only `dev` and `train_NNN` may be sampled, on every backend.  calibration,
    eval_id, eval_ood and gradient_reference never enter a pool: calibration
    fits the verifier, the eval splits are the held-out test, and the 64
    gradient_reference rows are diagnostic only.
    It is an allow-list, so a split added later is refused until it is named
    here.
  * On hf, `--split`, `--study-id` and `--phase` are required, and the phase
    fixes the split: pilot reads dev only, main reads train_NNN only.  The
    split is never inferred from the phase; a mismatch is refused before
    anything is resolved or loaded, not corrected.
  * On mock, existing calls keep working: without --split the pool is drawn
    from train_001 as before, and the record says so ("mock default").  The
    phase/split interlock is ruled for hf (5d124fd6) and is not applied to mock.

`read_split` hashes the bytes it parses, so the sha256 a meta records is the
hash of the rows that were actually sampled, and it refuses a file that no
longer matches the manifest entry `verify_dataset` checked a moment before.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

from .common import digest

PHASES = ("pilot", "main")
MOCK_DEFAULT_SPLIT = "train_001"
NEVER_SAMPLED = ("calibration", "eval_id", "eval_ood", "gradient_reference")
_TRAIN = re.compile(r"train_[0-9]{3}")


def is_train_split(name):
    """Whether `name` is one of the formal training splits train_NNN."""
    return isinstance(name, str) and _TRAIN.fullmatch(name) is not None


def check_selection(split, study_id, phase, hf):
    """The split this run samples and the identity it records, or ValueError.

    Returns {"split": {"name", "source"}, "identity": {"study_id", "phase",
    "missing"}}.  "source" is "cli" for a split given with --split and
    "mock default" for mock's train_001.  "missing" gives the reason for each
    identity field left out, which only mock may do (RUN_COST_SPEC 1: never
    guessed, so a missing one is recorded as null with its reason).
    """
    if split is not None and split != "dev" and not is_train_split(split):
        if split in NEVER_SAMPLED:
            reason = ("calibration, eval_id, eval_ood and gradient_reference never enter a pool: calibration "
                      "fits the verifier, the eval splits are the held-out test, gradient_reference is "
                      "diagnostic only")
        else:
            reason = "only dev and train_NNN (three digits) can be sampled"
        raise ValueError("--split %r cannot be sampled: %s" % (split, reason))
    if study_id is not None and (not study_id.strip() or study_id != study_id.strip()):
        raise ValueError("--study-id %r is empty or has surrounding whitespace" % study_id)
    if phase is not None and phase not in PHASES:
        raise ValueError("--phase must be one of %s, not %r" % (", ".join(PHASES), phase))
    if hf:
        for flag, value in (("--split", split), ("--study-id", study_id), ("--phase", phase)):
            if value is None:
                raise ValueError("%s is required on hf: the split a pool is drawn from, and the study and "
                                 "phase it belongs to, are never guessed (RUN_COST_SPEC 1 and 5)" % flag)
        if phase == "pilot" and split != "dev":
            raise ValueError("--phase pilot reads dev only, not %s: a pilot runs on development data "
                             "(RUN_COST_SPEC 5, science-aux 5d124fd6)" % split)
        if phase == "main" and not is_train_split(split):
            raise ValueError("--phase main reads train_NNN only, not %s: a main run uses the formal "
                             "training splits (RUN_COST_SPEC 5, science-aux 5d124fd6)" % split)
    missing = {}
    if study_id is None:
        missing["study_id"] = "not given (--study-id); optional on mock only, never guessed from a path"
    if phase is None:
        missing["phase"] = "not given (--phase); optional on mock only, pilot or main is not guessed"
    selected = {"name": split, "source": "cli"} if split is not None else \
        {"name": MOCK_DEFAULT_SPLIT, "source": "mock default"}
    return {"split": selected, "identity": {"study_id": study_id, "phase": phase, "missing": missing}}


def read_split(root, manifest, name):
    """The rows of `<name>.jsonl` under `root`, and the binding a meta records for them.

    `manifest` is what `verify_dataset(root)` returned.  The file must be one
    the manifest lists -- a file it does not list has no hash vouching for it --
    and the bytes read here must still hash to the manifest's entry.
    """
    file = name + ".jsonl"
    if file not in manifest["files"]:
        raise ValueError("--split %s: %s is not listed in the manifest at %s, so no hash vouches for its rows"
                         % (name, file, Path(root)))
    raw = (Path(root) / file).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if sha256 != manifest["files"][file]:
        raise ValueError("--split %s: %s changed after the manifest was verified: sha256 %s, the manifest "
                         "records %s" % (name, file, sha256, manifest["files"][file]))
    # Parsed as rsi.common.read_jsonl parses a file (default text decoding,
    # universal newlines, blank lines skipped), but from the bytes just hashed.
    rows = [json.loads(line) for line in io.TextIOWrapper(io.BytesIO(raw)) if line.strip()]
    return rows, {"name": name, "file": file, "sha256": sha256, "rows": len(rows)}


def dataset_binding(root, manifest):
    """The dataset a pool was drawn from: the manifest file's sha256 and what it records.

    `manifest` is what `verify_dataset(root)` returned; the manifest file is
    read again here, as bytes, and must still say the same thing.
    """
    path = Path(root) / "manifest.json"
    raw = path.read_bytes()
    if json.loads(raw) != manifest:
        raise ValueError("%s changed after it was verified" % path)
    return {"root": str(Path(root).resolve()), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "dataset_hash": digest(manifest), "task": manifest.get("task"), "data_seed": manifest.get("seed"),
            "version": manifest.get("version")}
