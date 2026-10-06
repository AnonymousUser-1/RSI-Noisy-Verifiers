from __future__ import annotations

"""The frozen base model pin, checked against the resolved configuration.

The study is pinned to `Qwen/Qwen3-1.7B` at one commit.  Without this module that
pin would live only in prose: every entry reached the hub through `pin_config`, which
turns whatever the config says into a resolved revision, and nothing compared the
result with the frozen value.  A config could therefore carry a legal 40-hex id
for a *different* commit, resolve to it without complaint, and every record would
still look pinned -- the revision is a commit id, `verify_shared_adapter` accepts
it, and the parameter hash says nothing about which base the adapter sits on.

`configs/pins/base_pin.json` is that frozen value as data, written once and
committed with this module.  It is prepared **before** any preflight and is never
written, updated or backfilled by a run: the value a run checks against is
therefore not one the run could have produced.  A pin that a run can rewrite is
not a pin.

Why the check is here and not inside `pin_config`
-------------------------------------------------
`rsi/experiment.py::pin_config` is called by entries that never load a model
(the CPU end-to-end path), and `rsi/backends.py` leaves revision resolution alone.  This module is the narrow carrier: it reads the frozen file and
compares, and each entry decides whether it is on the real-model path.  It
imports `require_commit_id` from `rsi.shared_adapter`, which is torch-free at
module level, so the pin can be checked without a GPU box.

Strict, with no normalization
-----------------------------
The comparison is on the **resolved** revision and is exact: no `strip()`, no
`lower()`, no alias handling.  `require_commit_id` is imported rather than
reimplemented, so the rule for "is this a commit id" stays in one place, and the
resolved value is put through it here as well -- a resolved revision that is not
a commit id is refused even if it happens to equal the pin's text.

Uppercase is refused, not repaired.  `resolve_revision` (pinned) applies
`.lower()` before its own check, so `"70D244CC..."` passes through it as typed;
`require_commit_id` then rejects it.  That asymmetry is deliberate and is
recorded in the a-3 receipt rather than papered over with a normalization here,
because normalizing would hide a caller that passes a revision that is not the
one the file spells.
"""
import json
import os
from pathlib import Path

from .common import file_hash
from .shared_adapter import require_commit_id

# `configs/` sits beside `rsi/`, at the repository root.
BASE_PIN = Path(__file__).resolve().parents[1] / "configs" / "pins" / "base_pin.json"

REQUIRED_KEYS = ("model", "revision")


def default_pin_path():
    """The study's pin, or the pin file RSI_BASE_PIN names.

    The same protocol run on another base model (e.g. Qwen3-4B) is checked against that model's own committed pin file, named
    explicitly, instead of an edit to the frozen one.  check_base_pin records
    the path and sha256 of whichever pin it used.
    """
    override = os.environ.get("RSI_BASE_PIN")
    return Path(override) if override else BASE_PIN


def read_base_pin(path=None):
    """Read the frozen pin.  A malformed pin file is an error, never a skip.

    The pin is the answer to "which base is this study on?", so a file that does
    not answer it must stop the run rather than let the run continue unchecked.
    `revision` is held to the same rule as a resolved revision, so an alias
    ("main"), a truncated id or a non-string cannot serve as an anchor.
    """
    path = Path(default_pin_path() if path is None else path)
    if not path.exists():
        raise ValueError("base pin %s does not exist; it is a frozen input of this study, not "
                         "something a run produces" % path)
    try:
        pin = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("base pin %s is not valid JSON: %s" % (path, exc)) from exc
    if not isinstance(pin, dict):
        raise ValueError("base pin %s must be a JSON object, not %s" % (path, type(pin).__name__))
    missing = [key for key in REQUIRED_KEYS if key not in pin]
    if missing:
        raise ValueError("base pin %s is missing %s" % (path, ", ".join(missing)))
    model = pin["model"]
    if not isinstance(model, str) or not model:
        raise ValueError("base pin %s: model must be a non-empty string, not %r" % (path, model))
    try:
        require_commit_id(pin["revision"], what="base pin %s revision" % path)
    except ValueError as exc:
        raise ValueError("base pin %s is not usable as an anchor: %s" % (path, exc)) from exc
    return pin


def check_base_pin(config, pin=None, path=None):
    """Refuse a resolved configuration whose base is not the frozen pin.

    `config` is the **pinned** configuration (`pin_config`'s output), so
    `config["revision"]` is what the hub resolved, not the alias the file asked
    for.  Returns the binding to record beside the artefact:

        {"pin_path", "pin_sha256", "model", "pin_revision", "resolved_revision"}

    The binding names the pin file and its own hash, and deliberately does not
    carry a hash of itself, so recording the binding does not invalidate it.  It
    records both sides of the comparison -- the pin's revision and the resolved
    one -- so a later reader can see that the run resolved to the frozen commit
    rather than take the pin file alone as the evidence.

    Raises ValueError on any mismatch, naming both sides -- an error that says
    only "revision mismatch" leaves the reader to guess which of the two values
    is the wrong one.
    """
    path = Path(default_pin_path() if path is None else path)
    pin = read_base_pin(path) if pin is None else pin
    model = config.get("model")
    # The resolved value is held to the commit-id rule here too; `pin_config`
    # only guarantees that `resolve_revision` returned, not that what it
    # returned is an id.
    resolved = require_commit_id(config.get("revision"),
                                 what="resolved revision of %s" % config.get("backend"))
    if pin["revision"] != resolved:
        raise ValueError(
            "base pin mismatch: %s pins %s to the revision %s, but the resolved configuration "
            "resolved to %s; this run would train on a different base than the frozen one"
            % (path, pin["model"], pin["revision"], resolved))
    if pin["model"] != model:
        raise ValueError(
            "base pin mismatch: %s pins the base model %r, but the resolved configuration names "
            "%r; the revision matches only by coincidence of text"
            % (path, pin["model"], model))
    return {"pin_path": str(path), "pin_sha256": file_hash(path),
            "model": pin["model"], "pin_revision": pin["revision"], "resolved_revision": resolved}
