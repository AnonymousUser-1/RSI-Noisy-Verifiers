"""Cost records: one JSON file per physical stage attempt.

A stage attempt (one h, one arm's update, one checkpoint's evaluation on one split, ...) is wrapped
in `stage(...)`, which writes `<dir>/<cost_id>.json` when the attempt ends, however it ends:

  {"schema": "rsi-cost-record/1", "cost_id": ..., "stage": "reference" | "train" | "evaluate" | ...,
   "owner": "shared" | "R" | "S" | "null" | "evaluation", "block": ..., "detail": {...},
   "status": "complete" | "failed", "failure": null or "Type: message",
   "started_utc": ..., "ended_utc": ...,
   "wall_seconds": monotonic seconds, GPU work settled at both ends,
   "gpus": GPUs held for the stage (CUDA_VISIBLE_DEVICES; 0 on CPU),
   "allocated_gpu_hours": wall_seconds x gpus / 3600, null when gpus is unknown,
   "new_responses", "new_output_tokens": generated during this attempt, null when not measured,
   "memory": {"allocated_peak_bytes", "reserved_peak_bytes"} on CUDA, else null,
   "measured": the measurements this record has; "missing": those it lacks}

The rules are those the study reports costs by: every attempt has its own cost_id and failed
attempts are kept; time is monotonic and GPU work is settled at the stage boundaries; a stage
holding n GPUs for t seconds costs t*n/3600 allocated GPU hours; memory peaks are reset at the
start of a stage and are never added across stages; a measurement that was not taken is null and
listed in "missing", never 0.  Nested stages are not supported: a record covers one stage.
experiments/one_step_cost.py adds them up.
"""
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "rsi-cost-record/1"
OWNERS = ("shared", "R", "S", "null", "evaluation")


def _cuda():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _gpus(torch):
    if torch is None:
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return None   # every GPU is visible; the stage may hold any number of them
    return len([d for d in visible.split(",") if d.strip()])


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Attempt:
    """The record of one stage attempt; the stage sets the counts it measures."""

    def __init__(self, cost_id, stage, owner, block, detail):
        self.cost_id, self.stage, self.owner, self.block, self.detail = cost_id, stage, owner, block, detail
        self.new_responses = None
        self.new_output_tokens = None

    def generated(self, responses, output_tokens):
        """Responses and output tokens this attempt generated (0 and 0 for a stage that generates none)."""
        self.new_responses = (self.new_responses or 0) + int(responses)
        self.new_output_tokens = (self.new_output_tokens or 0) + int(output_tokens)


def new_cost_id(stage):
    return "%s-%s-%s" % (stage, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:12])


@contextmanager
def stage(directory, stage_name, owner, block=None, detail=None):
    """Time one stage attempt and write its record to directory/<cost_id>.json when it ends."""
    if owner not in OWNERS:
        raise ValueError("cost owner must be one of %s, not %r" % (", ".join(OWNERS), owner))
    torch = _cuda()
    attempt = Attempt(new_cost_id(stage_name), stage_name, owner, block, detail or {})
    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started_utc, start = _utc(), time.monotonic()
    failure = None
    try:
        yield attempt
    except BaseException as exc:
        failure = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        memory = None
        if torch is not None:
            torch.cuda.synchronize()
            memory = {"allocated_peak_bytes": int(torch.cuda.max_memory_allocated()),
                      "reserved_peak_bytes": int(torch.cuda.max_memory_reserved())}
        wall = time.monotonic() - start
        gpus = _gpus(torch)
        record = {"schema": SCHEMA, "cost_id": attempt.cost_id, "stage": stage_name, "owner": owner,
                  "block": block, "detail": attempt.detail,
                  "status": "failed" if failure else "complete", "failure": failure,
                  "host": socket.gethostname(), "pid": os.getpid(),
                  "started_utc": started_utc, "ended_utc": _utc(),
                  "wall_seconds": wall, "gpus": gpus,
                  "allocated_gpu_hours": None if gpus is None else wall * gpus / 3600.0,
                  "new_responses": attempt.new_responses, "new_output_tokens": attempt.new_output_tokens,
                  "memory": memory}
        record["measured"] = sorted(k for k in ("allocated_gpu_hours", "new_responses", "new_output_tokens",
                                                "memory") if record[k] is not None) + ["wall_seconds"]
        record["missing"] = sorted(k for k in ("allocated_gpu_hours", "new_responses", "new_output_tokens",
                                               "memory") if record[k] is None)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (attempt.cost_id + ".json")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


def read_records(*directories):
    """Every cost record under the given directories, keyed by cost_id (a cost_id seen twice is one record)."""
    found = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")) if Path(directory).is_dir() else ():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema") != SCHEMA:
                continue
            old = found.get(record["cost_id"])
            if old is not None and old != record:
                raise ValueError("cost_id %s has two different records" % record["cost_id"])
            found[record["cost_id"]] = record
    return found
