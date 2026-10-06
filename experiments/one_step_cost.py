#!/usr/bin/env python3
"""The cost of a one-step comparison, from its cost records (rsi/cost.py), in the paper's cost table form.

  python experiments/one_step_cost.py --records DIR [DIR ...] --pools POOLS_DIR --out COST.csv

Reads every record under the given directories (h: reference_one_step/cost_records; the arms:
out_one_step/cost_records; the evaluation: out_one_step/evaluation_cost_records), counts each
cost_id once, and writes one row per cost owner:

  Shared preparation/pool   h, and the round-1 pools (an input: drawn and counted by the multi-round
                            run, so 0 new responses here and its time not counted again)
  R only, S only            the arm's update
  Null and evaluation       the null arm, and every evaluation (the base, which null is, and R and S)
  Deduplicated joint total  the sum of the rows above

Columns: wall seconds, allocated GPU hours, new responses, new output tokens, and coverage: the number
of complete records summed, and "observed lower bound" when a summed record lacks a measurement (a
missing value is never read as 0).  A row with no complete record is left empty, not 0.  Failed attempts are not summed; they are counted per row and
listed in COST_failed.csv.  Memory peaks are per stage and are not added: COST_memory.csv lists the
largest allocated and reserved peak per stage.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.common import read_json  # noqa: E402
from rsi.cost import read_records  # noqa: E402

ROWS = (("Shared preparation/pool", ("shared",)), ("R only", ("R",)), ("S only", ("S",)),
        ("Null and evaluation", ("null", "evaluation")))
MEASURES = ("wall_seconds", "allocated_gpu_hours", "new_responses", "new_output_tokens")


def summarize(records):
    rows = []
    totals = {m: 0.0 for m in MEASURES}
    total_missing, total_complete, total_failed = set(), 0, 0
    for label, owners in ROWS:
        mine = [r for r in records.values() if r["owner"] in owners]
        complete = [r for r in mine if r["status"] == "complete"]
        failed = len(mine) - len(complete)
        row = {"cost_owner": label}
        missing = set()
        for m in MEASURES:
            values = [r[m] for r in complete]
            if not values or any(v is None for v in values):
                missing.add(m)
            row[m] = sum(v for v in values if v is not None) if values else None
            totals[m] += row[m] or 0.0
        row["coverage"] = ("no complete record: not measured" if not complete else "%d complete record(s)" % len(complete))             + ("; failed attempts not summed: %d" % failed if failed else "")             + ("; observed lower bound (missing: %s)" % ", ".join(sorted(missing)) if missing and complete else "")
        rows.append(row)
        total_missing |= missing
        total_complete += len(complete)
        total_failed += failed
    total = dict({"cost_owner": "Deduplicated joint total"}, **totals)
    total["coverage"] = "%d unique cost_id(s) summed%s%s" % (
        total_complete, "; failed attempts not summed: %d" % total_failed if total_failed else "",
        "; observed lower bound (missing: %s)" % ", ".join(sorted(total_missing)) if total_missing else "")
    rows.append(total)
    return rows


def pool_note(pools):
    """The reused round-1 pools: an input of the one-step run, with no new responses here."""
    metas = sorted(Path(pools).glob("b*.jsonl.meta.json"))
    return {"cost_owner": "Shared preparation/pool", "pools": len(metas),
            "pool_sha256": ";".join(read_json(m)["hash"] for m in metas),
            "note": "reused round-1 pools: counted by the multi-round run that drew them; not counted again"}


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", required=True, help="directories of cost records")
    p.add_argument("--pools", required=True, help="the round-1 pools the run read (their sha256 are listed)")
    p.add_argument("--out", required=True, help="the cost table CSV; _failed.csv, _memory.csv and _pools.csv beside it")
    a = p.parse_args(argv)
    records = read_records(*a.records)
    if not records:
        print("refused: no cost records under %s" % ", ".join(a.records), file=sys.stderr)
        return 1
    write(a.out, summarize(records))
    stem = str(Path(a.out).with_suffix(""))
    failed = [{k: r[k] for k in ("cost_id", "stage", "owner", "block", "failure", "wall_seconds")}
              for r in records.values() if r["status"] != "complete"]
    if failed:
        write(stem + "_failed.csv", failed)
    peaks = defaultdict(lambda: {"allocated_peak_bytes": None, "reserved_peak_bytes": None})
    for r in records.values():
        if r["memory"]:
            for k in ("allocated_peak_bytes", "reserved_peak_bytes"):
                peaks[r["stage"]][k] = max(peaks[r["stage"]][k] or 0, r["memory"][k])
    if peaks:
        write(stem + "_memory.csv", [dict({"stage": s}, **v) for s, v in sorted(peaks.items())])
    write(stem + "_pools.csv", [pool_note(a.pools)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
