#!/usr/bin/env python3
"""Import an experiment's round-1 pools, shared adapter and data from an export, byte for byte.

  python experiments/import_pools.py EXPERIMENT SOURCE_DIR

For an experiment whose settings.sh permits importing (POOLS_IMPORT_ALLOWED=yes or POOLS_IMPORTED=yes):
its round-1 pools were drawn under an
earlier sampling configuration and a multi-round run on them already exists; the one-step comparison
(experiments/one_step.sh) runs on the same pools.  SOURCE_DIR holds the data (TASK/, with
manifest.json), pools/bNN.jsonl with .jsonl.meta.json for each seed in SEEDS, shared_adapter/, and
matching/matched_subsets.json (that multi-round run's round-1 matching on those pools).  Every file is
copied unchanged; no metadata is rewritten.  A pool's metadata keeps the generation settings it was
drawn with, and the experiment's config.json states exactly those (experiments/pool_check.py checks
it again before training).

Refused, with nothing written, when:
  * a pool does not hash to its metadata, or was not drawn with config.json's base, dtype, device and
    every generation field, with the block's seed, from train_001 of exactly SOURCE_DIR's data;
  * the shared adapter records another base or other LoRA settings than config.json;
  * the experiment already holds pools, a shared adapter, data or a round-1 reference that differ.
Writes $RSI_ROOT/data/DATA_NAME/, $WORK/pools/, $WORK/shared_adapter/,
$WORK/round1_reference/matched_subsets.json and $WORK/pools/IMPORTED.json (the source, every file's
sha256 and the settings the pools were drawn with).
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.pool_check import differences  # noqa: E402
from rsi.common import file_hash, load_config, read_json, verify_dataset, write_json  # noqa: E402
from rsi.shared_adapter import lora_mismatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def settings(name):
    found = {}
    for line in (REPO / "experiments" / name / "settings.sh").read_text(encoding="utf-8").splitlines():
        m = re.match(r'^([A-Z_]+)=(.*)$', line)
        if m:
            found[m.group(1)] = m.group(2).strip().strip('"')
    return found


def same_tree(a, b):
    files = lambda root: {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(Path(root).rglob("*"))
                          if p.is_file()}
    return files(a) == files(b)


def plan(name, source, root):
    """The copies to make, or SystemExit naming what is wrong.  Writes nothing."""
    s = settings(name)
    if s.get("POOLS_IMPORTED") != "yes" and s.get("POOLS_IMPORT_ALLOWED") != "yes":
        raise SystemExit("%s/settings.sh does not set POOLS_IMPORTED=yes or POOLS_IMPORT_ALLOWED=yes; "
                         "its pools are drawn by experiments/run.sh" % name)
    work = Path(root) / "experiments" / name
    if not (work / "pools" / "IMPORTED.json").exists() and any(
            (work / child).exists() for child in
            ("pools", "shared_adapter", "reference", "out", "out_one_step", "one_step_config.json")):
        raise SystemExit("This experiment already has generated inputs or run artifacts. "
                         "Use a fresh RSI_ROOT for original-input replay. Nothing was written.")
    config_path = REPO / "experiments" / name / "config.json"
    config = load_config(config_path)
    source = Path(source).resolve()
    data_src = source / s["TASK"]
    try:
        verify_dataset(data_src)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit("%s is not a dataset whose files match its manifest: %s" % (data_src, e))
    seeds = [int(x) for x in s["SEEDS"].split()]
    problems = []
    for seed in seeds:
        pool = source / "pools" / ("b%02d.jsonl" % seed)
        problems += ["%s: %s" % (pool.name, d) for d in differences(config_path, data_src, pool, seed)]
    record = read_json(source / "shared_adapter" / "shared_adapter.json")
    if (record.get("base_model"), record.get("base_revision")) != (config["model"], config["revision"]):
        problems.append("shared adapter base %s@%s, config %s@%s" % (record.get("base_model"), record.get("base_revision"),
                                                                  config["model"], config["revision"]))
    problems += ["shared adapter %s" % d for d in lora_mismatch(record, config["training"])]
    if problems:
        raise SystemExit("Not imported, nothing written:\n  " + "\n  ".join(problems))
    work = root / "experiments" / name
    data_dst = root / "data" / s.get("DATA_NAME", s["TASK"])
    copies = [(data_src, data_dst), (source / "shared_adapter", work / "shared_adapter"),
              # The multi-round run's round-1 matching on these pools, which the one-step matching reproduces.
              (source / "matching" / "matched_subsets.json", work / "round1_reference" / "matched_subsets.json")]
    copies += [(source / "pools" / ("b%02d%s" % (seed, ext)), work / "pools" / ("b%02d%s" % (seed, ext)))
               for seed in seeds for ext in (".jsonl", ".jsonl.meta.json")]
    for src, dst in copies:
        if dst.exists() and not (same_tree(src, dst) if src.is_dir() else file_hash(src) == file_hash(dst)):
            raise SystemExit("%s exists and differs from %s; set it aside first.  Nothing written." % (dst, src))
    return copies, work, config, seeds, record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment")
    p.add_argument("source")
    a = p.parse_args(argv)
    root = Path(os.environ.get("RSI_ROOT", Path.home() / "rsi-work"))
    copies, work, config, seeds, record = plan(a.experiment, a.source, root)
    for src, dst in copies:
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dst)
    meta = read_json(work / "pools" / ("b%02d.jsonl.meta.json" % seeds[0]))
    write_json(work / "pools" / "IMPORTED.json", {
        "source": str(Path(a.source).resolve()),
        "files": {str(dst.relative_to(root)): (file_hash(dst) if dst.is_file() else "directory") for _, dst in copies},
        "pools_drawn_with": {"generation": meta.get("generation"), "code": meta.get("code"),
                             "model": meta.get("model"), "revision": meta.get("revision")},
        "shared_adapter_parameter_hash": record["parameter_hash"],
        "note": "copied byte for byte; no metadata rewritten"})
    print("imported %d pools, the shared adapter and the data into %s" % (len(seeds), work))
    return 0


if __name__ == "__main__":
    sys.exit(main())
