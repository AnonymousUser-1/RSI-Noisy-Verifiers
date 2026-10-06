#!/usr/bin/env python3
"""Download the pinned raw files of GSM8K or DeepMind Mathematics and check their sha256.

Needs the network, so run it in a job (srun/sbatch), never inside a GPU allocation.  A file that
is already present with the pinned hash is kept.  import_data.py then works offline.
"""
import argparse
import os
import shutil
import urllib.request
from pathlib import Path

from rsi.common import file_hash
from rsi.external import SOURCES
from rsi.tasks import IMPORTED_TASKS


def fetch(kind, raw):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    for name, pinned in sorted(SOURCES[kind]["files"].items()):
        path = raw / name
        if path.is_file() and file_hash(path) == pinned["sha256"]:
            print("present", path)
            continue
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(pinned["url"], timeout=60) as response, open(partial, "wb") as stream:
            shutil.copyfileobj(response, stream, 1 << 20)
        actual = file_hash(partial)
        if actual != pinned["sha256"]:
            partial.unlink()
            raise SystemExit("%s has sha256 %s, not the pinned %s" % (pinned["url"], actual, pinned["sha256"]))
        os.replace(partial, path)
        print("fetched", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=IMPORTED_TASKS, required=True)
    p.add_argument("--raw", required=True, help="Directory for the raw files, e.g. data/raw/gsm8k")
    a = p.parse_args()
    fetch(a.task, a.raw)
