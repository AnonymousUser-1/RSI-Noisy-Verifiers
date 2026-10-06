#!/usr/bin/env python3
"""Generate one identical initial candidate pool for all matched-error branches.

The pool is drawn from the split named by --split: dev or train_NNN, a file the
dataset manifest lists (rsi/split_selection.py).  On hf, --split, --study-id and
--phase are required and must agree: pilot reads dev only, main reads train_NNN
only.  On mock an omitted --split draws from train_001 as before, and the meta
says so.

<out>.meta.json records what this run actually read and used: the split file,
its sha256 and row count; the dataset manifest's sha256; the CLI seed and the
seed generation received; the configuration in effect, the decoding and the
base pin; the code; the study and phase; and the command line.
"""
import argparse
import os
import sys
from pathlib import Path

from rsi.base_pin import check_base_pin
from rsi.common import file_hash, load_config, require_explicit, verify_dataset
from rsi.experiment import pin_config, sample_pool
from rsi.split_selection import PHASES, check_selection, dataset_binding, read_split


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["hf", "mock"])
    p.add_argument("--split", help="The split to sample: dev or train_NNN, listed in the manifest.  Required "
                                   "on hf; on mock, train_001 when omitted")
    p.add_argument("--study-id", help="The study this pool belongs to.  Required on hf")
    p.add_argument("--phase", choices=PHASES, help="pilot (reads dev) or main (reads train_NNN).  Required on hf")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    requested = load_config(args.config, args.seed, args.backend)
    # Which rows, for which study and phase, is settled from the command line
    # before anything is resolved, read or loaded: a pilot pointed at train_001
    # is refused here, not found in its meta afterwards (science-aux 5d124fd6).
    try:
        selection = check_selection(args.split, args.study_id, args.phase, requested["backend"] == "hf")
    except ValueError as exc:
        raise SystemExit("%s.  Nothing was written." % exc) from exc
    config = pin_config(requested)
    if config["backend"] == "hf":
        # The pool is drawn from the base the study is pinned to; a pool drawn
        # from another commit is not a pool for this study (rsi/base_pin.py).
        # Mock keeps its current behaviour exactly: nothing is loaded, so there
        # is nothing to check and the pin file is not read at all.
        # Before `verify_dataset`, not after: the pin says which base this run is
        # on, and a run whose base is wrong should say so rather than report a
        # manifest first.  Same order as compute_reference_gradient.py.
        try:
            base_pin = dict(check_base_pin(config), status="checked")
            # Every value the pool is drawn with comes from the config file (configs/README.md).
            require_explicit(args.config, "pool")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        base_pin = {"status": "not read", "reason": "mock loads no base model, so there is no base to check"}
    manifest = verify_dataset(args.data)
    try:
        tasks, split = read_split(args.data, manifest, selection["split"]["name"])
        data = dataset_binding(args.data, manifest)
    except ValueError as exc:
        raise SystemExit("%s.  Nothing was written." % exc) from exc
    split["source"] = selection["split"]["source"]
    limit = config["generation"]["prompts_per_pool"]
    if limit is not None:
        if limit > len(tasks):
            raise SystemExit("generation.prompts_per_pool is %d, but %s has only %d prompts.  Nothing was written."
                             % (limit, split["name"], len(tasks)))
        tasks = tasks[:limit]
    split["prompts_sampled"] = len(tasks)
    sample_pool(config, tasks, args.out, args.seed, inputs={
        "split": split, "data": data, "base_pin": base_pin,
        "config_file": {"path": str(Path(args.config).resolve()), "sha256": file_hash(args.config)},
        "requested_config": requested,
        "identity": selection["identity"],
        "argv": {"executable": sys.executable, "script": str(Path(__file__).resolve()), "cwd": os.getcwd(),
                 "args": [str(a) for a in (sys.argv[1:] if argv is None else argv)], "parsed": vars(args)}})


if __name__ == "__main__":
    main()
