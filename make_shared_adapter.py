#!/usr/bin/env python3
"""Create the one shared LoRA initialisation every branch must start from.

`HANDOFF.md` deliverable B.1.  Writes `shared_adapter/` plus a record with the
parameter hash, seed, and base revision, so two branches can be shown to start
from identical parameters rather than merely to have been seeded alike.  See
`rsi/shared_adapter.py` for why a saved artefact is required and a seed is not
enough.

Requires a real backend: the point is to produce real adapter weights.  This is a
GPU-side artefact and is not part of the CPU end-to-end record.
"""
import argparse

from rsi.base_pin import check_base_pin
from rsi.common import load_config, require_explicit
from rsi.experiment import pin_config
from rsi.shared_adapter import create_shared_adapter

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="Output shared_adapter/ directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["hf", "mock"])
    a = p.parse_args()
    # Pinned, so the recorded base revision is the resolved commit and not "main".
    config = pin_config(load_config(a.config, a.seed, a.backend))
    # Checked through the backend gate below, never beside it: on mock nothing is
    # loaded and no base is used, and mock's current behaviour is exactly what it
    # was (the pin file is not read at all).  The adapter on the real path is
    # about to be initialised *at* a base, and `verify_shared_adapter` only ever
    # proves the record agrees with the adapter, never that the base is the one
    # this study pinned (rsi/base_pin.py).
    if config["backend"] == "mock":
        raise SystemExit("The shared adapter must be initialised from a real backend; "
                         "--backend mock would record a hash for nothing")
    try:
        check_base_pin(config)
        require_explicit(a.config, "adapter")  # every value it uses is in the config file
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    record = create_shared_adapter(config, a.out, a.seed)
    print("shared adapter at %s" % a.out)
    print("  parameter_hash: %s" % record["parameter_hash"])
    print("  base: %s @ %s" % (record["base_model"], record["base_revision"]))
    print("  rank/alpha/dropout/targets: %s/%s/%s/%s"
          % (record["protocol"]["rank"], record["protocol"]["alpha"],
             record["protocol"]["dropout"], record["protocol"]["target_modules"]))
    print("  adapter params: %d across %d tensors" % (record["num_params"], len(record["param_names"])))
