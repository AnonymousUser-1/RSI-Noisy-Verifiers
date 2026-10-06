#!/usr/bin/env python3
"""Was a round-1 pool drawn with this experiment's base and sampling?  Used by experiments/lib.sh.

  python experiments/pool_check.py --config CONFIG --data DATA --pool POOL --seed N

Exit 0 when POOL (with POOL.meta.json) was drawn on a real model (hf) from DATA's train_001 with
seed N and exactly CONFIG's model, revision, dtype, device and every generation field; exit 1,
saying what differs, otherwise.  lib.sh uses it twice: to reuse the sibling experiment's pool
instead of drawing the same one again, and before training, to refuse a pool drawn before
config.json's sampling was edited (the multi-round runner checks only a pool's base and data).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.common import digest, drawn_generation, file_hash, load_config, read_json, verify_dataset  # noqa: E402


def differences(config_path, data, pool, seed):
    pool = Path(pool)
    meta_path = Path(str(pool) + ".meta.json")
    if not pool.is_file() or not meta_path.is_file():
        return ["no pool or no meta file"]
    config = load_config(config_path)
    meta = read_json(meta_path)
    found = []
    if file_hash(pool) != meta.get("hash"):
        found.append("the pool file changed after its meta was written")
    if meta.get("backend") != "hf":
        found.append("backend %r, not hf" % meta.get("backend"))
    for key in ("model", "revision"):
        if meta.get(key) != config[key]:
            found.append("%s %r, config %r" % (key, meta.get(key), config[key]))
    for key in ("dtype", "device"):
        if (meta.get("config") or {}).get(key) != config[key]:
            found.append("%s %r, config %r" % (key, (meta.get("config") or {}).get(key), config[key]))
    drawn = drawn_generation(meta.get("generation"))
    changed = sorted(k for k in set(drawn) | set(config["generation"]) if drawn.get(k) != config["generation"].get(k))
    if changed:
        found.append("generation " + ", ".join("%s %r, config %r" % (k, drawn.get(k), config["generation"].get(k))
                                               for k in changed))
    if (meta.get("seed") or {}).get("cli") != seed:
        found.append("seed %r, not %d" % ((meta.get("seed") or {}).get("cli"), seed))
    if (meta.get("split") or {}).get("name") != "train_001":
        found.append("split %r, not train_001" % (meta.get("split") or {}).get("name"))
    if (meta.get("data") or {}).get("dataset_hash") != digest(verify_dataset(data)):
        found.append("another dataset (a prompt or the questions differ)")
    return found


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args(argv)
    found = differences(a.config, a.data, a.pool, a.seed)
    if found:
        print("%s does not match %s: %s" % (a.pool, a.config, "; ".join(found)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
