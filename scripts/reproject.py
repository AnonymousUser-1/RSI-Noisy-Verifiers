#!/usr/bin/env python3
"""Recompute h^T Delta theta and ||Delta theta|| for a chain of LoRA adapters, on a CPU.

    python scripts/reproject.py --h REFERENCE/h.pt --start SHARED_ADAPTER ROUND_1_ADAPTER [ROUND_2_ADAPTER ...]

Each round's displacement is Delta theta_t = theta_t - theta_{t-1}, with theta_0 the shared adapter
(`--start`), over the trainable LoRA tensors. This is the computation `run_iterative_experiment.py`
records as `training.h_T_delta_theta` and `training.delta_theta_norm` in each round's `complete.json`;
pass `--records` with the matching round directories to compare against them.

The release does not include trained adapters (only the Qwen3-4B shared adapter and its reference
gradients), so this script is for use with adapters obtained separately. Needs torch and safetensors.
"""
import argparse
import json
import math
from pathlib import Path


def load_adapter(path):
    from safetensors.torch import load_file
    path = Path(path)
    tensors = load_file(str(path / "adapter_model.safetensors" if path.is_dir() else path))
    return {canonical(k): v for k, v in tensors.items()}


def canonical(name):
    """PEFT saves `...lora_A.weight`; the recorded h uses the in-memory `...lora_A.default.weight`."""
    return name.replace(".default.", ".")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", required=True, help="h.pt (a dict of LoRA tensors)")
    ap.add_argument("--start", required=True, help="the shared adapter directory (theta_0)")
    ap.add_argument("adapters", nargs="+", help="round adapters in order (directories or .safetensors)")
    ap.add_argument("--records", nargs="*", help="round directories holding complete.json, in the same order")
    args = ap.parse_args()

    import torch
    h = {canonical(k): v for k, v in torch.load(args.h, weights_only=True).items()}
    previous = load_adapter(args.start)
    if set(previous) != set(h):
        raise SystemExit("h and the start adapter name different tensors")
    rows = []
    for i, adapter in enumerate(args.adapters):
        current = load_adapter(adapter)
        if set(current) != set(h):
            raise SystemExit("%s names different tensors than h" % adapter)
        delta = {k: current[k].double() - previous[k].double() for k in h}
        row = {"round": i + 1, "adapter": str(adapter),
               "h_T_delta_theta": sum(float((h[k].double() * delta[k]).sum()) for k in h),
               "delta_theta_norm": math.sqrt(sum(float((d ** 2).sum()) for d in delta.values()))}
        if args.records:
            record = json.loads((Path(args.records[i]) / "complete.json").read_text(encoding="utf-8"))["training"]
            row["recorded"] = {k: record[k] for k in ("h_T_delta_theta", "delta_theta_norm")}
        rows.append(row)
        previous = current
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
