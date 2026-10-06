#!/usr/bin/env python3
"""Render one final-accuracy phase diagram from simulate.py output."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--p0", type=float, default=0.5)
    p.add_argument("--samples", type=int, default=512)
    a = p.parse_args()
    final = {}
    with open(a.input) as stream:
        for r in csv.DictReader(stream):
            if float(r["p0"]) == a.p0 and int(r["samples"]) == a.samples:
                key = (float(r["tpr"]), float(r["fpr"]), int(r["seed"]))
                if key not in final or int(r["round"]) > int(final[key]["round"]):
                    final[key] = r
    if not final:
        p.error("No matching simulator cells")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for (tpr, fpr, seed), r in final.items():
        groups[(tpr, fpr)].append(float(r["accuracy"])-a.p0)
    ys, xs = sorted({k[0] for k in groups}), sorted({k[1] for k in groups})
    values = np.array([[np.mean(groups[(y, x)]) if (y, x) in groups else np.nan for x in xs] for y in ys])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(values, origin="lower", aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Final accuracy minus initial accuracy")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
