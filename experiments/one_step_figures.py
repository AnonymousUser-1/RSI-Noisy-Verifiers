#!/usr/bin/env python3
"""Figures of a one-step run that has passed its acceptance checks (experiments/one_step_check.py).

  python experiments/one_step_figures.py --run OUT_ONE_STEP --label LABEL --out PREFIX

Reads the run's diagnostics.json and evaluation_greedy_SPLIT/ results; computes nothing new.  Writes,
each figure as PREFIX_NAME.pdf (vector) and PREFIX_NAME.png (300 dpi), with the plotted numbers in
PREFIX_NAME.json:
  projection    h^T Delta theta of R and S per block, paired by block; S - R per block with its mean
                and 95% t interval.  null is 0 by construction (no update).
  displacement  |Delta theta|_2 of R and S per block, paired by block.
  gradients     the per-example gradient contributions ||g_i||/K, by arm and by correct/error example.
                R and S train on the same correct examples, so their correct groups coincide and any
                difference is in the error examples.
  error         per evaluated split: the error rate (1 - greedy Pass@1) of R and S minus null's (the
                base), per block, paired by block; S - R with its mean and 95% t interval.
Intervals are Student-t over the blocks (plot_multiround.interval); with n blocks, t(0.975, n - 1).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.one_step_check import blocks, evaluated, pass1  # noqa: E402
from plot_multiround import COLORS, interval  # noqa: E402
from rsi.common import read_json  # noqa: E402

UPDATED = ("R", "S")
DPI = 300


def diagnostics(run, block, arm):
    return read_json(Path(run) / block / arm / "round_001" / "diagnostics.json")


def paired(values):
    """values: {block: {"R": x, "S": y}} -> the per-block S - R and its interval."""
    diffs = {b: v["S"] - v["R"] for b, v in values.items() if v.get("R") is not None and v.get("S") is not None}
    return {"per_block": diffs, "s_minus_r": interval(list(diffs.values())) if diffs else None}


def collect(run):
    run = Path(run)
    names = blocks(run)
    data = {"blocks": names,
            "projection": {b: {a: diagnostics(run, b, a)["main_diagnostic"]["h_T_delta_theta"] for a in UPDATED}
                           for b in names},
            "displacement": {b: {a: diagnostics(run, b, a)["delta_theta"]["norm"] for a in UPDATED} for b in names},
            "gradients": {a: {"correct": [], "error": []} for a in UPDATED},
            "error": {}}
    for b in names:
        for a in UPDATED:
            for sample in diagnostics(run, b, a)["per_sample"]:
                data["gradients"][a]["correct" if sample["correct"] else "error"].append(sample["norm"])
    for split, directory in evaluated(run).items():
        base = {b: pass1(directory, b, "null") for b in names}
        data["error"][split] = {b: {a: (None if pass1(directory, b, a) is None or base[b] is None
                                        else base[b] - pass1(directory, b, a)) for a in UPDATED} for b in names}
    data["projection_paired"] = paired(data["projection"])
    data["displacement_paired"] = paired(data["displacement"])
    data["error_paired"] = {split: paired(v) for split, v in data["error"].items()}
    return data


def save(fig, prefix, name, numbers):
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": DPI})):
        fig.savefig("%s_%s.%s" % (prefix, name, ext), bbox_inches="tight", **kwargs)
    Path("%s_%s.json" % (prefix, name)).write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    print("wrote %s_%s.pdf/.png/.json" % (prefix, name))


def draw_paired(ax_pairs, ax_diff, values, summary, ylabel, zero_line):
    names = sorted(values)
    for i, b in enumerate(names):
        ax_pairs.plot([0, 1], [values[b]["R"], values[b]["S"]], color="grey", linewidth=0.8, zorder=1)
    for x, arm in enumerate(UPDATED):
        ax_pairs.scatter([x] * len(names), [values[b][arm] for b in names], color=COLORS[arm], zorder=2, label=arm)
    ax_pairs.set_xticks([0, 1], list(UPDATED))
    ax_pairs.set_xlim(-0.5, 1.5)
    ax_pairs.set_ylabel(ylabel)
    if zero_line:
        ax_pairs.axhline(0, color="k", linewidth=0.8, linestyle=":", label="null (no update)")
    # Below the axes, so it never covers a block's point.
    ax_pairs.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    if ax_diff is None or summary["s_minus_r"] is None:
        return
    diffs = summary["per_block"]
    ax_diff.scatter(range(len(diffs)), [diffs[b] for b in sorted(diffs)], color="k", s=14)
    s = summary["s_minus_r"]
    ax_diff.errorbar([len(diffs) + 0.5], [s["mean"]], yerr=[s["half_width"]], color="k", marker="D", capsize=4)
    ax_diff.set_xticks(list(range(len(diffs))) + [len(diffs) + 0.5], sorted(diffs) + ["mean"], fontsize=8)
    ax_diff.axhline(0, color="grey", linewidth=0.8)
    ax_diff.set_ylabel("S - R")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the one-step run's --out directory")
    p.add_argument("--label", required=True, help="the experiment's name on the figures")
    p.add_argument("--out", required=True, help="output path prefix")
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt

    data = collect(a.run)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = len(data["blocks"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    draw_paired(left, right, data["projection"], data["projection_paired"], r"$h^\top \Delta\theta$", True)
    fig.suptitle("%s: projection of the one-step update on h (%d blocks)" % (a.label, n), fontsize=10)
    save(fig, a.out, "projection", {"per_block": data["projection"], "paired": data["projection_paired"]})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    draw_paired(ax, None, data["displacement"], data["displacement_paired"], r"$\|\Delta\theta\|_2$", False)
    fig.suptitle("%s: one-step displacement (%d blocks)" % (a.label, n), fontsize=10)
    save(fig, a.out, "displacement", {"per_block": data["displacement"], "paired": data["displacement_paired"]})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    groups = [(arm, kind) for arm in UPDATED for kind in ("correct", "error")]
    parts = ax.boxplot([data["gradients"][arm][kind] for arm, kind in groups], patch_artist=True,
                       medianprops={"color": "k"})
    for patch, (arm, kind) in zip(parts["boxes"], groups):
        patch.set_facecolor(COLORS[arm])
        patch.set_alpha(0.35 if kind == "correct" else 0.8)
    ax.set_xticks(range(1, len(groups) + 1), ["%s %s" % g for g in groups], fontsize=8)
    ax.set_ylabel(r"$\|g_i\|/K$ (per-example contribution)")
    fig.suptitle("%s: per-example gradient contributions, all blocks" % a.label, fontsize=10)
    save(fig, a.out, "gradients", {"groups": {"%s_%s" % g: data["gradients"][g[0]][g[1]] for g in groups}})
    plt.close(fig)

    for split, values in data["error"].items():
        fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
        draw_paired(left, right, values, data["error_paired"][split], "error - null error", True)
        fig.suptitle("%s: %s error change against null (%d blocks)" % (a.label, split, n), fontsize=10)
        save(fig, a.out, "error_" + split, {"per_block": values, "paired": data["error_paired"][split]})
        plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
