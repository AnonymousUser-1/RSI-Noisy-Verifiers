#!/usr/bin/env python3
"""Figures and a table of Pass@1 per round for one or more multi-round runs.

  python plot_multiround.py --run OUT_A --label unaudited --run OUT_B --label "audited B=16" \
      --out figures/graph_qwen3-1.7b

Reads each run's evaluation_greedy_SPLIT/ (evaluate_multiround.py) and writes, for each split
(default: eval_id and eval_ood, those evaluated):
  PREFIX_SPLIT_pass1.png          (a) Pass@1 vs round, R and S, mean over seed blocks with a 95% t
                                  interval; (b) S - R per round, paired by block; (c) the target-error rate
  PREFIX_SPLIT_by_difficulty.png  Pass@1 vs round for each difficulty, R and S
  PREFIX_SPLIT_summary.json/.csv  the plotted numbers: per run, arm and round the mean, sd, n and
                                  interval, and S - R with its interval and paired t-test p-value
Round 0 is the base model, the same checkpoint for every block and arm.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ("R", "S")
COLORS = {"R": "#1f77b4", "S": "#d62728"}
STYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "D")


def interval(values):
    """Mean, sd, n and the 95% t-interval half-width (0 for a single value)."""
    from scipy.stats import t
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "sd": None, "n": n, "half_width": 0.0}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return {"mean": mean, "sd": sd, "n": n, "half_width": float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)}


def load(run, split):
    directory = Path(run) / ("evaluation_greedy_" + split)
    if not (directory / "protocol.json").exists():
        raise ValueError("%s has no evaluation: run evaluate_multiround.py --run %s first" % (directory, run))
    protocol = json.loads((directory / "protocol.json").read_text())
    base = json.loads((directory / "round_000.json").read_text()) if (directory / "round_000.json").exists() else None
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob("b*_round_*.json"))]
    return protocol, base, rows


def per_block(base, rows, metric):
    """{arm: {round: {block: value}}}; round 0 is the base's value for every block."""
    table = {arm: defaultdict(dict) for arm in ARMS}
    blocks = sorted({r["block"] for r in rows})
    for r in rows:
        table[r["arm"]][r["round"]][r["block"]] = metric(r)
    if base is not None:
        for arm in ARMS:
            for block in blocks:
                table[arm][0][block] = metric(base)
    return table


def summarize(base, rows, metric):
    from scipy.stats import ttest_rel
    table = per_block(base, rows, metric)
    rounds = sorted(set(table["R"]) | set(table["S"]))
    out = {"arms": {arm: {} for arm in ARMS}, "s_minus_r": {}}
    for arm in ARMS:
        for t in rounds:
            if table[arm][t]:
                out["arms"][arm][t] = dict(interval(list(table[arm][t].values())), blocks=sorted(table[arm][t]))
    for t in rounds:
        paired = sorted(set(table["R"][t]) & set(table["S"][t]))
        if not paired:
            continue
        diffs = [table["S"][t][b] - table["R"][t][b] for b in paired]
        entry = dict(interval(diffs), blocks=paired)
        if t > 0 and len(paired) > 1 and any(d != diffs[0] for d in diffs):
            entry["paired_t_p"] = float(ttest_rel([table["S"][t][b] for b in paired],
                                                  [table["R"][t][b] for b in paired]).pvalue)
        out["s_minus_r"][t] = entry
    return out


def draw_arms(ax, summary, style, marker, label):
    for arm in ARMS:
        points = sorted(summary["arms"][arm].items())
        if not points:
            continue
        x = [t for t, _ in points]
        y = [p["mean"] for _, p in points]
        e = [p["half_width"] for _, p in points]
        ax.errorbar(x, y, yerr=e, color=COLORS[arm], linestyle=style, marker=marker, capsize=3,
                    label="%s, %s" % (arm, label) if label else arm)


def plot(runs, out, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pass1 = lambda r: r["pass1"]
    target = lambda r: r["target_error_rate"]
    tables = {"pass1": {}, "target_error_rate": {}, "by_difficulty": {}}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    difficulties = None
    for i, (label, (protocol, base, rows)) in enumerate(runs.items()):
        style, marker = STYLES[i % len(STYLES)], MARKERS[i % len(MARKERS)]
        name = label if len(runs) > 1 else ""
        s_pass1 = summarize(base, rows, pass1)
        s_target = summarize(base, rows, target)
        tables["pass1"][label], tables["target_error_rate"][label] = s_pass1, s_target
        draw_arms(axes[0], s_pass1, style, marker, name)
        diff = sorted(s_pass1["s_minus_r"].items())
        axes[1].errorbar([t for t, _ in diff], [d["mean"] for _, d in diff], yerr=[d["half_width"] for _, d in diff],
                         color="k", linestyle=style, marker=marker, capsize=3, label=label)
        draw_arms(axes[2], s_target, style, marker, name)
        source = rows + ([base] if base else [])
        difficulties = difficulties or sorted({d for r in source for d in r["pass1_by_difficulty"]})
        tables["by_difficulty"][label] = {
            d: summarize(base, rows, lambda r, d=d: r["pass1_by_difficulty"][d]) for d in difficulties}
    task = next(iter(runs.values()))[0]
    targets = ", ".join(task["target_signatures"])
    axes[0].set(title="(a) Pass@1 on %s (greedy)" % split, xlabel="Iteration t", ylabel="Pass@1")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set(title="(b) S - R, paired by seed block", xlabel="Iteration t", ylabel="Pass@1 difference")
    axes[2].set(title="(c) Target error (%s) rate" % targets, xlabel="Iteration t", ylabel="Share of answers")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle("%s, %s: mean over seed blocks, 95%% t interval" % (task["base"]["model"], task["task"]))
    fig.tight_layout()
    fig.savefig(str(out) + "_pass1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(difficulties), figsize=(5.3 * len(difficulties), 4.4), squeeze=False)
    for i, label in enumerate(runs):
        for ax, d in zip(axes[0], difficulties):
            draw_arms(ax, tables["by_difficulty"][label][d], STYLES[i % len(STYLES)], MARKERS[i % len(MARKERS)],
                      label if len(runs) > 1 else "")
    for ax, d in zip(axes[0], difficulties):
        ax.set(title="Difficulty %s" % d, xlabel="Iteration t", ylabel="Pass@1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(str(out) + "_by_difficulty.png", dpi=150)
    plt.close(fig)
    return tables


def write_tables(tables, out):
    Path(str(out) + "_summary.json").write_text(json.dumps(tables, indent=2, sort_keys=True, default=str))
    with open(str(out) + "_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "metric", "difficulty", "series", "round", "mean", "sd", "n", "ci95_half_width", "paired_t_p"])
        for metric in ("pass1", "target_error_rate"):
            for label, s in tables[metric].items():
                rows_out(w, label, metric, "all", s)
        for label, per in tables["by_difficulty"].items():
            for d, s in per.items():
                rows_out(w, label, "pass1", d, s)


def rows_out(w, label, metric, difficulty, s):
    for arm in ARMS:
        for t, p in sorted(s["arms"][arm].items()):
            w.writerow([label, metric, difficulty, arm, t, p["mean"], p["sd"], p["n"], p["half_width"], ""])
    for t, p in sorted(s["s_minus_r"].items()):
        w.writerow([label, metric, difficulty, "S-R", t, p["mean"], p["sd"], p["n"], p["half_width"],
                    p.get("paired_t_p", "")])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="a run's --out directory; repeat to compare runs")
    p.add_argument("--label", action="append", help="one label per --run (default: the directory name)")
    p.add_argument("--splits", nargs="+",
                   help="default: eval_id and eval_ood, those the first run has evaluated")
    p.add_argument("--out", required=True, help="output path prefix, e.g. figures/graph_qwen3-1.7b")
    a = p.parse_args(argv)
    labels = a.label or [Path(r).name for r in a.run]
    if len(labels) != len(a.run):
        p.error("give one --label per --run")
    splits = a.splits or [s for s in ("eval_id", "eval_ood")
                          if (Path(a.run[0]) / ("evaluation_greedy_" + s) / "protocol.json").exists()]
    if not splits:
        print("refused: %s has no evaluation: run evaluate_multiround.py first" % a.run[0], file=sys.stderr)
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for split in splits:
        try:
            runs = {label: load(run, split) for label, run in zip(labels, a.run)}
        except ValueError as e:
            print("refused: %s" % e, file=sys.stderr)
            return 1
        if len({json.dumps({k: v[0][k] for k in ("split", "question_ids", "decoding", "base")}, sort_keys=True)
                for v in runs.values()}) > 1:
            print("refused: the runs were evaluated on other questions, decoding or base (%s)" % split,
                  file=sys.stderr)
            return 1
        prefix = "%s_%s" % (a.out, split)
        write_tables(plot(runs, prefix, split), prefix)
        print("wrote %s_pass1.png, %s_by_difficulty.png, %s_summary.json/.csv" % (prefix, prefix, prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
