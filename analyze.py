#!/usr/bin/env python3
"""Aggregate trajectories, verification diagnostics, paired contrasts, and plots."""
import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from rsi.common import digest, read_json, read_jsonl, write_json
from rsi.tasks import judge


def save_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def confidence(values):
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    half = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1)/np.sqrt(len(values)))
    return mean, mean-half, mean+half


def analyze(root, out, include_demo=False, plots=False, compare=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    measurements, diagnostics, groups, finals = [], [], {}, []
    manifests = sorted(Path(root).rglob("run.json"))
    for path in manifests:
        manifest, run = read_json(path), path.parent
        if manifest["DEMO_ONLY"] and not include_demo:
            continue
        cfg = copy.deepcopy(manifest["config"])
        seed = cfg.pop("seed")
        # Calibration contents, rather than an arbitrary filename, define the method.
        cfg["verifier"]["calibration"] = manifest["calibration"]
        group = digest({"config": cfg, "dataset": manifest["dataset_hash"]})[:12]
        label = "%s / %s / %s B=%d%s" % (manifest["task"], cfg["verifier"]["kind"], cfg["audit"]["policy"],
                                           cfg["audit"]["budget"], " (frozen)" if cfg["frozen"] else "")
        groups[group] = {"label": label, "config": cfg, "dataset_hash": manifest["dataset_hash"]}
        per_split = defaultdict(list)
        for eval_path in sorted((run / "evaluation").glob("round_*.json")):
            row = read_json(eval_path)
            measurements.append(dict(row, group=group, label=label, seed=seed, run=str(run)))
            per_split[row["split"]].append(row)
        for split, rows in per_split.items():
            rows.sort(key=lambda r: r["round"])
            if rows[-1]["round"] != cfg["rounds"] or rows[0]["round"] != 0:
                continue
            final = rows[-1]
            finals.append({"group": group, "label": label, "seed": seed, "split": split,
                           "accuracy": final["accuracy"], "gain": final["accuracy"]-rows[0]["accuracy"],
                           "worst_regression": max(0, rows[0]["accuracy"]-min(r["accuracy"] for r in rows)),
                           "online_labels": final["cumulative_audit_queries"],
                           "total_exposed_labels": final["cumulative_audit_queries"]+final["calibration_queries"],
                           "round": final["round"], "draws": final["draws"], "run": str(run)})
        data = Path(manifest["data_path"])
        for done_path in sorted(run.glob("round_*/complete.json")):
            done = read_json(done_path)
            tasks = {x["id"]: x for x in read_jsonl(data / ("train_%03d.jsonl" % done["round"]))}
            attempt = run / done["attempt"]
            # Denominators are counted once, from the common original pool that every
            # stage was drawn from. Stage TPR/FPR taken over the stage's own rows is the
            # statistic the design explicitly rejects,
            # because selection, provisional reduction and the training cap each shrink
            # and reweight the pool, so a rate computed inside a stage is not comparable
            # across stages or branches. The within-stage rates are still emitted, under
            # `stage_tpr` / `stage_fpr`, so the two readings stay visible side by side.
            pool = read_jsonl(attempt / "candidates.jsonl")
            pool_truths = [judge(tasks[c["task_id"]], c["response"]) for c in pool]
            n_pos = sum(t["correct"] for t in pool_truths)
            n_neg = len(pool) - n_pos
            for stage in ("selection", "provisional", "retained"):
                rows = read_jsonl(attempt / (stage + ".jsonl"))
                truths = [judge(tasks[r["task_id"]], r["response"]) for r in rows]
                accepted = [r["accepted"] if stage == "selection" else True for r in rows]
                good = sum(t["correct"] for t in truths)
                bad = len(rows)-good
                tp = sum(t["correct"] and a for t, a in zip(truths, accepted))
                fp = sum(not t["correct"] and a for t, a in zip(truths, accepted))
                errors = defaultdict(int)
                for truth, accept in zip(truths, accepted):
                    if accept and not truth["correct"]:
                        errors[truth["error"]] += 1
                probabilities = [r.get("verifier_score") for r in rows]
                brier = (sum((s-float(t["correct"]))**2 for s, t in zip(probabilities, truths))/len(rows)
                         if rows and all(s is not None for s in probabilities) else None)
                kept = tp+fp
                diagnostics.append({"group": group, "seed": seed, "round": done["round"], "stage": stage,
                                    "n": len(rows), "accepted": kept,
                                    # ---------------- common-original-pool denominators
                                    "pool_n": len(pool), "n_positive": n_pos, "n_negative": n_neg,
                                    "tpr": tp/n_pos if n_pos else None,
                                    "fpr": fp/n_neg if n_neg else None,
                                    "yield": kept/(n_pos+n_neg) if (n_pos+n_neg) else None,
                                    # `precision = C / K` is this same ratio; the
                                    # existing column name is kept so old CSVs still line up.
                                    "purity": tp/kept if kept else None,
                                    "quota_target_tpr": cfg["verifier"]["tpr"],
                                    "quota_target_fpr": cfg["verifier"]["fpr"],
                                    "denominator": "common_original_pool",
                                    # ---------------- kept visible: within-stage rates
                                    "stage_tpr": tp/good if good else None,
                                    "stage_fpr": fp/bad if bad else None,
                                    "brier": brier, "accepted_errors": dict(errors),
                                    "tpr_fpr_are_selection_rates": stage == "selection"})
    save_csv(out / "learning_curves.csv", measurements)
    save_csv(out / "final_per_seed.csv", finals)
    save_csv(out / "verification_diagnostics.csv", diagnostics)
    write_json(out / "groups.json", groups)
    grouped = defaultdict(list)
    for row in finals:
        grouped[(row["group"], row["split"])].append(row)
    summary = []
    for (group, split), rows in grouped.items():
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("Duplicate seed/run for group " + group + "; narrow --root")
        mean, low, high = confidence([r["accuracy"] for r in rows])
        summary.append({"group": group, "label": groups[group]["label"], "split": split,
                        "seeds": len(rows), "accuracy_mean": mean, "ci_low": low, "ci_high": high,
                        "online_labels": rows[0]["online_labels"]})
    save_csv(out / "summary.csv", summary)
    if compare:
        left, right = compare
        if left not in groups or right not in groups:
            raise ValueError("Use group IDs from groups.json")
        if groups[left]["dataset_hash"] != groups[right]["dataset_hash"]:
            raise ValueError("Paired comparison requires the same task dataset")
        contrasts = []
        for split in ("eval_id", "eval_ood", "dev"):
            l = {r["seed"]: r for r in finals if r["group"] == left and r["split"] == split}
            r = {r["seed"]: r for r in finals if r["group"] == right and r["split"] == split}
            common = sorted(set(l) & set(r))
            if not common:
                continue
            if any((l[s]["round"], l[s]["draws"]) != (r[s]["round"], r[s]["draws"]) for s in common):
                raise ValueError("Comparison endpoints/draw counts differ")
            mean, low, high = confidence([l[s]["accuracy"]-r[s]["accuracy"] for s in common])
            contrasts.append({"left": left, "right": right, "split": split, "paired_seeds": common,
                              "difference": mean, "ci_low": low, "ci_high": high})
        write_json(out / "paired_contrast.json", contrasts)
    if plots and measurements:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for split in sorted({r["split"] for r in measurements}):
            fig, ax = plt.subplots(figsize=(9, 5))
            for group in groups:
                by_round = defaultdict(list)
                for r in measurements:
                    if r["group"] == group and r["split"] == split:
                        by_round[r["round"]].append(r["accuracy"])
                if not by_round:
                    continue
                times = sorted(by_round)
                means = [np.mean(by_round[t]) for t in times]
                ax.plot(times, means, marker="o", label=groups[group]["label"])
            ax.set(xlabel="Self-training round", ylabel="Mean single-answer correctness", title=split)
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(out / ("learning_%s.png" % split), dpi=180)
            fig.savefig(out / ("learning_%s.pdf" % split))
            plt.close(fig)
        # One cost curve per task/model/verifier to avoid conflating experiments.
        cost_groups = defaultdict(list)
        for row in finals:
            cfg = groups[row["group"]]["config"]
            if row["split"] == "eval_id" and cfg["verifier"]["kind"] == "persistent" and cfg["audit"]["weighting"]:
                identity = (groups[row["group"]]["dataset_hash"], cfg["model"], cfg["revision"])
                cost_groups[identity].append(row)
        for identity, rows in cost_groups.items():
            fig, ax = plt.subplots(figsize=(7, 4))
            policy_points = defaultdict(lambda: defaultdict(list))
            for row in rows:
                cfg = groups[row["group"]]["config"]
                policy_points[cfg["audit"]["policy"]][row["total_exposed_labels"]].append(row["accuracy"])
            for policy, points in sorted(policy_points.items()):
                xs = sorted(points)
                means = [float(np.mean(points[x])) for x in xs]
                ax.plot(xs, means, marker="o", label=policy)
            ax.set(xlabel="Trusted correctness labels exposed to learning", ylabel="Final single-answer correctness",
                   title=identity[1])
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / ("audit_cost_%s.png" % digest(identity)[:10]), dpi=180)
            fig.savefig(out / ("audit_cost_%s.pdf" % digest(identity)[:10]))
            plt.close(fig)
    print("Analyzed %d final seed/split records. Groups: %s" % (len(finals), ", ".join(groups)))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-demo", action="store_true", help="Allow explicitly marked fake-model smoke outputs")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("LEFT_GROUP", "RIGHT_GROUP"))
    a = p.parse_args()
    analyze(a.root, a.out, a.include_demo, a.plots, a.compare)
