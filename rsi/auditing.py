from __future__ import annotations

from collections import defaultdict

from .common import rng_for
from .tasks import judge, stratum


def allocation(groups, budget, policy, previous):
    """Allocate before looking at any current-round audit label."""
    from scipy.stats import beta
    counts = {g: 0 for g in groups}
    budget = min(budget, sum(len(v) for v in groups.values()))
    names = sorted(groups)
    # Every nonempty stratum receives one audit when budget permits.
    for g in names[:budget]:
        counts[g] += 1
    priorities = {}
    for g, rows in groups.items():
        old = previous.get(g, {"errors": 0, "correct": 0})
        priorities[g] = (len(rows) * float(beta.ppf(0.9, 1 + old["errors"], 1 + old["correct"]))
                         if policy == "adaptive" else 1.0)
    for _ in range(budget - sum(counts.values())):
        eligible = [g for g in names if counts[g] < len(groups[g])]
        # Weighted fair allocation, capped at the available stratum population.
        g = max(eligible, key=lambda x: (priorities[x] / (counts[x] + 1), x))
        counts[g] += 1
    return counts


def audit_pool(pool, tasks, config, seed, round_number, previous=None):
    policy, budget = config["policy"], min(config["budget"], len(pool))
    if policy == "none" or budget == 0:
        return [dict(c, weight=1.0) for c in pool], {"queries": 0, "groups": {}, "labels": [],
                                                  "estimated_error": None}
    previous = previous or {}
    groups = defaultdict(list)
    for c in pool:
        groups[stratum(tasks[c["task_id"]])].append(c)
    rng = rng_for(seed, round_number, "audits")
    if policy == "uniform":
        sampled = rng.sample(pool, budget)
        inclusion = {c["id"]: budget / len(pool) for c in sampled}
    else:
        counts = allocation(groups, budget, policy, previous)
        sampled, inclusion = [], {}
        for g in sorted(groups):
            selected = rng.sample(groups[g], counts[g])
            sampled.extend(selected)
            inclusion.update({c["id"]: counts[g] / len(groups[g]) for c in selected})
    stats = {g: {"errors": 0, "correct": 0, "population": len(rows)} for g, rows in groups.items()}
    labels, audited = [], {}
    for c in sampled:
        # This is the sole online trusted query. No reference solution is revealed.
        correct = judge(tasks[c["task_id"]], c["response"])["correct"]
        audited[c["id"]] = correct
        g = stratum(tasks[c["task_id"]])
        stats[g]["correct" if correct else "errors"] += 1
        labels.append({"candidate_id": c["id"], "correct": correct, "stratum": g,
                       "inclusion_probability": inclusion[c["id"]]})
    retained = []
    for c in pool:
        if c["id"] in audited:
            if audited[c["id"]]:
                retained.append(dict(c, weight=1.0))
            continue
        s = stats[stratum(tasks[c["task_id"]])]
        n = s["errors"] + s["correct"]
        weight = max(0.0, 1 - 2 * (1 + s["errors"]) / (2 + n)) if n and config["weighting"] else 1.0
        retained.append(dict(c, weight=weight))
    if policy == "uniform":
        estimate = sum(not x["correct"] for x in labels) / budget
    elif all(s["errors"] + s["correct"] for s in stats.values()):
        estimate = sum(s["population"] * s["errors"] / (s["errors"] + s["correct"])
                       for s in stats.values()) / len(pool)
    else:
        estimate = None  # Never claim unbiased coverage of unaudited strata.
    return retained, {"queries": len(labels), "groups": stats, "labels": labels,
                      "estimated_error": estimate, "policy": policy}
