"""Opt-in trusted auditing after R/S selection, with explicit pre/post statistics.

Selection labels are measurement-only. The allocator receives no hidden error
taxonomy or correctness labels; audit_pool obtains only its budgeted labels.
An iterative audit ledger survives interrupted training and reuses completed
audit results rather than charging those queries again on resume.
"""
import math
from pathlib import Path

from .auditing import audit_pool
from .common import digest, read_json, write_json


def enabled(config):
    """Validate the audit section, including integer query budgets."""
    if config["policy"] not in ("none", "uniform", "balanced", "adaptive"):
        raise ValueError("Unknown audit policy")
    if type(config["budget"]) is not int or config["budget"] < 0:
        raise ValueError("Audit budget must be a nonnegative integer")
    if config["policy"] == "none" and config["budget"]:
        raise ValueError("Audit policy none requires budget 0")
    if type(config["weighting"]) is not bool:
        raise ValueError("Audit weighting must be a boolean")
    return config["policy"] != "none" and config["budget"] > 0


def positive_rows(rows):
    """Validate weights and identify the examples the optimizer actually uses."""
    if any(not math.isfinite(float(r["weight"])) or float(r["weight"]) < 0 for r in rows):
        raise ValueError("Training weights must be finite and nonnegative")
    if not math.isfinite(sum(float(r["weight"]) for r in rows)):
        raise ValueError("Total training weight must be finite")
    return [r for r in rows if float(r["weight"]) > 0]


def statistics(rows, pool_counts):
    """Rates refer to the original pool, not the retained subset itself."""
    active = positive_rows(rows)
    k, c = len(rows), sum(bool(r["correct"]) for r in rows)
    n_plus, n_minus = pool_counts["N_plus"], pool_counts["N_minus"]
    weight_sum = sum(float(r["weight"]) for r in rows)
    weight_sq = sum(float(r["weight"]) ** 2 for r in rows)
    active_c = sum(bool(r["correct"]) for r in active)
    return {"K": k, "C": c, "E": k - c, "N_plus": n_plus, "N_minus": n_minus,
            "final_TPR": c / n_plus if n_plus else None,
            "final_FPR": (k - c) / n_minus if n_minus else None,
            "precision": c / k if k else None,
            "yield": k / (n_plus + n_minus) if n_plus + n_minus else None,
            "weight_sum": weight_sum, "weights_all_one": all(r["weight"] == 1 for r in rows),
            "effective_sample_size": weight_sum ** 2 / weight_sq if weight_sq else 0,
            "positive_weight_examples": len(active), "positive_weight_correct": active_c,
            "positive_weight_errors": len(active) - active_c,
            "positive_weight_TPR": active_c / n_plus if n_plus else None,
            "positive_weight_FPR": (len(active) - active_c) / n_minus if n_minus else None}


def apply_audit(rows, tasks, config, seed, round_number, pool_counts, previous=None, cache_path=None):
    """Return post-audit rows and a cost/label/statistics report.

    cache_path, when supplied, is outside the retryable round directory. A
    changed selection, dataset, config or history is refused, not re-audited.
    The budget is per arm per round, capped at the pre-audit subset size.
    """
    active = enabled(config)
    positive_rows(rows)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate selected candidate IDs before auditing")
    previous = previous or {}
    counts = {key: pool_counts[key] for key in ("N_plus", "N_minus")}
    binding = {"selection_hash": digest(rows),
               "tasks_hash": digest({r["task_id"]: tasks[r["task_id"]] for r in rows}),
               "config": config, "seed": seed, "round": round_number,
               "previous_hash": digest(previous), "pool_counts": counts}
    cache = Path(cache_path) if cache_path is not None and active else None
    if cache is not None and cache.exists():
        saved = read_json(cache)
        if saved["binding"] != binding:
            raise ValueError("Audit ledger inputs changed; refusing extra queries: %s" % cache)
        payload = {"rows": saved["rows"], "report": saved["report"]}
        if saved["payload_hash"] != digest(payload):
            raise ValueError("Audit ledger payload changed: %s" % cache)
        positive_rows(saved["rows"])
        return saved["rows"], dict(saved["report"], reused=True)

    # No correctness/error fields are exposed to the audit allocation policy.
    visible = [{key: r[key] for key in ("id", "task_id", "response", "weight")} for r in rows]
    if active:
        retained, report = audit_pool(visible, tasks, config, seed, round_number, previous)
    else:
        retained, report = visible, {"queries": 0, "groups": {}, "labels": [], "estimated_error": None}
    by_id = {r["id"]: r for r in rows}
    after = [dict(by_id[r["id"]], weight=r["weight"]) for r in retained]
    report = dict(report, enabled=active, policy=config["policy"], requested_budget=config["budget"],
                  available_examples=len(rows), seed=seed, round=round_number, reused=False,
                  before=statistics(rows, counts), after=statistics(after, counts),
                  matching_constraints_stage="pre_audit",
                  post_audit_matching_guaranteed=not active,
                  oracle_access="correctness-only trusted audit queries; no reference solutions")
    if report["queries"] > min(config["budget"], len(rows)):
        raise RuntimeError("Auditor exceeded the per-arm per-round query budget")
    if cache is not None:
        payload = {"rows": after, "report": report}
        write_json(cache, dict(schema=1, binding=binding, payload_hash=digest(payload), **payload))
    return after, report
