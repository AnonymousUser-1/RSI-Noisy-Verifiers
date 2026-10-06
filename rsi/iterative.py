"""Per-arm selection for rounds 2+ of the multi-round (iterative) extension.

Round 1 is the joint matched construction (rsi.matching): one shared pool, the
same prompts, the same correct answers and per-task token counts within
matching.token_tolerance in R and S.  From round 2 each arm samples its own pool from its own checkpoint, so
the two pools hold different answers and that construction no longer applies.
Team decision 2026-10-01: from round 2, each arm selects its own K from its own
pool, and only the counts and the rates are matched -- K examples, C:E as matching.error_fraction sets (3:1 by default),
one example per prompt, every prompt from the round's common prompt set.

The rule is the round-1 rule without the shared pool:
  * R takes its E errors from prompts drawn uniformly among the prompts that
    have any error, then one error uniformly inside each prompt;
  * S takes them from prompts drawn uniformly among the prompts that have a
    target error (rsi.matching.targets_for: `nonshortest` for graph,
    `ignore_parentheses` for arithmetic, `intermediate` for gsm8k, `sign` for
    dmmath), then one target error uniformly inside each;
  * both then take C correct answers from prompts drawn uniformly among the
    remaining prompts that have one, one correct answer uniformly inside each.
Nothing is relaxed: an arm that cannot fill a quota is reported infeasible with
its certificate, and the caller stops that arm.
"""
from __future__ import annotations

from collections import defaultdict

from .common import rng_for
from .matching import DEFAULT_ERROR_FRACTION, TARGET_SIGNATURES, _validated_targets, quotas

ARMS = ("R", "S")


def select_own_subset(candidates, arm, k, seed, label, target_signatures=TARGET_SIGNATURES,
                      error_fraction=DEFAULT_ERROR_FRACTION):
    """Select one arm's K examples from its own pool.

    `candidates` are rsi.matching.Candidate rows of this arm's pool for this
    round; `label` names the draw (e.g. (block, round, arm)) so that every
    arm-round draws from its own stream; `target_signatures` are S's target
    errors (rsi.matching.targets_for of the dataset's task).  Returns
    {"feasible", "rows", "ids", "certificate", "audit"}.
    """
    target_signatures = _validated_targets(target_signatures)
    if arm not in ARMS:
        raise ValueError("arm must be R or S, not %r" % (arm,))
    c_quota, e_quota = quotas(k, error_fraction)   # ValueError unless E = K * error_fraction is whole

    # An answer cut at max_new_tokens is never selected (rsi/experiment.py ROW_FIELDS).
    truncated = sum(1 for c in candidates if c.truncated)
    candidates = [c for c in candidates if not c.truncated]
    correct, errors, targets = defaultdict(list), defaultdict(list), defaultdict(list)
    for c in sorted(candidates, key=lambda c: c.candidate_id):
        if c.correct:
            correct[c.task_id].append(c)
        else:
            errors[c.task_id].append(c)
            if c.error in target_signatures:
                targets[c.task_id].append(c)
    eligible = targets if arm == "S" else errors
    n_plus = sum(len(v) for v in correct.values())
    n_minus = sum(len(v) for v in errors.values())
    certificate = {"K": k, "C": c_quota, "E": e_quota, "arm": arm,
                   "prompts_in_pool": len(set(correct) | set(errors)),
                   "error_prompts_available": len(eligible), "correct_prompts_available": len(correct),
                   "N_plus": n_plus, "N_minus": n_minus, "truncated_excluded": truncated}

    def infeasible(reason):
        certificate.update(feasible=False, reason=reason)
        return {"feasible": False, "rows": [], "ids": [], "certificate": certificate, "audit": None}

    if len(eligible) < e_quota:
        return infeasible("%d prompts have an eligible error for %s; E = %d" % (len(eligible), arm, e_quota))
    error_tasks = rng_for(seed, label, "error_prompts").sample(sorted(eligible), e_quota)
    open_correct = sorted(set(correct) - set(error_tasks))
    if len(open_correct) < c_quota:
        return infeasible("%d prompts outside the error prompts have a correct answer; C = %d"
                          % (len(open_correct), c_quota))
    correct_tasks = rng_for(seed, label, "correct_prompts").sample(open_correct, c_quota)

    pick = rng_for(seed, label, "candidates")
    rows = [pick.choice(eligible[t]) for t in sorted(error_tasks)]
    rows += [pick.choice(correct[t]) for t in sorted(correct_tasks)]
    rows.sort(key=lambda c: c.task_id)
    certificate.update(feasible=True, reason=None)
    audit = {"K": len(rows), "C": c_quota, "E": e_quota,
             "target_hits": sum(1 for c in rows if not c.correct and c.error in target_signatures),
             "target_signatures": list(target_signatures),
             "error_signatures": dict(sorted(_count(c.error for c in rows if not c.correct).items())),
             "N_plus": n_plus, "N_minus": n_minus,
             "final_TPR": c_quota / n_plus, "final_FPR": e_quota / n_minus if n_minus else None,
             "precision": c_quota / k, "yield": k / (n_plus + n_minus),
             "supervised_tokens": sum(c.tokens for c in rows)}
    return {"feasible": True, "rows": rows, "ids": [c.candidate_id for c in rows],
            "certificate": certificate, "audit": audit}


def _count(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts
