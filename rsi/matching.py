from __future__ import annotations

"""Joint construction of the R and S final training subsets.

This module replaces the old "filter the pool with quotas, then reduce to one
answer per prompt, then audit, then cap" chain.  That chain ran two independent
draws over the same candidate pool, so two branches could agree on every
*candidate-level* acceptance statistic and still hand the trainer different
numbers of examples, different problems, different error counts, and different
supervised token counts.

Here both branches are built from the **common original pool** at once, and the
construction is handed to a constraint solver whose hard constraints are the
matched quantities themselves:

  * the same set of task ids appears in both branches;
  * the same set of correct-response ids appears in both branches;
  * per task, the numbers of supervised response tokens (including the real EOS,
    excluding prompt and padding) of R's and S's answers differ by at most
    `token_tolerance` (config matching.token_tolerance; 0 by default, i.e.
    identical; None: no limit);
  * C:E is exactly as `error_fraction` sets (config matching.error_fraction:
    E = K * error_fraction; 0.25, i.e. 3:1, by default) with common K = C + E;
  * every weight is 1.0 and every branch performs exactly one update.

What is left free is drawn at random from the seed and the block's own pool
(`pool_key`), never from the block's name, so the seed blocks draw
independently and overlap only by chance:

  * *which* tasks take the C correct and the E error roles.  Among equally
    constrained tasks the solver follows a random order.  (In task-id order
    every block took nearly the same lowest-id correct prompts.)
  * *which* responses fill those tasks.  A correct task's answer, shared by R
    and S, is drawn uniformly inside its length bucket.  S takes a target error
    (`nonshortest` by default) that has a non-target error within
    `token_tolerance` tokens: at tolerance 0 at the task's first such length
    (the frozen rule), otherwise uniformly among them.  R draws its error
    uniformly among the task's errors within `token_tolerance` tokens of S's and
    may land on the same candidate as S; that coincidence is kept, never redrawn.

A separate arithmetic intervention explicitly passes `ignore_parentheses`; it
does not change the frozen graph default.  The entries pass targets_for(the
dataset's task), which keeps those two and gives the imported tasks theirs
(gsm8k 'intermediate', dmmath 'sign').

Nothing here trains, and nothing here looks at gradients or test results.  The
caller is expected to run this only after every block's pool exists and is hash
bound.
"""

import math
from collections import defaultdict
import operator
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .common import digest, rng_for

# The graph main line keeps its original default.  Arithmetic callers must opt
# into their separate, recorded target rather than replacing this constant.
TARGET_SIGNATURES = ("nonshortest",)
ARITHMETIC_TARGET_SIGNATURES = ("ignore_parentheses",)
# Each task's target is its persistent controlled verifier's, the first label of
# rsi.tasks.ERROR_SIGNATURES (tests/test_external_tasks.py holds the two together).
TARGETS_BY_TASK = {"graph": TARGET_SIGNATURES, "arithmetic": ARITHMETIC_TARGET_SIGNATURES,
                   "gsm8k": ("intermediate",), "dmmath": ("sign",)}


def targets_for(task):
    """S's target error signatures for a task, as the dataset manifest names it."""
    if task not in TARGETS_BY_TASK:
        raise ValueError("No S target is declared for task %r; known: %s" % (task, ", ".join(TARGETS_BY_TASK)))
    return TARGETS_BY_TASK[task]

# Declared, uniform downsizing ladder.  The C:E ratio and the length tolerance are
# config fields (matching.error_fraction, matching.token_tolerance), 3:1 and 0 by default.  Within a
# run nothing is relaxed: no seed change, no size outside the ladder, no tolerance beyond
# the configured one.
K_LADDER = (64, 32, 16)
DEFAULT_ERROR_FRACTION = 0.25   # C:E = 3:1
DEFAULT_TOKEN_TOLERANCE = 0     # identical per-task supervised token counts


def quotas(k, error_fraction=DEFAULT_ERROR_FRACTION):
    """(C, E) for one arm's K, as Python ints: E = K * error_fraction wrong answers, C = K - E correct.
    The product is exact (error_fraction as written in decimal, so 0.07 x 100 is 7).  Raises
    ValueError unless K is an integer and E a whole number with 1 <= E <= K - 1."""
    try:
        k = operator.index(k)
        if isinstance(error_fraction, bool):
            raise TypeError
        errors = Fraction(str(error_fraction)) * k
    except (TypeError, ValueError):
        raise ValueError("K must be an integer and error_fraction a number, not %r and %r" % (k, error_fraction))
    if errors.denominator != 1 or not 1 <= errors <= k - 1:
        raise ValueError("error_fraction %r gives K=%d a quota of %r wrong answers; it must give a whole "
                         "number from 1 to K-1 at every K of the ladder %s (e.g. 0.25 = 3:1, 0.125 = 7:1, "
                         "0.5 = 1:1)" % (error_fraction, k, errors, K_LADDER))
    return k - int(errors), int(errors)


def _within(a, b, tolerance):
    return tolerance is None or abs(a - b) <= tolerance

# Bounded search.  A real pool is intended to be solvable greedily; the cap
# exists so an infeasible pool reports infeasibility instead of hanging.
MAX_SEARCH_NODES = 200000


class MatchingFailure(RuntimeError):
    """Raised when the hard constraints cannot be met.  Never caught to relax."""


@dataclass(frozen=True)
class Candidate:
    """One generated response with the facts the construction is allowed to use."""

    candidate_id: str
    task_id: str
    response: str
    correct: bool
    error: str
    tokens: int  # supervised response tokens, including EOS
    truncated: bool = False  # cut at max_new_tokens: never matched, selected or trained on


def candidate_from_row(row: dict, task: dict, judgement: dict, tokens: int) -> Candidate:
    return Candidate(candidate_id=row["id"], task_id=row["task_id"], response=row["response"],
                     correct=bool(judgement["correct"]), error=judgement["error"], tokens=int(tokens),
                     truncated=bool(row.get("truncated", False)))


def pool_key(candidates: Sequence[Candidate]) -> str:
    """What keys a pool's random draws: its candidate ids and their sampled responses.

    Candidate ids are digest([task_id, sample]) and repeat across blocks; the
    responses do not, so every seed block draws its own.  Neither the block's
    name nor the order of its rows enters, so one pool and one seed always give
    one matching.
    """
    return digest(sorted([c.candidate_id, c.response] for c in candidates))


def _buckets(candidates: Sequence[Candidate]) -> Dict[str, Dict[int, List[str]]]:
    """task -> token length -> candidate ids present at that length (both labels)."""
    out: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        out[c.task_id][c.tokens].append(c.candidate_id)
    return {t: dict(v) for t, v in out.items()}


@dataclass
class _TaskView:
    task_id: str
    correct_ids: Dict[int, List[str]]      # length -> correct candidate ids
    error_ids: Dict[int, List[str]]        # length -> error candidate ids
    target_ids: Dict[int, List[str]]       # length -> error ids in the caller's target signatures
    signal_lengths: List[int]              # target lengths with a non-target error within the tolerance
    token_tolerance: Optional[int] = 0     # how far R's error may be from S's in supervised tokens

    def window(self, length: int) -> List[str]:
        """The error ids R draws from when S's error has `length` tokens."""
        return [cid for ln in sorted(self.error_ids) if _within(ln, length, self.token_tolerance)
                for cid in self.error_ids[ln]]

    @property
    def has_correct(self) -> bool:
        return any(self.correct_ids.values())

    @property
    def bucket(self) -> int:
        """Length of the correct response.  A single length only: the contract
        asserts the same per-task supervised token count in both branches, so
        the bucket is not a free choice."""
        return next(iter(self.correct_ids)) if self.correct_ids else -1

    @property
    def eligible(self) -> bool:
        """J_E membership: a target error with a non-target error within the token tolerance."""
        return bool(self.signal_lengths)


def _views(candidates: Sequence[Candidate],
           target_signatures: Iterable[str] = TARGET_SIGNATURES,
           token_tolerance: Optional[int] = DEFAULT_TOKEN_TOLERANCE) -> Dict[str, _TaskView]:
    by_task: Dict[str, List[Candidate]] = defaultdict(list)
    for c in candidates:
        if not c.truncated:  # an answer cut at max_new_tokens takes no role
            by_task[c.task_id].append(c)
    views: Dict[str, _TaskView] = {}
    for task_id, rows in by_task.items():
        correct: Dict[int, List[str]] = defaultdict(list)
        errors: Dict[int, List[str]] = defaultdict(list)
        target: Dict[int, List[str]] = defaultdict(list)
        for c in sorted(rows, key=lambda x: x.candidate_id):
            if c.correct:
                correct[c.tokens].append(c.candidate_id)
            else:
                errors[c.tokens].append(c.candidate_id)
                if c.error in target_signatures:
                    target[c.tokens].append(c.candidate_id)
        # Signal lengths: a target error's length with a non-target error within the tolerance
        # (with tolerance 0: a length holding both a target and a non-target error); no
        # correct candidate is needed at that length.
        other = [ln for ln in errors for _ in range(len(errors[ln]) - len(target.get(ln, [])))]
        signal = [ln for ln in sorted(target) if any(_within(o, ln, token_tolerance) for o in other)]
        views[task_id] = _TaskView(task_id=task_id, correct_ids=dict(correct), error_ids=dict(errors),
                                   target_ids=dict(target), signal_lengths=signal, token_tolerance=token_tolerance)
    return views


def _pick(rng, ids: Sequence[str], preferred: FrozenSet[str] = frozenset()) -> str:
    """Uniform inside the bucket, optionally restricted to preferred ids."""
    pool = sorted(ids)
    if preferred:
        narrowed = [i for i in pool if i in preferred]
        if narrowed:
            pool = narrowed
    return pool[0] if len(pool) == 1 else rng.choice(pool)


def _solve(views: Dict[str, _TaskView], c_quota: int, e_quota: int, seed, *, draw_key: str) -> Optional[dict]:
    """Assign each task at most one role and fill both quotas exactly.

    A task can serve as a correct task (contributes 1 to C) or as an error task
    (contributes 1 to E), because both branches must agree on the task set and
    on the per-task token count; serving both would need two lengths.

    Inside a node the most constrained task is tried first: tasks with one legal
    role (correct-only or error-only) before the flexible ones, so a flexible
    task only fills what the one-role tasks cannot.  (SCIENCE_ROUND2 3 lists the
    classes as correct-only, flexible, error-only; `pressure` re-sorts every
    node, so both one-role classes come first together and the initial
    `ordered` list does not set the order.)  The search is complete: the
    documented counterexample (C=E=1, task A correct+same-length target/other
    errors, task B correct only) has the single valid assignment B->correct,
    A->error, and any order finds it.  Where several assignments are valid, the
    order decides which one is returned.

    Equally constrained tasks come in a random order drawn from `seed` and
    `draw_key` (pool_key), so the tasks that fill each quota are a uniform draw
    rather than the lowest task ids.  Every task of an 8-answer pool has 8
    candidates, so this tie-break decides which prompts are used; in task-id
    order every seed block filled its correct quota with nearly the same
    prompts.
    """
    target_tasks = sorted(views)
    correct_only = [t for t in target_tasks if views[t].has_correct and not views[t].eligible]
    error_only = [t for t in target_tasks if views[t].eligible and not views[t].has_correct]
    flexible = [t for t in target_tasks if views[t].has_correct and views[t].eligible]

    shuffled = list(target_tasks)
    rng_for(seed, draw_key, "matched_subset_task_order").shuffle(shuffled)
    rank = {t: i for i, t in enumerate(shuffled)}
    rng = rng_for(seed, draw_key, "matched_subset_assignment")
    chosen: Dict[str, str] = {}
    nodes = [0]

    def recurse(pool: List[str], c_left: int, e_left: int) -> bool:
        nodes[0] += 1
        if nodes[0] > MAX_SEARCH_NODES:
            raise MatchingFailure("Matching search exceeded %d nodes; pool structure is pathological"
                                  % MAX_SEARCH_NODES)
        if c_left == 0 and e_left == 0:
            return True
        if c_left + e_left > len(pool):
            return False
        if c_left > sum(1 for t in pool if views[t].has_correct):
            return False
        if e_left > sum(1 for t in pool if views[t].eligible):
            return False
        # Most constrained first: fewest legal roles, then fewest candidate ids, then this pool's
        # random order.
        def pressure(t: str) -> Tuple[int, int, int]:
            roles = (1 if views[t].has_correct else 0) + (1 if views[t].eligible else 0)
            total = sum(len(v) for v in views[t].correct_ids.values()) + \
                    sum(len(v) for v in views[t].error_ids.values())
            return (roles, total, rank[t])

        for t in sorted(pool, key=pressure):
            order: List[str] = []
            if e_left and views[t].eligible:
                order.append("error")
            if c_left and views[t].has_correct:
                order.append("correct")
            # When both roles are legal the random draw decides which is tried
            # first; the full search still runs, so a valid assignment is never
            # missed because of the coin flip.
            if len(order) == 2 and rng.random() < 0.5:
                order.reverse()
            rest = [x for x in pool if x != t]
            for role in order:
                chosen[t] = role
                ok = recurse(rest, c_left - (role == "correct"), e_left - (role == "error"))
                if ok:
                    return True
                del chosen[t]
        return False

    ordered = correct_only + flexible + error_only  # recurse re-sorts by pressure at every node
    if not recurse(ordered, c_quota, e_quota):
        return None
    return dict(chosen)


def _materialise(views, assignment, candidates_by_id, seed, c_quota=None, e_quota=None, *, draw_key):
    """Turn a role assignment into the two candidate sets, with the same
    correct id in both branches.

    Each assigned task contributes exactly one candidate per branch:
    - correct role: one shared correct candidate in both R and S
    - error role: one error candidate in R, one (possibly different) error in S

    Error candidates: S takes a target error that has a non-target error within
    the token tolerance (at tolerance 0 at the task's first signal length, the
    frozen rule; otherwise uniformly among them), R draws uniformly among the
    errors within the tolerance of S's.  The correct role does not add any error
    candidates.

    ``c_quota``/``e_quota`` are the caller's request.  When given, the produced
    sets must match them exactly, and a mismatch raises: a subset whose size the
    certificate does not describe is a bug in this function, not an infeasible
    pool, so it must not come back as ``feasible=False`` and walk the K ladder
    down.

    ``draw_key`` (pool_key) keys the draws, as in _solve, so no two blocks share a
    random stream.
    """
    rng = rng_for(seed, draw_key, "matched_subset_selection")
    r_ids, s_ids = set(), set()
    correct_common: List[str] = []
    error_pairs: List[Tuple[str, str]] = []

    for task_id in sorted(assignment):
        view = views[task_id]
        role = assignment[task_id]

        if role == "correct":
            # Pick one correct candidate from this task's signal bucket
            # (the bucket is the correct candidate's length)
            length = view.bucket
            if length not in view.correct_ids:
                raise MatchingFailure(f"Task {task_id} assigned 'correct' role but has no correct candidates")
            good = _pick(rng, view.correct_ids[length])
            correct_common.append(good)
            r_ids.add(good)
            s_ids.add(good)

        elif role == "error":
            # One error candidate per branch: S takes a target error that has a non-target error
            # within the token tolerance; R draws uniformly among the errors within the tolerance
            # of S's (S's own included).
            if not view.signal_lengths:
                raise MatchingFailure(f"Task {task_id} assigned 'error' role but has no signal lengths")
            if view.token_tolerance == 0:
                # The frozen rule: S at the first signal length (a length holding a target and a
                # non-target error), R uniform inside that length bucket.
                error_length = view.signal_lengths[0]
                s_choice = _pick(rng, view.error_ids[error_length],
                               preferred=frozenset(view.target_ids.get(error_length, [])))
            else:
                # S uniform among all qualifying target errors: always taking the shortest (the first
                # signal length) would make S's errors systematically shorter than R's.
                qualifying = {cid: ln for ln in view.signal_lengths for cid in view.target_ids[ln]}
                s_choice = _pick(rng, list(qualifying))
                error_length = qualifying[s_choice]
            r_choice = _pick(rng, view.window(error_length))

            r_ids.add(r_choice)
            s_ids.add(s_choice)
            error_pairs.append((r_choice, s_choice))
        else:
            raise MatchingFailure(f"Unknown role {role!r} for task {task_id}")

    if c_quota is not None and len(correct_common) != c_quota:
        raise MatchingFailure("_materialise produced %d correct candidates but the request was %d"
                              % (len(correct_common), c_quota))
    if e_quota is not None and len(error_pairs) != e_quota:
        raise MatchingFailure("_materialise produced %d error candidates but the request was %d"
                              % (len(error_pairs), e_quota))
    if c_quota is not None and e_quota is not None:
        for name, ids in (("R", r_ids), ("S", s_ids)):
            if len(ids) != c_quota + e_quota:
                raise MatchingFailure("_materialise produced K_%s=%d but the request was %d"
                                      % (name, len(ids), c_quota + e_quota))
    return r_ids, s_ids, correct_common, error_pairs


def construct_matched_subsets(candidates: Sequence[Candidate], k: int, seed=0,
                              k_ladder: Iterable[int] = K_LADDER, *,
                              target_signatures: Iterable[str] = TARGET_SIGNATURES,
                              error_fraction=DEFAULT_ERROR_FRACTION,
                              token_tolerance: Optional[int] = DEFAULT_TOKEN_TOLERANCE) -> dict:
    """Build the R and S final subsets for one block at size `k`.

    Returns a record with both subsets, the hard-constraint audit, the
    denominators (N+, N-), the derived final_TPR/final_FPR/yield/precision, and
    an explicit infeasibility certificate when the constraints cannot be met.
    Raises MatchingFailure only for a malformed request; an infeasible pool is a
    *returned* failure so the caller can record it and try the next ladder step.
    """
    target_signatures = _validated_targets(target_signatures)
    try:
        c_quota, e_quota = quotas(k, error_fraction)
    except ValueError as exc:
        raise MatchingFailure(str(exc)) from exc
    if token_tolerance is not None and (isinstance(token_tolerance, bool) or not isinstance(token_tolerance, int)
                                        or token_tolerance < 0):
        raise MatchingFailure("token_tolerance must be a nonnegative integer or None, not %r" % (token_tolerance,))
    # The pool's draws are keyed by the whole pool; an answer cut at max_new_tokens then takes no
    # role, so no matched subset holds a truncated answer.  N+ and N- count the usable answers.
    draw_key = pool_key(candidates)
    truncated = sum(1 for c in candidates if c.truncated)
    candidates = [c for c in candidates if not c.truncated]
    views = _views(candidates, target_signatures, token_tolerance)
    by_id = {c.candidate_id: c for c in candidates}
    n_plus = sum(1 for c in candidates if c.correct)
    n_minus = len(candidates) - n_plus
    n_plus_tasks = sum(1 for v in views.values() if v.has_correct)
    n_eligible_tasks = sum(1 for v in views.values() if v.eligible)

    # Necessary conditions, reported verbatim so a failure is diagnosable.
    certificate = {"K": k, "C": c_quota, "E": e_quota, "error_fraction": error_fraction,
                   "token_tolerance": token_tolerance, "tasks_in_pool": len(views),
                   "tasks_with_correct": n_plus_tasks, "tasks_eligible_J_E": n_eligible_tasks,
                   "N_plus": n_plus, "N_minus": n_minus, "truncated_excluded": truncated,
                   "condition_one_task_each": c_quota + e_quota <= len(views),
                   "condition_correct_tasks": c_quota <= n_plus_tasks,
                   "condition_error_tasks": e_quota <= n_eligible_tasks}
    if not all((certificate["condition_one_task_each"], certificate["condition_correct_tasks"],
                certificate["condition_error_tasks"])):
        certificate["feasible"] = False
        certificate["reason"] = "necessary condition failed"
        return {"feasible": False, "R": [], "S": [], "certificate": certificate}

    assignment = _solve(views, c_quota, e_quota, seed, draw_key=draw_key)
    if assignment is None:
        certificate["feasible"] = False
        certificate["reason"] = "no role assignment satisfies the quotas"
        return {"feasible": False, "R": [], "S": [], "certificate": certificate}

    r_ids, s_ids, correct_common, error_pairs = _materialise(views, assignment, by_id, seed,
                                                            c_quota=c_quota, e_quota=e_quota, draw_key=draw_key)

    # --- hard-constraint audit -------------------------------------------------
    def rows(ids: FrozenSet[str]) -> List[Candidate]:
        return [by_id[i] for i in sorted(ids)]

    r_rows, s_rows = rows(r_ids), rows(s_ids)
    r_tasks = {c.task_id for c in r_rows}
    s_tasks = {c.task_id for c in s_rows}
    r_correct = {c.candidate_id for c in r_rows if c.correct}
    s_correct = {c.candidate_id for c in s_rows if c.correct}
    r_errors = [c for c in r_rows if not c.correct]
    s_errors = [c for c in s_rows if not c.correct]

    audit = {
        "K_R": len(r_rows), "K_S": len(s_rows),
        "K_equal": len(r_rows) == len(s_rows) == k,
        "C_R": len(r_correct), "C_S": len(s_correct),
        "E_R": len(r_errors), "E_S": len(s_errors),
        "error_fraction": error_fraction,
        "ratio_C_to_E": len(r_correct) == c_quota and len(r_errors) == e_quota
                        and len(s_correct) == c_quota and len(s_errors) == e_quota,
        "same_task_ids": r_tasks == s_tasks,
        "same_correct_ids": r_correct == s_correct,
        "weights_all_one": all(getattr(c, "weight", 1.0) == 1.0 for c in r_rows + s_rows) if r_rows and s_rows else True,
        "updates": 1,
    }
    tokens_r = {c.task_id: c.tokens for c in r_rows}
    tokens_s = {c.task_id: c.tokens for c in s_rows}
    audit["task_token_counts_within_tolerance"] = (set(tokens_r) == set(tokens_s) and all(
        _within(tokens_r[t], tokens_s[t], token_tolerance) for t in tokens_r))
    audit["token_length_tolerance"] = token_tolerance
    audit["max_token_mismatch"] = max((abs(tokens_r.get(t, -1) - tokens_s.get(t, -1)) for t in r_tasks | s_tasks),
                                      default=0)
    audit["error_signature_pairs"] = [{"R": by_id[a].error, "S": by_id[b].error,
                                       "same_candidate": a == b} for a, b in error_pairs]
    audit["S_target_hits"] = sum(1 for c in s_errors if c.error in target_signatures)
    audit["R_target_hits"] = sum(1 for c in r_errors if c.error in target_signatures)
    audit["N_plus"] = n_plus
    audit["N_minus"] = n_minus
    audit["final_TPR"] = len(r_correct) / n_plus if n_plus else None
    audit["final_FPR"] = len(r_errors) / n_minus if n_minus else None
    audit["final_TPR_equal"] = audit["final_TPR"] == (len(s_correct) / n_plus if n_plus else None)
    audit["final_FPR_equal"] = audit["final_FPR"] == (len(s_errors) / n_minus if n_minus else None)
    audit["yield"] = k / (n_plus + n_minus) if n_plus + n_minus else None
    audit["precision"] = len(r_correct) / k
    audit["quota_target"] = "not used in the main path; the legacy quota target is a separate legacy quantity"

    # Validate hard constraints before returning feasible=True
    all_hard_constraints_met = (
        audit["K_equal"] and
        audit["ratio_C_to_E"] and
        audit["same_task_ids"] and
        audit["same_correct_ids"] and
        audit["task_token_counts_within_tolerance"] and
        audit["weights_all_one"] and
        audit["updates"] == 1
    )

    if not all_hard_constraints_met:
        certificate["feasible"] = False
        certificate["reason"] = "hard constraints violated after materialisation"
        return {"feasible": False, "R": [], "S": [], "certificate": certificate, "audit": audit}

    return {"feasible": True, "R": [c.candidate_id for c in r_rows], "S": [c.candidate_id for c in s_rows],
            "certificate": certificate, "audit": audit,
            "hash": digest({"K": k, "R": sorted(r_ids), "S": sorted(s_ids)})}


def _validated_targets(target_signatures):
    if isinstance(target_signatures, str):
        raise MatchingFailure("target_signatures must be a collection, not one string")
    targets = tuple(target_signatures)
    if not targets or any(not isinstance(t, str) or not t.strip() for t in targets):
        raise MatchingFailure("target_signatures must contain nonempty error names")
    if len(set(targets)) != len(targets):
        raise MatchingFailure("target_signatures must not contain duplicates")
    return targets


def select_k_for_blocks(blocks: Dict[str, Sequence[Candidate]], seed=0,
                        k_ladder: Iterable[int] = K_LADDER, *,
                        target_signatures: Iterable[str] = TARGET_SIGNATURES,
                        error_fraction=DEFAULT_ERROR_FRACTION,
                        token_tolerance: Optional[int] = DEFAULT_TOKEN_TOLERANCE) -> dict:
    """Pick the largest ladder size that every block can satisfy.

    All blocks move together: a size is chosen only if it is feasible for all of
    them.  If even the smallest step fails, this returns feasible=False with the
    per-block certificates and the caller must stop before training anything.
    """
    target_signatures = _validated_targets(target_signatures)
    attempts = []
    for k in k_ladder:
        per_block = {name: construct_matched_subsets(cands, k, seed, target_signatures=target_signatures,
                                                     error_fraction=error_fraction, token_tolerance=token_tolerance)
                     for name, cands in sorted(blocks.items())}
        ok = all(r["feasible"] for r in per_block.values())
        attempts.append({"K": k, "feasible": ok,
                         "certificates": {n: r["certificate"] for n, r in per_block.items()}})
        if ok:
            return {"feasible": True, "K": k, "seed": seed, "error_fraction": error_fraction,
                    "token_tolerance": token_tolerance, "per_block": per_block, "attempts": attempts}
    return {"feasible": False, "K": None, "seed": seed, "error_fraction": error_fraction,
            "token_tolerance": token_tolerance, "per_block": {}, "attempts": attempts,
            "reason": "no ladder size is feasible for every block; stop before training (no seed change, "
                      "no tolerance beyond the configured one)"}


def expected_common_correct_overlap(a_correct: FrozenSet[str], b_correct: FrozenSet[str]) -> dict:
    """Coverage check for the J_C / J_E overlap counterexample.

    The construction may place a task in J_C for one block and in J_E for
    another block only when the task satisfies both roles; within a block the
    roles are exclusive.  This helper exists so tests and the report can state
    the overlap explicitly instead of implying it away.
    """
    shared = a_correct & b_correct
    return {"shared_correct_ids": sorted(shared), "shared_count": len(shared),
            "only_in_first": sorted(a_correct - b_correct), "only_in_second": sorted(b_correct - a_correct)}
