"""Per-arm selection for rounds 2+ of the multi-round extension (rsi.iterative).

From round 2 each arm samples its own pool from its own checkpoint, so R and S
no longer share candidates and the round-1 joint matching does not apply.  What
stays matched is the count: K examples, C:E = 3:1, one example per prompt, all
from the round's common prompt set.  R takes its errors uniformly among the
prompts that have any error; S takes them among the prompts that have a
`nonshortest` error.  The rest is uniform.
"""
import unittest
from collections import Counter

from rsi.iterative import select_own_subset
from rsi.matching import Candidate


def pool(tasks=60, with_target=30, correct_every=1):
    """Task i: a correct answer (every `correct_every`-th task), a nonedge error, and a
    nonshortest error for the first `with_target` tasks."""
    rows = []
    for i in range(tasks):
        task = "t%03d" % i
        if i % correct_every == 0:
            rows.append(Candidate("%s-c" % task, task, "c", True, "correct", 5))
        rows.append(Candidate("%s-n" % task, task, "n", False, "nonedge", 5))
        if i < with_target:
            rows.append(Candidate("%s-s" % task, task, "s", False, "nonshortest", 7))
    return rows


class SelectOwnSubsetTests(unittest.TestCase):
    def test_counts_are_matched_one_example_per_prompt(self):
        for arm in ("R", "S"):
            result = select_own_subset(pool(), arm, 16, seed=0, label=("b00", 2, arm))
            self.assertTrue(result["feasible"], arm)
            rows = result["rows"]
            self.assertEqual(len(rows), 16)
            self.assertEqual(sum(r.correct for r in rows), 12)
            self.assertEqual(len({r.task_id for r in rows}), 16)
            self.assertEqual([r.task_id for r in rows], sorted(r.task_id for r in rows))
            audit = result["audit"]
            self.assertEqual((audit["K"], audit["C"], audit["E"]), (16, 12, 4))

    def test_s_takes_only_target_errors_and_r_takes_any_error(self):
        s = select_own_subset(pool(), "S", 16, seed=0, label="x")
        self.assertEqual({r.error for r in s["rows"] if not r.correct}, {"nonshortest"})
        signatures = Counter()
        for seed in range(20):
            r = select_own_subset(pool(), "R", 16, seed=seed, label="x")
            signatures.update(c.error for c in r["rows"] if not c.correct)
        self.assertEqual(set(signatures), {"nonedge", "nonshortest"})

    def test_rates_use_the_arm_s_own_pool_as_denominator(self):
        candidates = pool()
        result = select_own_subset(candidates, "S", 16, seed=0, label="x")
        n_plus = sum(c.correct for c in candidates)
        n_minus = len(candidates) - n_plus
        audit = result["audit"]
        self.assertEqual((audit["N_plus"], audit["N_minus"]), (n_plus, n_minus))
        self.assertEqual(audit["final_TPR"], 12 / n_plus)
        self.assertEqual(audit["final_FPR"], 4 / n_minus)
        self.assertEqual(audit["precision"], 0.75)
        self.assertEqual(audit["target_hits"], 4)

    def test_the_draw_is_fixed_by_seed_and_label(self):
        a = select_own_subset(pool(), "R", 16, seed=3, label=("b00", 2, "R"))
        b = select_own_subset(pool(), "R", 16, seed=3, label=("b00", 2, "R"))
        c = select_own_subset(pool(), "R", 16, seed=3, label=("b00", 3, "R"))
        self.assertEqual(a["ids"], b["ids"])
        self.assertNotEqual(a["ids"], c["ids"])

    def test_too_few_target_prompts_is_infeasible_not_relaxed(self):
        result = select_own_subset(pool(with_target=3), "S", 16, seed=0, label="x")
        self.assertFalse(result["feasible"])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["certificate"]["error_prompts_available"], 3)
        self.assertIn("error", result["certificate"]["reason"])
        self.assertTrue(select_own_subset(pool(with_target=3), "R", 16, seed=0, label="x")["feasible"])

    def test_too_few_correct_prompts_is_infeasible(self):
        result = select_own_subset(pool(tasks=20, correct_every=5), "R", 16, seed=0, label="x")
        self.assertFalse(result["feasible"])
        self.assertIn("correct", result["certificate"]["reason"])

    def test_k_must_split_three_to_one(self):
        with self.assertRaises(ValueError):
            select_own_subset(pool(), "R", 10, seed=0, label="x")
        with self.assertRaises(ValueError):
            select_own_subset(pool(), "null", 16, seed=0, label="x")


if __name__ == "__main__":
    unittest.main()


class ArithmeticTargetTests(unittest.TestCase):
    """Arithmetic: S's target is ignore_parentheses (rsi.matching.targets_for), not nonshortest."""

    def arithmetic_pool(self, tasks=60, with_target=30):
        rows = []
        for i in range(tasks):
            task = "a%03d" % i
            rows.append(Candidate("%s-c" % task, task, "c", True, "correct", 3))
            rows.append(Candidate("%s-o" % task, task, "o", False, "other", 3))
            if i < with_target:
                rows.append(Candidate("%s-p" % task, task, "p", False, "ignore_parentheses", 3))
        return rows

    def test_targets_follow_the_task(self):
        from rsi.matching import targets_for
        self.assertEqual(targets_for("graph"), ("nonshortest",))
        self.assertEqual(targets_for("arithmetic"), ("ignore_parentheses",))
        with self.assertRaises(ValueError):
            targets_for("chess")

    def test_s_takes_only_ignore_parentheses_errors(self):
        from rsi.matching import targets_for
        result = select_own_subset(self.arithmetic_pool(), "S", 16, seed=0, label="x",
                                   target_signatures=targets_for("arithmetic"))
        self.assertTrue(result["feasible"])
        self.assertEqual({r.error for r in result["rows"] if not r.correct}, {"ignore_parentheses"})
        self.assertEqual(result["audit"]["target_hits"], 4)
        # With the graph default, an arithmetic pool has no target errors at all.
        self.assertFalse(select_own_subset(self.arithmetic_pool(), "S", 16, seed=0, label="x")["feasible"])
