"""Round-1 matching draws its prompts per block (matched-dynamics/DECISION.md, 2026-10-02).

Equally constrained prompts used to be taken in task-id order.  Every prompt of an 8-answer
pool has 8 candidates, so that order decided which prompts filled the quotas: every seed block
filled its correct quota with the same lowest-id prompts, and the blocks shared their correct
examples (12 of 12 at K = 16 on the H100 Qwen3-1.7B graph pools; the matching seed changed
nothing).  The order is now a random draw keyed by the seed and the block's own pool.
"""
import random
import unittest
from collections import Counter

from rsi.matching import Candidate, _solve, _views, construct_matched_subsets, pool_key, select_k_for_blocks

CORRECT_TASKS, ERROR_TASKS = 80, 20


def pool(block, correct_tasks=CORRECT_TASKS, error_tasks=ERROR_TASKS):
    """One seed block's pool over shared prompts, as real seed blocks are: every prompt is answered
    in every block, and the sampled responses differ from block to block.

    c000...: two correct answers of one length (they can take the correct role only);
    e000...: no correct answer, a nonshortest and a format error of one length (J_E, error role only).
    """
    rng = random.Random(block)
    rows = []
    for t in range(correct_tasks):
        task = "c%03d" % t
        rows += [Candidate("%s-%d" % (task, s), task, "path %d" % rng.randrange(10 ** 9), True, "correct", 5)
                 for s in range(2)]
    for t in range(error_tasks):
        task = "e%03d" % t
        rows.append(Candidate(task + "-0", task, "detour %d" % rng.randrange(10 ** 9), False, "nonshortest", 6))
        rows.append(Candidate(task + "-1", task, "junk %d" % rng.randrange(10 ** 9), False, "format", 6))
    return rows


def role_tasks(result, candidates, correct):
    """The prompts R holds in the given role (S holds the same prompts)."""
    by_id = {c.candidate_id: c for c in candidates}
    return {by_id[i].task_id for i in result["R"] if by_id[i].correct == correct}


class SeedBlockTests(unittest.TestCase):
    def test_seed_blocks_draw_their_own_prompts(self):
        blocks = {"b%02d" % b: pool(b) for b in range(3)}
        result = select_k_for_blocks(blocks, seed=0, k_ladder=(16,))
        self.assertTrue(result["feasible"])
        correct = {name: role_tasks(result["per_block"][name], blocks[name], True) for name in blocks}
        errors = {name: role_tasks(result["per_block"][name], blocks[name], False) for name in blocks}
        for name, block in result["per_block"].items():
            audit = block["audit"]
            for constraint in ("K_equal", "ratio_C_to_E", "same_task_ids", "same_correct_ids",
                               "task_token_counts_within_tolerance"):
                self.assertTrue(audit[constraint], (name, constraint))
            self.assertEqual((len(correct[name]), len(errors[name])), (12, 4))
            # Not the lowest task ids, which every block used to take.
            self.assertNotEqual(sorted(correct[name]), ["c%03d" % t for t in range(12)], name)
            self.assertNotEqual(sorted(errors[name]), ["e%03d" % t for t in range(4)], name)
        # 12 of 80 drawn per block: about 0.3 prompts in all three by chance (task-id order: 12).
        self.assertLessEqual(len(set.intersection(*correct.values())), 3)
        self.assertLessEqual(len(set.intersection(*errors.values())), 2)

    def test_a_blocks_draw_depends_on_its_pool_and_the_seed_only(self):
        candidates = pool(0)
        first = construct_matched_subsets(candidates, 16, seed=0)
        # The block's name is not an input, and neither is the order of its rows.
        renamed = select_k_for_blocks({"b07": list(reversed(candidates))}, seed=0, k_ladder=(16,))
        again = renamed["per_block"]["b07"]
        self.assertEqual((first["R"], first["S"], first["hash"]), (again["R"], again["S"], again["hash"]))
        self.assertEqual(pool_key(candidates), pool_key(list(reversed(candidates))))
        # Another seed, or another block's pool over the same prompts, is another draw.
        for other_seed, other_pool in ((1, candidates), (0, pool(1))):
            other = construct_matched_subsets(other_pool, 16, seed=other_seed)
            self.assertNotEqual(role_tasks(other, other_pool, True), role_tasks(first, candidates, True))

    def test_every_prompt_is_drawn_about_equally_often(self):
        candidates = pool(0)
        correct, errors = Counter(), Counter()
        for seed in range(200):
            result = construct_matched_subsets(candidates, 16, seed=seed)
            correct.update(role_tasks(result, candidates, True))
            errors.update(role_tasks(result, candidates, False))
        # 12 of 80 correct prompts per draw: 30 draws each on average (sd 5); task-id order gave
        # the same 12 prompts 200 times and the other 68 never.
        self.assertEqual(len(correct), CORRECT_TASKS)
        self.assertTrue(10 < min(correct.values()) <= max(correct.values()) < 55, correct)
        # 4 of 20 error prompts per draw: 40 each on average (sd 6).
        self.assertEqual(len(errors), ERROR_TASKS)
        self.assertTrue(15 < min(errors.values()) <= max(errors.values()) < 70, errors)


def flexible(task):
    """A task that can take either role: a correct answer, and a target and a non-target error at one length."""
    return [Candidate(task + "-c", task, task + " c", True, "correct", 5),
            Candidate(task + "-n", task, task + " n", False, "nonshortest", 6),
            Candidate(task + "-f", task, task + " f", False, "format", 6)]


def correct_only(task):
    return [Candidate(task + "-c", task, task + " c", True, "correct", 5)]


def error_only(task):
    return [Candidate(task + "-n", task, task + " n", False, "nonshortest", 6),
            Candidate(task + "-f", task, task + " f", False, "format", 6)]


class OrderTests(unittest.TestCase):
    """The random order only breaks ties: tasks with one legal role still go before flexible ones."""

    def solve(self, candidates, c_quota, e_quota, seed):
        return _solve(_views(candidates), c_quota, e_quota, seed, draw_key=pool_key(candidates))

    def test_the_documented_counterexample_has_one_answer(self):
        """A flexible, B correct only, C = E = 1: the only valid assignment, which the complete
        search finds whatever the order."""
        candidates = flexible("A") + correct_only("B")
        for seed in range(20):
            self.assertEqual(self.solve(candidates, 1, 1, seed), {"B": "correct", "A": "error"})

    def test_one_role_tasks_go_before_flexible_ones(self):
        """With an error-only task D added there are three valid assignments; most constrained first
        takes B and D and keeps the flexible A in reserve, under every draw."""
        candidates = flexible("A") + correct_only("B") + error_only("D")
        for seed in range(50):
            self.assertEqual(self.solve(candidates, 1, 1, seed), {"B": "correct", "D": "error"})

    def test_flexible_tasks_only_fill_what_one_role_tasks_cannot(self):
        candidates = [row for t in range(12) for row in correct_only("c%02d" % t)]
        candidates += [row for t in range(4) for row in flexible("f%02d" % t)]
        candidates += [row for t in range(4) for row in error_only("e%02d" % t)]
        for seed in range(50):
            assignment = self.solve(candidates, 12, 4, seed)
            self.assertFalse([t for t in assignment if t.startswith("f")], seed)
        # One error-only task short: exactly one flexible task takes the error role.
        short = [c for c in candidates if c.task_id != "e03"]
        for seed in range(50):
            assignment = self.solve(short, 12, 4, seed)
            self.assertEqual([role for t, role in assignment.items() if t.startswith("f")], ["error"], seed)


if __name__ == "__main__":
    unittest.main()
