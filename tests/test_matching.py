"""Regression test for the matched-subset cardinality defect.

At afea0f6, `rsi/matching.py:_materialise` added a correct id to *both*
branches for every assigned task, and an error id on top for error-role tasks.
An error-role task therefore contributed two candidates per branch, and both
branches came out at ``c_quota + 2 * e_quota`` rather than ``c_quota + e_quota``:

    request K=16 (C=12, E=4)  ->  K_R = K_S = 20, C_R = 16, E_R = 4

The certificate still reported ``K=16`` and the result was still returned as
``feasible: True``.  b78c994 fixed the loop  and added a post-hoc
audit that returns ``feasible: False``; `_materialise` now also checks its own
output against the request and raises, so a size bug stops the run instead of
reading as an infeasible pool and stepping the K ladder down.  These tests pin
the cardinality and the guard.
"""
import unittest

from rsi.matching import Candidate, MatchingFailure, construct_matched_subsets


def block(task_count, length=4):
    """`task_count` tasks, each with 1 correct + 1 nonshortest + 1 format error.

    All three responses sit at the *same* supervised token count, which is what
    makes a task eligible for the J_E role: the matched construction requires a
    length that carries both a target-signature error and a non-target one.
    """
    candidates = []
    for t in range(task_count):
        task_id = "task%02d" % t
        candidates.append(Candidate("%s-c" % task_id, task_id, "[]", True, "correct", length))
        candidates.append(Candidate("%s-n" % task_id, task_id, "[a]", False, "nonshortest", length))
        candidates.append(Candidate("%s-f" % task_id, task_id, "x", False, "format", length))
    return candidates


class MatchedSubsetCardinalityTests(unittest.TestCase):
    def test_subset_size_matches_the_requested_k(self):
        """K_R and K_S must be exactly `k`; the afea0f6 loop gave ``c_quota + 2 * e_quota``."""
        for k in (16, 32):
            with self.subTest(k=k):
                result = construct_matched_subsets(block(40), k, seed=0)
                self.assertTrue(result["feasible"], result["certificate"])
                self.assertEqual(len(result["R"]), k)
                self.assertEqual(len(result["S"]), k)

    def test_correct_and_error_counts_match_the_3_to_1_split(self):
        """The C:E = 3:1 quotas are what `_materialise` has to deliver."""
        result = construct_matched_subsets(block(40), 16, seed=0)
        audit = result["audit"]
        self.assertEqual(audit["C_R"], 12)
        self.assertEqual(audit["E_R"], 4)
        self.assertEqual(audit["K_R"], 16)
        self.assertTrue(audit["ratio_C_to_E"])
        self.assertTrue(audit["K_equal"])

    def test_materialise_refuses_to_return_a_size_it_was_not_asked_for(self):
        """The guard itself: a mismatched request raises rather than returning."""
        from rsi.matching import _materialise, _views, pool_key

        candidates = block(16)
        views = _views(candidates)
        by_id = {c.candidate_id: c for c in candidates}
        assignment = {"%s" % c.task_id: "correct" for c in candidates if c.correct}
        # 16 correct-role tasks but a request of only 12 correct ids.
        with self.assertRaises(MatchingFailure):
            _materialise(views, assignment, by_id, 0, c_quota=12, e_quota=4, draw_key=pool_key(candidates))


if __name__ == "__main__":
    unittest.main()
