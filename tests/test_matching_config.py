"""matching.error_fraction and matching.token_tolerance: the C:E ratio and the per-prompt token tolerance
of the R/S selection are config fields.  Their defaults (0.25, i.e. 3:1, and 0) keep the frozen rule."""
import argparse
import json
import random
import tempfile
import unittest
from pathlib import Path

import one_step
from rsi.common import load_config, require_explicit
from rsi.iterative import select_own_subset
from rsi.matching import Candidate, construct_matched_subsets, quotas, select_k_for_blocks
from run_matched_experiment import check_branch_rows


def task(t, target_tokens=12, other_tokens=(12,), correct=4, correct_tokens=10):
    """One prompt's 8 answers: `correct` correct ones, one target error and non-target errors."""
    rows = [Candidate("c%d_%d" % (t, s), "t%03d" % t, "ok %d %d" % (t, s), True, "correct", correct_tokens)
            for s in range(correct)]
    rows.append(Candidate("c%d_t" % t, "t%03d" % t, "detour %d" % t, False, "nonshortest", target_tokens))
    rows += [Candidate("c%d_o%d" % (t, i), "t%03d" % t, "nonedge %d %d" % (t, i), False, "nonedge", n)
             for i, n in enumerate(other_tokens)]
    return rows


def pool(n=120, **kwargs):
    return [c for t in range(n) for c in task(t, **kwargs)]


class QuotaTests(unittest.TestCase):
    def test_the_fraction_sets_the_wrong_answers(self):
        self.assertEqual(quotas(64), (48, 16))
        self.assertEqual(quotas(64, 0.125), (56, 8))
        self.assertEqual(quotas(16, 0.5), (8, 8))
        self.assertEqual(quotas(16, 0.0625), (15, 1))
        for k, fraction in ((64, 0.2), (16, 0.03125), (64, 0.0), (64, 1.0), (16, 1.5), (64.0, 0.25), (64, True)):
            with self.subTest(k=k, fraction=fraction), self.assertRaises(ValueError):
                quotas(k, fraction)

    def test_the_product_is_exact_and_the_quotas_are_ints(self):
        self.assertEqual(quotas(100, 0.07), (93, 7))
        self.assertTrue(all(type(q) is int for q in quotas(64, 0.125)))


class RoundOneTests(unittest.TestCase):
    def test_the_error_fraction_sets_C_and_E_in_both_arms(self):
        for fraction, (c, e) in ((0.25, (48, 16)), (0.125, (56, 8)), (0.5, (32, 32))):
            with self.subTest(fraction=fraction):
                result = construct_matched_subsets(pool(), 64, seed=3, error_fraction=fraction)
                self.assertTrue(result["feasible"], result["certificate"])
                audit = result["audit"]
                self.assertEqual((audit["C_R"], audit["E_R"], audit["C_S"], audit["E_S"]), (c, e, c, e))
                self.assertTrue(audit["ratio_C_to_E"])
                self.assertEqual(result["certificate"]["error_fraction"], fraction)

    def test_a_tolerance_admits_errors_within_it_and_no_further(self):
        # Every prompt: S's target error has 12 tokens, the other errors 15 and 40.
        candidates = pool(other_tokens=(15, 40))
        for tolerance, feasible in ((0, False), (2, False), (3, True), (27, True), (None, True)):
            with self.subTest(tolerance=tolerance):
                result = construct_matched_subsets(candidates, 16, seed=1, token_tolerance=tolerance)
                self.assertEqual(result["feasible"], feasible, result["certificate"])
                if not feasible:
                    continue
                audit, by_id = result["audit"], {c.candidate_id: c for c in candidates}
                self.assertTrue(audit["task_token_counts_within_tolerance"])
                self.assertEqual(audit["token_length_tolerance"], tolerance)
                r_errors = [by_id[i] for i in result["R"] if not by_id[i].correct]
                s_errors = [by_id[i] for i in result["S"] if not by_id[i].correct]
                self.assertTrue(all(c.error == "nonshortest" for c in s_errors))
                if tolerance == 3:   # R's error is S's own (12) or the 15-token one, never the 40-token one
                    self.assertTrue(all(c.tokens in (12, 15) for c in r_errors))
                    self.assertLessEqual(audit["max_token_mismatch"], 3)

    def test_the_ladder_moves_together_with_the_settings(self):
        blocks = {"b00": pool(other_tokens=(15,)), "b01": pool(other_tokens=(15,))}
        self.assertFalse(select_k_for_blocks(blocks, seed=0)["feasible"])
        result = select_k_for_blocks(blocks, seed=0, error_fraction=0.125, token_tolerance=3)
        self.assertEqual((result["feasible"], result["K"]), (True, 64))
        self.assertEqual((result["error_fraction"], result["token_tolerance"]), (0.125, 3))

    def test_the_default_draws_are_main_s(self):
        """With the defaults the selection is the one the code made before the settings existed (checked
        against main on 2,500 random pools when they were added); this pins one of them."""
        rng = random.Random(7)
        candidates = []
        for t in range(150):
            base = rng.randint(3, 30)
            for s in range(8):
                correct = rng.random() < 0.4
                candidates.append(Candidate("c%d_%d" % (t, s), "t%03d" % t, "r %d %d %f" % (t, s, rng.random()),
                                            correct, "correct" if correct else rng.choice(
                                                ["nonshortest", "nonedge", "endpoint"]), base + rng.randint(0, 4)))
        result = construct_matched_subsets(candidates, 64, seed=5)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["hash"], construct_matched_subsets(candidates, 64, seed=5, error_fraction=0.25,
                                                                   token_tolerance=0)["hash"])
        self.assertEqual(result["hash"], "ee37b2066724ecc055fbcf23a2fdf987ce72e5c3b07c698e7275d861091196b0")


class SDrawTests(unittest.TestCase):
    def test_at_a_tolerance_S_draws_among_all_qualifying_targets(self):
        """One error prompt: targets of 10 and 12 tokens, other errors of 10, 11 and 13; three correct-only
        prompts fill C.  At tolerance 0 only the 10-token target qualifies (the frozen rule); at a
        tolerance both do, and S takes either, not always the shorter one."""
        rows = [Candidate("ok", "t0", "ok", True, "correct", 5),
                Candidate("T10", "t0", "a", False, "nonshortest", 10), Candidate("T12", "t0", "b", False, "nonshortest", 12),
                Candidate("N10", "t0", "c", False, "nonedge", 10), Candidate("N11", "t0", "d", False, "nonedge", 11),
                Candidate("N13", "t0", "e", False, "nonedge", 13)]
        rows += [Candidate("ok%d" % i, "c%d" % i, "ok %d" % i, True, "correct", 5) for i in range(3)]
        by_id = {c.candidate_id: c for c in rows}
        for tolerance, expected in ((0, {"T10"}), (2, {"T10", "T12"}), (None, {"T10", "T12"})):
            with self.subTest(tolerance=tolerance):
                picks = []
                for seed in range(200):
                    result = construct_matched_subsets(rows, 4, seed=seed, token_tolerance=tolerance)
                    self.assertTrue(result["feasible"])
                    s_error = next(i for i in result["S"] if not by_id[i].correct)
                    r_error = next(i for i in result["R"] if not by_id[i].correct)
                    if tolerance is not None:
                        self.assertLessEqual(abs(by_id[s_error].tokens - by_id[r_error].tokens), tolerance)
                    picks.append(s_error)
                self.assertEqual(set(picks), expected)
                if len(expected) == 2:
                    self.assertGreater(min(picks.count(p) for p in expected), 50)


class RecordTests(unittest.TestCase):
    def test_the_matching_records_carry_the_settings(self):
        from match_candidates import write_matching
        blocks = {"b00": pool(other_tokens=(15,))}
        result = select_k_for_blocks(blocks, seed=0, error_fraction=0.125, token_tolerance=3)
        with tempfile.TemporaryDirectory() as tmp:
            write_matching(tmp, blocks, [64, 32, 16], result)
            for name in ("ladder.json", "matched_subsets.json"):
                record = json.loads((Path(tmp) / name).read_text())
                self.assertEqual((record["error_fraction"], record["token_tolerance"]), (0.125, 3), name)

    def test_a_one_step_run_refuses_a_block_matched_with_other_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = json.loads((Path(__file__).resolve().parents[1] / "configs" / "matched_pool.json").read_text())
            raw["matching"] = {"error_fraction": 0.125, "token_tolerance": 0}
            (tmp / "config.json").write_text(json.dumps(raw))
            (tmp / "data").mkdir()
            (tmp / "data" / "train_001.jsonl").write_text("")
            (tmp / "pool.jsonl").write_text("")
            (tmp / "matched.json").write_text(json.dumps({"per_block": {"b00": {
                "R": [], "S": [], "certificate": {"error_fraction": 0.25, "token_tolerance": 0}}}}))
            args = argparse.Namespace(config=str(tmp / "config.json"), data=str(tmp / "data"), pool=str(tmp / "pool.jsonl"),
                                      matched=str(tmp / "matched.json"), block=None, shared_adapter=str(tmp),
                                      out=str(tmp / "out"), seed=0, backend="mock")
            with self.assertRaises(SystemExit) as refused:
                one_step.main(args)
            self.assertIn("was matched with", str(refused.exception))
            self.assertFalse((tmp / "out").exists())


class LaterRoundTests(unittest.TestCase):
    def test_the_error_fraction_sets_each_arm_s_own_selection(self):
        for arm in ("R", "S"):
            with self.subTest(arm=arm):
                selection = select_own_subset(pool(), arm, 64, 0, ("b00", 2, arm), error_fraction=0.125)
                self.assertTrue(selection["feasible"])
                rows = [c for c in pool() if c.candidate_id in set(selection["ids"])]
                self.assertEqual((sum(c.correct for c in rows), sum(not c.correct for c in rows)), (56, 8))


class TrainedRowsTests(unittest.TestCase):
    def rows(self, correct, wrong, tokens):
        return ([{"id": "c%d" % i, "task_id": "t%d" % i, "correct": True, "weight": one_step.REQUIRED_WEIGHT,
                  "tokens": 10} for i in range(correct)] +
                [{"id": "e%d_%d" % (i, tokens), "task_id": "t%d" % (correct + i), "correct": False,
                  "weight": one_step.REQUIRED_WEIGHT, "tokens": tokens} for i in range(wrong)])

    def test_the_rows_an_arm_trains_on_are_checked_with_the_settings(self):
        r, s = self.rows(56, 8, 12), self.rows(56, 8, 15)
        check_branch_rows("b00", r, s, 64, error_fraction=0.125, token_tolerance=3)
        check_branch_rows("b00", r, s, 64, error_fraction=0.125, token_tolerance=None)
        for kwargs in ({"error_fraction": 0.125, "token_tolerance": 2}, {"token_tolerance": 3}):
            with self.subTest(**kwargs), self.assertRaises(SystemExit):
                check_branch_rows("b00", r, s, 64, **kwargs)


class ConfigTests(unittest.TestCase):
    def load(self, matching, stage=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            raw = json.loads((Path(__file__).resolve().parents[1] / "configs" / "matched_iterative.json").read_text())
            if matching is None:
                del raw["matching"]
            else:
                raw["matching"] = matching
            path.write_text(json.dumps(raw))
            if stage:
                require_explicit(path, stage)
            return load_config(path)

    def test_valid_and_invalid_settings(self):
        self.assertEqual(self.load({"error_fraction": 0.125, "token_tolerance": None})["matching"],
                         {"error_fraction": 0.125, "token_tolerance": None})
        for bad in ({"error_fraction": 0.2, "token_tolerance": 0}, {"error_fraction": True, "token_tolerance": 0},
                    {"error_fraction": 0.25, "token_tolerance": -1}, {"error_fraction": 0.25, "token_tolerance": 1.5},
                    {"error_fraction": 0.25, "token_tolerance": 0, "extra": 1}):
            with self.subTest(matching=bad), self.assertRaises(ValueError):
                self.load(bad)

    def test_the_selection_stages_require_both_fields(self):
        for stage in ("one_step", "iterative", "pilot"):
            with self.subTest(stage=stage), self.assertRaises(ValueError) as refused:
                self.load(None, stage)
            self.assertIn("matching.error_fraction", str(refused.exception))
        self.load(None, "pool")   # sampling does not read them
        with self.assertRaises(ValueError) as one_missing:
            self.load({"error_fraction": 0.25}, "iterative")
        self.assertIn("matching.token_tolerance", str(one_missing.exception))


if __name__ == "__main__":
    unittest.main()
