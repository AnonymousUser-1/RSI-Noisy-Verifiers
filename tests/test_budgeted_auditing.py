"""Budget, label isolation, original-pool rates and durable audit reuse."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rsi import auditing
from rsi.budgeted_auditing import apply_audit, enabled, positive_rows, statistics
from rsi.common import read_json, write_json


class BudgetedAuditingTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {str(i): {"task": "arithmetic", "expression": "1+1", "difficulty": str(i % 2)}
                      for i in range(8)}
        self.rows = [{"id": str(i), "task_id": str(i), "response": "2" if i < 6 else "3",
                      "correct": i < 6, "error": None if i < 6 else "other", "weight": 1.0}
                     for i in range(8)]
        self.counts = {"N_plus": 12, "N_minus": 8}

    def run_audit(self, policy="uniform", budget=3, weighting=False, **kwargs):
        return apply_audit(self.rows, self.tasks, {"policy": policy, "budget": budget, "weighting": weighting},
                           2027, 1, self.counts, **kwargs)

    def test_each_policy_queries_only_its_budget_and_removes_audited_errors(self):
        for policy in ("uniform", "balanced", "adaptive"):
            for budget in (1, 3, 64):
                with self.subTest(policy=policy, budget=budget), \
                        mock.patch("rsi.auditing.judge", wraps=auditing.judge) as oracle:
                    rows, report = self.run_audit(policy, budget)
                    self.assertEqual(oracle.call_count, min(budget, len(self.rows)))
                    self.assertEqual(report["queries"], oracle.call_count)
                    bad = {x["candidate_id"] for x in report["labels"] if not x["correct"]}
                    self.assertEqual({x["id"] for x in rows}, {x["id"] for x in self.rows} - bad)
                    self.assertFalse(report["post_audit_matching_guaranteed"])

    def test_no_audit_is_a_zero_cost_identity_even_with_nonuniform_weights(self):
        self.rows[0]["weight"] = 0.25
        for policy in ("none", "uniform", "balanced", "adaptive"):
            with mock.patch("rsi.auditing.judge") as oracle:
                rows, report = self.run_audit(policy, 0)
                self.assertEqual(rows, self.rows)
                self.assertEqual(report["queries"], 0)
                self.assertFalse(report["enabled"])
                oracle.assert_not_called()

    def test_allocator_cannot_see_current_labels_or_error_taxonomy(self):
        with mock.patch("rsi.budgeted_auditing.audit_pool", wraps=auditing.audit_pool) as allocator:
            self.run_audit("adaptive")
        visible = allocator.call_args.args[0]
        self.assertTrue(all(set(r) == {"id", "task_id", "response", "weight"} for r in visible))

    def test_rates_keep_original_pool_denominators_after_removal(self):
        rows, report = self.run_audit(budget=64)
        self.assertEqual((report["before"]["K"], report["before"]["C"], report["before"]["E"]), (8, 6, 2))
        self.assertEqual(report["before"]["final_TPR"], 6 / 12)
        self.assertEqual(report["before"]["final_FPR"], 2 / 8)
        self.assertEqual(report["after"]["final_TPR"], 6 / 12)
        self.assertEqual(report["after"]["final_FPR"], 0)
        self.assertEqual(report["after"]["yield"], 6 / 20)
        self.assertEqual(report["after"]["precision"], 1)
        self.assertEqual(report["after"]["effective_sample_size"], len(rows))

    def test_weighting_changes_optimizer_coverage_not_retained_counts(self):
        # Audit one known error in each stratum; the posterior weight is zero.
        def choose_errors(population, k):
            return [r for r in population if r["id"] in ("6", "7")][:k]

        with mock.patch("random.Random.sample", side_effect=choose_errors):
            rows, report = self.run_audit("balanced", 2, True)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["weight"] == 0 for r in rows))
        self.assertEqual(report["after"]["C"], 6)
        self.assertEqual(report["after"]["positive_weight_correct"], 0)
        self.assertEqual(report["after"]["positive_weight_TPR"], 0)

    def test_completed_ledger_reuses_labels_without_another_oracle_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "ledger.json"
            rows, report = self.run_audit(cache_path=cache)
            with mock.patch("rsi.auditing.judge") as oracle:
                again, reused = self.run_audit(cache_path=cache)
                oracle.assert_not_called()
            self.assertEqual(rows, again)
            self.assertEqual(report["queries"], reused["queries"])
            self.assertTrue(reused["reused"])

    def test_changed_ledger_inputs_are_refused_instead_of_reaudited(self):
        for field in ("response", "task", "history", "budget", "counts"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                cache = Path(tmp) / "ledger.json"
                self.setUp()
                self.run_audit(cache_path=cache)
                kwargs = {"cache_path": cache}
                if field == "response":
                    self.rows[0]["response"] = "4"
                elif field == "task":
                    self.tasks["0"]["expression"] = "2+2"
                elif field == "history":
                    kwargs["previous"] = {"0": {"errors": 1, "correct": 1}}
                elif field == "budget":
                    kwargs["budget"] = 4
                else:
                    self.counts["N_plus"] += 1
                with mock.patch("rsi.auditing.judge") as oracle, self.assertRaisesRegex(ValueError, "inputs changed"):
                    self.run_audit(**kwargs)
                oracle.assert_not_called()

    def test_tampered_ledger_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "ledger.json"
            self.run_audit(cache_path=cache)
            saved = read_json(cache)
            saved["report"]["queries"] = 0
            write_json(cache, saved)
            with self.assertRaisesRegex(ValueError, "payload changed"):
                self.run_audit(cache_path=cache)

    def test_duplicate_ids_and_invalid_weights_fail_before_queries(self):
        self.rows.append(dict(self.rows[0]))
        with mock.patch("rsi.auditing.judge") as oracle, self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_audit()
        oracle.assert_not_called()
        for weight in (-1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                positive_rows([dict(self.rows[0], weight=weight)])

    def test_bad_config_is_rejected(self):
        for policy, budget, weighting in (("unknown", 2, False), ("none", 1, False),
                                           ("uniform", -1, False), ("uniform", 1.5, False),
                                           ("uniform", True, False), ("uniform", 1, "false")):
            with self.subTest(policy=policy, budget=budget), self.assertRaises(ValueError):
                enabled({"policy": policy, "budget": budget, "weighting": weighting})

    def test_empty_pool_statistics_are_explicit(self):
        report = statistics([], {"N_plus": 0, "N_minus": 0})
        self.assertIsNone(report["final_TPR"])
        self.assertIsNone(report["final_FPR"])
        self.assertIsNone(report["precision"])
        self.assertEqual(report["weight_sum"], 0)
        self.assertEqual(report["effective_sample_size"], 0)
