"""Real budgeted oracle queries with GPU generation/training stubbed.

CPU toy-gradient tests separately exercise the real one-step optimizer. These
tests cover both runner entry points, cost records, weights, adaptive history,
and interrupted-training resume, not scientific model performance.
"""
import math
from pathlib import Path
from unittest import mock

import one_step
import run_iterative_experiment
from rsi import auditing
from rsi.budgeted_auditing import positive_rows
from rsi.common import digest, load_config, read_json, read_jsonl, write_json
from rsi.experiment import pin_config
import test_run_iterative_experiment as iterative_fixtures
from test_run_iterative_experiment import IterFixture
from test_run_matched_experiment import BLOCKS, Fixture, CONFIG as MOCK_CONFIG, run_entry
from test_run_matched_experiment_hf import HFFixture, CONFIG as HF_CONFIG

from pin_isolation import isolate as setUpModule, restore as tearDownModule  # noqa: F401


def audit_config(directory, original, budget, policy="adaptive", weighting=True):
    path = Path(directory) / "audit_config.json"
    write_json(path, dict(read_json(original), audit={"policy": policy, "budget": budget, "weighting": weighting}))
    return path


def stub_weighted_train(config, adapter_in, rows, out_adapter, seed, h_path):
    active = positive_rows(rows)
    Path(out_adapter).mkdir(parents=True)
    write_json(Path(out_adapter) / "stub.json", {"DEMO_ONLY": True})
    steps = math.ceil(len(active) / config["training"]["effective_batch_size"]) * config["training"]["epochs"]
    return {"trained": bool(active), "steps": steps, "mean_loss": 0.5 if active else None,
            "delta_theta_norm": 0.1 if active else 0.0, "h_T_delta_theta": 0.01 if active else 0.0}


class MatchedAuditTests(Fixture):
    def test_mock_runner_actually_audits_both_arms_but_not_the_frozen_null(self):
        scratch = self.scratch()
        config = audit_config(scratch, MOCK_CONFIG, 64, weighting=False)
        out = scratch / "audited"
        with mock.patch("rsi.auditing.judge", wraps=auditing.judge) as oracle:
            code, _, message = run_entry(self.data, self.pools, out, config=str(config))
        self.assertEqual(code, 0, message)
        self.assertEqual(oracle.call_count, len(BLOCKS) * 2 * 16)
        for block in BLOCKS:
            for arm in ("R", "S", "null"):
                directory = out / block / arm
                complete = read_json(directory / "round_001/complete.json")
                self.assertEqual(complete["audit_queries"], 0 if arm == "null" else 16)
                self.assertEqual(complete["examples"], 16 if arm == "null" else 12)
                self.assertEqual(complete["selected_examples"], 16)
                self.assertEqual(read_json(directory / "finished.json")["cumulative_audit_queries"], complete["audit_queries"])
                rows = read_jsonl(directory / "round_001/training.jsonl")
                self.assertTrue(all(not ({"correct", "error"} & r.keys()) for r in rows))
                if arm != "null":
                    report = read_json(directory / "round_001/audit.json")
                    self.assertEqual((report["before"]["C"], report["before"]["E"]), (12, 4))
                    self.assertEqual((report["after"]["C"], report["after"]["E"]), (12, 0))
                else:
                    self.assertFalse((directory / "round_001/audit.json").exists())

    def test_strict_standalone_entry_still_refuses_auditing(self):
        config = load_config(MOCK_CONFIG, 0, "mock")
        config["audit"] = {"policy": "uniform", "budget": 2, "weighting": False}
        with self.assertRaises(ValueError):
            one_step.check_isolation(config)
        one_step.check_isolation(config, allow_auditing=True)


class HFAuditTests(HFFixture):
    def test_audited_rows_and_weights_reach_the_hf_one_step_arm(self):
        scratch = self.scratch()
        config = audit_config(scratch, HF_CONFIG, 2)
        resolved = pin_config(load_config(config, 0, "hf"))
        adapter, reference = self.write_inputs(scratch / "inputs", h_config=digest(resolved))
        seen = {}

        def capture(branch, rows, cfg, adapter_dir, output, seed, recorder_kwargs):
            seen[Path(output).parent.parent.name, branch] = rows
            return self.fake_arm(branch, rows, cfg, adapter_dir, output, seed, recorder_kwargs)

        with mock.patch("rsi.auditing.judge", wraps=auditing.judge) as oracle:
            code, _, message = self.run_hf(scratch / "out", arm=capture, config=str(config),
                                           shared_adapter=str(adapter), reference_gradient=str(reference))
        self.assertEqual(code, 0, message)
        self.assertEqual(oracle.call_count, 12)
        for block in BLOCKS:
            for arm in ("R", "S"):
                report = read_json(scratch / "out" / block / arm / "round_001/audit.json")
                self.assertEqual(len(seen[block, arm]), report["after"]["K"])
                self.assertAlmostEqual(sum(r["weight"] for r in seen[block, arm]), report["after"]["weight_sum"])
            self.assertTrue(all(r["weight"] == 1 for r in seen[block, "null"]))

    def test_reference_binding_requires_the_same_audit_enabled_config(self):
        scratch = self.scratch()
        config = audit_config(scratch, HF_CONFIG, 2)
        with mock.patch("rsi.auditing.judge") as oracle:
            code, _, message = self.run_hf(scratch / "out", config=str(config))
        self.assertNotEqual(code, 0)
        self.assertIn("config", message)
        oracle.assert_not_called()


class IterAuditTests(IterFixture):
    def setup_audit(self, budget=2, weighting=True):
        scratch = self.scratch()
        config = audit_config(scratch, self.config, budget, weighting=weighting)
        adapter, reference = self.write_inputs(scratch / "inputs", h_config=digest(pin_config(load_config(config, 0, "hf"))))
        return scratch / "out", {"config": str(config), "shared_adapter": str(adapter), "reference_gradient": str(reference)}

    def test_per_round_and_cumulative_budgets_and_adaptive_history_are_per_arm(self):
        out, kwargs = self.setup_audit()
        with mock.patch("rsi.auditing.judge", wraps=auditing.judge) as oracle, \
                mock.patch("rsi.budgeted_auditing.audit_pool", wraps=auditing.audit_pool) as allocator:
            code, message = self.run_entry(out, train_arm=stub_weighted_train, **kwargs)
        self.assertEqual(code, 0, message)
        self.assertEqual(oracle.call_count, len(BLOCKS) * 2 * 4 * 2)
        calls = iter(allocator.call_args_list)
        for block in BLOCKS:
            for t in range(1, 5):
                for arm in ("R", "S"):
                    round_dir = out / block / arm / ("round_%03d" % t)
                    complete, audit = read_json(round_dir / "complete.json"), read_json(round_dir / "audit.json")
                    self.assertEqual((complete["audit_queries"], complete["cumulative_audit_queries"]), (2, 2 * t))
                    self.assertEqual(complete["selected_examples"], 16)
                    self.assertEqual(complete["examples"], audit["after"]["K"])
                    self.assertEqual(complete["training"]["steps"], complete["expected_optimizer_steps"])
                    self.assertEqual(complete["expected_optimizer_steps"], math.ceil(audit["after"]["positive_weight_examples"] / 4))
                    previous = {} if t == 1 else read_json(out / block / arm / ("round_%03d/complete.json" % (t - 1)))["audit_groups"]
                    self.assertEqual(next(calls).args[5], previous)
                    rows = read_jsonl(round_dir / "training.jsonl")
                    self.assertTrue(all(not ({"correct", "error"} & r.keys()) for r in rows))
            for arm in ("R", "S"):
                self.assertEqual(read_json(out / block / arm / "finished.json")["cumulative_audit_queries"], 8)

    def test_deletions_change_steps_without_violating_the_pre_audit_quota(self):
        out, kwargs = self.setup_audit(64, weighting=False)
        code, message = self.run_entry(out, train_arm=stub_weighted_train, **kwargs)
        self.assertEqual(code, 0, message)
        self.assertEqual(read_json(out / "experiment.json")["steps_per_round"], 4)
        for block in BLOCKS:
            for arm in ("R", "S"):
                for t in range(1, 5):
                    complete = read_json(out / block / arm / ("round_%03d/complete.json" % t))
                    self.assertEqual((complete["selected_examples"], complete["examples"], complete["training"]["steps"]), (16, 12, 3))
                    self.assertEqual(complete["cumulative_audit_queries"], 16 * t)

    def test_resume_reuses_completed_audit_after_interrupted_training(self):
        out, kwargs = self.setup_audit()
        generated = []

        def nondeterministic(config, adapter, tasks, samples, seed):
            generated.append(str(adapter))
            # Every call permutes the candidate IDs carrying each answer. A
            # resample of the interrupted round would invalidate its audit.
            return iterative_fixtures.fixture_pool(tasks, samples, seed + len(generated), adapter)

        def crash(config, adapter_in, rows, out_adapter, seed, h_path):
            if Path(out_adapter).as_posix().endswith("b00/R/round_002/adapter"):
                raise RuntimeError("interrupted training after completed audit")
            return stub_weighted_train(config, adapter_in, rows, out_adapter, seed, h_path)

        with mock.patch("rsi.auditing.judge", wraps=auditing.judge) as oracle:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_entry(out, generate_pool=nondeterministic, train_arm=crash, **kwargs)
            self.assertEqual(oracle.call_count, 6)
            code, message = self.run_entry(out, generate_pool=nondeterministic, train_arm=stub_weighted_train, resume=True, **kwargs)
            self.assertEqual(code, 0, message)
            self.assertEqual(oracle.call_count, 48)
            self.assertEqual(len(generated), 18)  # no resampling of the audited interrupted round
            self.assertTrue(read_json(out / "b00/R/round_002/audit.json")["reused"])
            code, message = self.run_entry(out, train_arm=stub_weighted_train, resume=True, **kwargs)
            self.assertEqual(code, 0, message)
            self.assertEqual(oracle.call_count, 48)  # finished rounds also never requery

    def test_changed_pool_ledger_is_refused_before_generation_or_queries(self):
        out, kwargs = self.setup_audit()

        def crash(config, adapter_in, rows, out_adapter, seed, h_path):
            if Path(out_adapter).as_posix().endswith("b00/R/round_002/adapter"):
                raise RuntimeError("interrupt")
            return stub_weighted_train(config, adapter_in, rows, out_adapter, seed, h_path)

        with self.assertRaises(RuntimeError):
            self.run_entry(out, train_arm=crash, **kwargs)
        cache = out / "b00/R/audit_ledger/round_002.pool.json"
        saved = read_json(cache)
        saved["pool"][0]["response"] = "[]"
        # If the first response is already [], make sure this is a real change.
        saved["pool"][0]["response"] += "tampered"
        write_json(cache, saved)
        with mock.patch("rsi.auditing.judge") as oracle, \
                mock.patch.object(run_iterative_experiment, "generate_pool") as generator, \
                self.assertRaisesRegex(ValueError, "pool ledger"):
            # run_entry has its own generator patch, so explicitly supply this mock.
            self.run_entry(out, generate_pool=generator, train_arm=stub_weighted_train, resume=True, **kwargs)
        oracle.assert_not_called()
        generator.assert_not_called()

    def test_all_zero_weight_post_audit_subset_preserves_next_round_checkpoint(self):
        out, kwargs = self.setup_audit()
        real = run_iterative_experiment.apply_audit

        def zero_weight(*args, **kw):
            rows, report = real(*args, **kw)
            # Wiring regression: emulate an auditor returning no optimizer signal.
            return [dict(r, weight=0) for r in rows], report

        with mock.patch.object(run_iterative_experiment, "apply_audit", side_effect=zero_weight):
            code, message = self.run_entry(out, train_arm=stub_weighted_train, **kwargs)
        self.assertEqual(code, 0, message)
        for block in BLOCKS:
            for arm in ("R", "S"):
                for t in range(1, 5):
                    directory = out / block / arm / ("round_%03d" % t)
                    complete = read_json(directory / "complete.json")
                    self.assertEqual(complete["expected_optimizer_steps"], 0)
                    self.assertEqual(complete["training"]["steps"], 0)
                    self.assertTrue((directory / "adapter").is_dir())


class ArithmeticAuditedIterTests(iterative_fixtures.ArithmeticIterTests):
    def test_arithmetic_uses_the_same_budgeted_intervention_in_all_rounds(self):
        scratch = self.scratch()
        config = audit_config(scratch, self.config, 64, weighting=False)
        adapter, reference = self.write_inputs(scratch / "inputs", h_config=digest(pin_config(load_config(config, 0, "hf"))))
        out = scratch / "out"
        with mock.patch("rsi.auditing.judge", wraps=auditing.judge) as oracle:
            code, message = self.run_entry(out, generate_pool=self.fake_arithmetic_generate,
                                           train_arm=stub_weighted_train,
                                           config=str(config), shared_adapter=str(adapter), reference_gradient=str(reference))
        self.assertEqual(code, 0, message)
        k = read_json(out / "experiment.json")["K"]
        self.assertEqual(oracle.call_count, len(BLOCKS) * 2 * 4 * k)
        for block in BLOCKS:
            for arm in ("R", "S"):
                for t in range(1, 5):
                    report = read_json(out / block / arm / ("round_%03d/audit.json" % t))
                    self.assertEqual(report["queries"], k)
                    self.assertEqual(report["after"]["E"], 0)
                    self.assertEqual(report["after"]["C"], 3 * k // 4)
