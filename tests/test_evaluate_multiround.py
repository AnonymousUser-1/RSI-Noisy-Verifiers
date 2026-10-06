"""evaluate_multiround.py and plot_multiround.py on a fixture run, with the mock backend.

The mock model answers correctly with the probability its adapter's mock.json gives (0.55 without
an adapter), so Pass@1 follows the checkpoint; a wrong graph answer is a non-shortest walk 70% of
the time.  HFBackend's greedy switch is checked on a stub model.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import evaluate_multiround
import plot_multiround
from generate_data import generate
from rsi.backends import HFBackend
from rsi.common import digest, read_json, read_jsonl, verify_dataset, write_json
from rsi.tasks import judge

EVAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "matched_evaluation.json"
ROUNDS = 2
P = {("b00", "R", 1): 0.6, ("b00", "R", 2): 0.7, ("b00", "S", 1): 0.5, ("b00", "S", 2): 0.4,
     ("b01", "R", 1): 0.62, ("b01", "R", 2): 0.75, ("b01", "S", 1): 0.52, ("b01", "S", 2): 0.45,
     ("b02", "R", 1): 0.58, ("b02", "R", 2): 0.72, ("b02", "S", 1): 0.48}  # b02/S round 2 still training
CONFIG = {"backend": "mock", "model": "Qwen/Qwen3-1.7B", "revision": "a" * 40, "dtype": "bfloat16",
          "device": "cpu", "rounds": ROUNDS,
          "generation": {"batch_size": 16, "candidates": 4, "max_new_tokens": 256, "max_sequence_length": 2048,
                         "temperature": 1.3, "top_p": 1.0, "top_k": None}}


def make_run(root, data, config=CONFIG):
    run = root / "out"
    for (block, arm, t), p in P.items():
        arm_dir = run / block / arm
        write_json(arm_dir / "run.json", {"config": config, "data_path": str(data), "task": "graph",
                                          "dataset_hash": digest(verify_dataset(data))})
        write_json(arm_dir / ("round_%03d" % t) / "adapter" / "mock.json", {"p": p})
        write_json(arm_dir / ("round_%03d" % t) / "complete.json", {"adapter": "round_%03d/adapter" % t})
    return run


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.data = cls.root / "data"
        generate(cls.data, "graph", 7, ROUNDS, 8, 4, 4, 400, 200)
        cls.run_dir = make_run(cls.root, cls.data)
        with mock.patch.object(evaluate_multiround, "backend", wraps=evaluate_multiround.backend) as loads:
            cls.results = quiet(evaluate_multiround.evaluate_run, cls.run_dir)
        cls.loads = loads.call_count
        cls.out = cls.run_dir / "evaluation_greedy_eval_id"
        cls.ood = cls.run_dir / "evaluation_greedy_eval_ood"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_round_zero_and_every_completed_checkpoint(self):
        names = sorted(p.stem for p in self.out.glob("*.json") if p.stem != "protocol")
        expected = sorted(["round_000"] + ["%s_%s_round_%03d" % key for key in P])
        self.assertEqual(names, expected)
        self.assertFalse((self.out / "b02_S_round_002.json").exists())

    def test_summary_matches_the_judged_answers(self):
        tasks = {t["id"]: t for t in read_jsonl(self.data / "eval_id.jsonl")}
        for name in ("round_000", "b00_R_round_002", "b01_S_round_001"):
            summary = read_json(self.out / (name + ".json"))
            rows = read_jsonl(self.out / (name + ".jsonl"))
            self.assertEqual(len(rows), len(tasks))
            self.assertEqual(len({r["task_id"] for r in rows}), len(tasks))
            correct = [judge(tasks[r["task_id"]], r["response"])["correct"] for r in rows]
            self.assertAlmostEqual(summary["pass1"], sum(correct) / len(rows))
            nonshortest = sum(r["error"] == "nonshortest" for r in rows)
            self.assertAlmostEqual(summary["target_error_rate"], nonshortest / len(rows))
            self.assertEqual(sum(summary["questions_by_difficulty"].values()), len(tasks))

    def test_both_held_out_splits_by_default_each_checkpoint_loaded_once(self):
        for out, split, questions in ((self.out, "eval_id", 400), (self.ood, "eval_ood", 200)):
            protocol = read_json(out / "protocol.json")
            self.assertEqual((protocol["split"], protocol["questions"]), (split, questions))
            names = sorted(p.stem for p in out.glob("*.json") if p.stem != "protocol")
            self.assertEqual(len(names), 1 + len(P))
            self.assertEqual(read_json(out / "b01_S_round_002.json")["split"], split)
        self.assertEqual(len(self.results), 2 * (1 + len(P)))
        self.assertEqual(self.loads, 1 + len(P))

    def test_one_split_on_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), self.data)
            config = Path(tmp) / "evaluation.json"
            config.write_text(json.dumps({"splits": ["eval_ood"], "batch_size": 32}))
            self.assertEqual(quiet(evaluate_multiround.main, ["--run", str(run), "--config", str(config)]), 0)
            self.assertTrue((run / "evaluation_greedy_eval_ood" / "round_000.json").exists())
            self.assertFalse((run / "evaluation_greedy_eval_id").exists())

    def test_out_of_memory_halves_the_batch_and_keeps_it(self):
        real = evaluate_multiround.backend
        seen = []

        def tight(cfg, adapter=None):
            model = real(cfg, adapter=adapter)
            generate = model.generate

            def limited(tasks, candidates, seed):
                seen.append(cfg["generation"]["batch_size"])
                if cfg["generation"]["batch_size"] > 8:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return generate(tasks, candidates, seed)
            model.generate = limited
            return model

        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), self.data)
            with mock.patch.object(evaluate_multiround, "backend", side_effect=tight):
                results = quiet(evaluate_multiround.evaluate_run, run, ["eval_id"], 32, [1])
            self.assertEqual(seen[:4], [32, 16, 8, 8])
            self.assertTrue(all(r["batch_size"] == 8 for r in results))
            self.assertTrue(all(r["questions"] == 400 for r in results))

    def test_out_of_memory_in_a_later_batch_keeps_the_batches_done(self):
        real = evaluate_multiround.backend
        calls = []

        def flaky(cfg, adapter=None):
            model = real(cfg, adapter=adapter)
            generate = model.generate

            def third_fails(tasks, candidates, seed):
                calls.append([t["id"] for t in tasks])
                if len(calls) == 3:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return generate(tasks, candidates, seed)
            model.generate = third_fails
            return model

        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), self.data)
            with mock.patch.object(evaluate_multiround, "backend", side_effect=flaky):
                # rounds=[99]: no such round, so only round 0 is evaluated.
                results = quiet(evaluate_multiround.evaluate_run, run, ["eval_id"], 32, [99])
            self.assertEqual([len(c) for c in calls[:5]], [32, 32, 32, 16, 16])
            self.assertEqual(calls[3] + calls[4], calls[2])      # the failed batch again, in two halves
            done = [i for n, c in enumerate(calls) if n != 2 for i in c]
            self.assertEqual(len(done), 400)                      # no batch generated twice
            self.assertEqual(len(set(done)), 400)
            rows = read_jsonl(run / "evaluation_greedy_eval_id" / "round_000.jsonl")
            self.assertEqual(sorted(r["task_id"] for r in rows), sorted(done))
            self.assertEqual([r["batch_size"] for r in results], [16])

    def test_other_runtime_errors_are_not_retried(self):
        def broken(cfg, adapter=None):
            model = mock.Mock()
            model.generate.side_effect = RuntimeError("device-side assert triggered")
            return model

        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), self.data)
            with mock.patch.object(evaluate_multiround, "backend", side_effect=broken), \
                    self.assertRaisesRegex(RuntimeError, "device-side"):
                quiet(evaluate_multiround.evaluate_run, run, ["eval_id"])

    def test_each_checkpoint_is_evaluated_with_its_own_adapter(self):
        pass1 = {(r["block"], r["arm"], r["round"]): r["pass1"] for r in self.results
                 if r["round"] and r["split"] == "eval_id"}
        self.assertGreater(pass1[("b00", "R", 2)], pass1[("b00", "S", 2)] + 0.15)
        base = read_json(self.out / "round_000.json")
        self.assertIsNone(base["adapter"])
        self.assertAlmostEqual(base["pass1"], 0.55, delta=0.08)

    def test_an_arm_without_an_adapter_is_round_zero(self):
        """A one-step run's null arm (run_matched_experiment.py) saves no adapter: it is the base."""
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), self.data)
            write_json(run / "b00" / "null" / "run.json", read_json(run / "b00" / "R" / "run.json"))
            write_json(run / "b00" / "null" / "round_001" / "complete.json", {"adapter": None})
            results = quiet(evaluate_multiround.evaluate_run, run, ["eval_id"], 32, [1])
            self.assertFalse(any(r["arm"] == "null" for r in results))
            self.assertTrue((run / "evaluation_greedy_eval_id" / "round_000.json").exists())

    def test_evaluation_json_may_state_the_loop_stop(self):
        stop = {"span": 128, "max_period": 32}
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), self.data)
            config = Path(tmp) / "evaluation.json"
            config.write_text(json.dumps({"splits": ["eval_id"], "batch_size": 32, "repetition_stop": stop}))
            self.assertEqual(evaluate_multiround.read_evaluation_config(config)["repetition_stop"], stop)
            self.assertEqual(quiet(evaluate_multiround.main, ["--run", str(run), "--config", str(config),
                                                             "--rounds", "1"]), 0)
            protocol = read_json(run / "evaluation_greedy_eval_id" / "protocol.json")
            self.assertEqual(protocol["decoding"]["repetition_stop"], stop)     # the run's config has none
            config.write_text(json.dumps({"splits": ["eval_id"], "batch_size": 32, "repetition_stop": {"span": 3}}))
            with self.assertRaisesRegex(ValueError, "repetition_stop"):
                evaluate_multiround.read_evaluation_config(config)

    def test_protocol_is_greedy_one_answer(self):
        protocol = read_json(self.out / "protocol.json")
        self.assertEqual(protocol["decoding"]["temperature"], 0.0)
        self.assertEqual(protocol["answers_per_question"], 1)
        self.assertEqual(protocol["target_signatures"], ["nonshortest"])
        self.assertEqual(protocol["questions"], 400)

    def test_rerun_skips_finished_checkpoints(self):
        with mock.patch.object(evaluate_multiround, "backend", side_effect=AssertionError("generated again")):
            again = quiet(evaluate_multiround.evaluate_run, self.run_dir, batch_size=8)
        self.assertEqual(len(again), len(self.results))

    def test_other_settings_are_refused(self):
        protocol = read_json(self.out / "protocol.json")
        try:
            write_json(self.out / "protocol.json", dict(protocol, decoding=dict(protocol["decoding"], temperature=1.0)))
            with self.assertRaisesRegex(ValueError, "other settings .*decoding"):
                quiet(evaluate_multiround.evaluate_run, self.run_dir)
        finally:
            write_json(self.out / "protocol.json", protocol)

    def test_a_changed_adapter_is_refused(self):
        path = self.run_dir / "b01" / "R" / "round_001" / "adapter" / "mock.json"
        try:
            write_json(path, {"p": 0.99})
            with self.assertRaisesRegex(ValueError, "another adapter"):
                quiet(evaluate_multiround.evaluate_run, self.run_dir)
        finally:
            write_json(path, {"p": P[("b01", "R", 1)]})

    def test_arms_must_share_the_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), self.data)
            manifest = read_json(run / "b01" / "S" / "run.json")
            manifest["config"] = dict(manifest["config"], revision="b" * 40)
            write_json(run / "b01" / "S" / "run.json", manifest)
            with self.assertRaisesRegex(ValueError, "another base"):
                quiet(evaluate_multiround.evaluate_run, run)

    def test_cli_refuses_a_directory_without_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(quiet(evaluate_multiround.main, ["--run", tmp, "--config", str(EVAL_CONFIG)]), 1)

    def test_the_splits_and_batch_size_come_from_the_evaluation_config(self):
        self.assertEqual(evaluate_multiround.read_evaluation_config(EVAL_CONFIG), json.loads(EVAL_CONFIG.read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ({}, {"batch_size": 8}, {"splits": ["eval_id"], "batch_size": 0},
                        {"splits": [], "batch_size": 8}, {"splits": ["train_001"], "batch_size": 8},
                        {"splits": ["eval_id", "eval_id"], "batch_size": 8}, {"splits": "eval_id", "batch_size": 8},
                        {"splits": ["eval_id"], "batch_size": 8, "temperature": 0.7}):
                path = Path(tmp) / "evaluation.json"
                path.write_text(json.dumps(bad))
                with self.subTest(config=bad), self.assertRaises(ValueError):
                    evaluate_multiround.read_evaluation_config(path)
                self.assertEqual(quiet(evaluate_multiround.main, ["--run", str(self.run_dir), "--config", str(path)]), 1)

    def test_answers_cut_at_max_new_tokens_are_counted(self):
        rows = read_jsonl(self.out / "round_000.jsonl")
        self.assertEqual(read_json(self.out / "round_000.json")["truncated_answers"], 0)
        cut = [dict(r, truncated=i < 3) for i, r in enumerate(rows)]
        tasks = {t["id"]: t for t in read_jsonl(self.data / "eval_id.jsonl")}
        self.assertEqual(evaluate_multiround.summarize(cut, tasks, ["nonshortest"])["truncated_answers"], 3)

    def test_plot_and_tables(self):
        prefix = self.root / "figures" / "fixture"
        self.assertEqual(quiet(plot_multiround.main, ["--run", str(self.run_dir), "--label", "unaudited",
                                                      "--run", str(self.run_dir), "--label", "copy",
                                                      "--out", str(prefix)]), 0)
        for split in ("eval_id", "eval_ood"):
            for suffix in ("_pass1.png", "_by_difficulty.png", "_summary.json", "_summary.csv"):
                self.assertTrue(Path("%s_%s%s" % (prefix, split, suffix)).stat().st_size > 0, (split, suffix))
        ood = json.loads(Path(str(prefix) + "_eval_ood_summary.json").read_text())["pass1"]["unaudited"]
        self.assertAlmostEqual(ood["arms"]["S"]["1"]["mean"], sum(
            read_json(self.ood / ("%s_S_round_001.json" % b))["pass1"] for b in ("b00", "b01", "b02")) / 3)
        tables = json.loads(Path(str(prefix) + "_eval_id_summary.json").read_text())
        pass1 = tables["pass1"]["unaudited"]
        values = [read_json(self.out / ("%s_R_round_002.json" % b))["pass1"] for b in ("b00", "b01", "b02")]
        self.assertAlmostEqual(pass1["arms"]["R"]["2"]["mean"], sum(values) / 3)
        self.assertEqual(pass1["arms"]["R"]["2"]["n"], 3)
        # Round 2 of S has two blocks; S - R pairs only the blocks both arms finished.
        self.assertEqual(pass1["s_minus_r"]["2"]["blocks"], ["b00", "b01"])
        self.assertEqual(pass1["s_minus_r"]["0"]["mean"], 0.0)
        base = read_json(self.out / "round_000.json")["pass1"]
        self.assertEqual(pass1["arms"]["S"]["0"]["mean"], base)
        self.assertEqual(pass1["arms"]["S"]["0"]["half_width"], 0.0)


class IntervalTests(unittest.TestCase):
    def test_t_interval(self):
        got = plot_multiround.interval([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(got["mean"], 0.3)
        self.assertAlmostEqual(got["sd"], 0.158113883, places=6)
        self.assertAlmostEqual(got["half_width"], 2.776445105 * 0.158113883 / 5 ** 0.5, places=6)
        self.assertEqual(plot_multiround.interval([0.4])["half_width"], 0.0)


class StubModel:
    def __init__(self):
        self.calls = []
        self.generation_config = type("G", (), {"eos_token_id": 9})()
        self.config = type("C", (), {})()

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        return torch.cat([input_ids, torch.tensor([[5, 9]] * len(input_ids))], dim=1)


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    chat_template = "stub"

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def decode(self, tokens, skip_special_tokens=True):
        return "[\"v1\"]"


class GreedySwitchTests(unittest.TestCase):
    def sample(self, temperature):
        b = HFBackend.__new__(HFBackend)
        b.torch, b.device, b.model, b.tokenizer = torch, "cpu", StubModel(), StubTokenizer()
        b.config = {"generation": {"batch_size": 4, "max_new_tokens": 8, "max_sequence_length": 64,
                                   "temperature": temperature, "top_p": 1.0, "top_k": None}}
        with contextlib.redirect_stdout(io.StringIO()):
            rows = b.generate([{"id": "t1", "prompt": "p"}], 1, 0)
        self.assertEqual(rows[0]["completion_tokens"], 2)
        return b.model.calls[0]

    def test_temperature_zero_is_greedy(self):
        call = self.sample(0.0)
        self.assertFalse(call["do_sample"])
        self.assertIsNone(call["temperature"])
        self.assertIsNone(call["top_p"])
        self.assertIsNone(call["top_k"])

    def test_positive_temperature_samples_as_before(self):
        call = self.sample(1.3)
        self.assertTrue(call["do_sample"])
        self.assertEqual((call["temperature"], call["top_p"], call["top_k"]), (1.3, 1.0, None))


class CudaCacheCapTests(unittest.TestCase):
    """rsi.backends.limit_cuda_cache: Windows only, CUDA only, no computation changed."""

    def fake_torch(self):
        calls = []
        cuda = type("Cuda", (), {"set_per_process_memory_fraction": staticmethod(lambda f, d: calls.append((f, d))),
                                 "current_device": staticmethod(lambda: 0)})
        return type("Torch", (), {"cuda": cuda, "device": staticmethod(torch.device)}), calls

    def test_windows_cuda_caps_the_cache_on_an_indexed_device(self):
        from rsi.backends import CUDA_CACHE_FRACTION, limit_cuda_cache
        torch_, calls = self.fake_torch()
        self.assertEqual(limit_cuda_cache(torch_, "cuda", platform="win32"), CUDA_CACHE_FRACTION)
        self.assertEqual(limit_cuda_cache(torch_, "cuda:1", platform="win32"), CUDA_CACHE_FRACTION)
        self.assertEqual(calls, [(CUDA_CACHE_FRACTION, 0), (CUDA_CACHE_FRACTION, 1)])

    def test_linux_and_cpu_are_untouched(self):
        from rsi.backends import limit_cuda_cache
        torch_, calls = self.fake_torch()
        self.assertIsNone(limit_cuda_cache(torch_, "cuda", platform="linux"))
        self.assertIsNone(limit_cuda_cache(torch_, "cpu", platform="win32"))
        self.assertEqual(calls, [])

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_real_torch_accepts_the_call(self):
        from rsi.backends import limit_cuda_cache
        self.assertIsNotNone(limit_cuda_cache(torch, "cuda", platform="win32"))


if __name__ == "__main__":
    unittest.main()
