"""No answer cut at max_new_tokens is ever trained on; a pool's size and settings come from its config.

A generated row records `truncated` when no stop id came within max_new_tokens (rsi/backends.py).
The row is judged and kept in its pool, but round-1 matching, own-pool selection in later rounds,
the legacy loop's verifier and the training entry points never use it, and evaluation counts it.
`generation.prompts_per_pool` is the number of prompts a pool samples (null: the whole split).
"""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import one_step
import sample_candidates
from generate_data import generate
from rsi import backends
from rsi.common import DEFAULTS, load_config, read_json, read_jsonl, require_explicit
from rsi.iterative import select_own_subset
from rsi.matching import Candidate, candidate_from_row, construct_matched_subsets


def candidates(truncate=()):
    """t00-t09 error-only (no correct answer), t10-t49 with a correct answer and a J_E error pair.

    Every id in `truncate` is a row cut at max_new_tokens."""
    rows = []
    for t in range(50):
        task = "t%02d" % t
        if t >= 10:
            rows.append(Candidate(task + "-c", task, task + " c", True, "correct", 5, truncated=task + "-c" in truncate))
        rows.append(Candidate(task + "-n", task, task + " n", False, "nonshortest", 6, truncated=task + "-n" in truncate))
        rows.append(Candidate(task + "-f", task, task + " f", False, "format", 6))
    return rows


# The correct answers of t10-t14 and the target errors of t00-t04 are cut: 35 prompts keep a correct answer.
CUT = {"t%02d-c" % t for t in range(10, 15)} | {"t%02d-n" % t for t in range(5)}


class SelectionTests(unittest.TestCase):
    def test_round_one_matching_never_takes_a_truncated_answer(self):
        pool = candidates(CUT)
        for seed in range(10):
            result = construct_matched_subsets(pool, 16, seed=seed)
            self.assertTrue(result["feasible"], result["certificate"])
            self.assertFalse((set(result["R"]) | set(result["S"])) & CUT, seed)
            self.assertEqual(result["certificate"]["truncated_excluded"], len(CUT))
            self.assertEqual(result["audit"]["N_plus"] + result["audit"]["N_minus"], len(pool) - len(CUT))

    def test_later_round_selection_never_takes_a_truncated_answer(self):
        pool = candidates(CUT)
        for arm in ("R", "S"):
            for seed in range(10):
                result = select_own_subset(pool, arm, 16, seed, ("b00", 2, arm))
                self.assertTrue(result["feasible"], result["certificate"])
                self.assertFalse(set(result["ids"]) & CUT, (arm, seed))
                self.assertEqual(result["certificate"]["truncated_excluded"], len(CUT))

    def test_the_flag_travels_from_the_pool_row(self):
        row = {"id": "x", "task_id": "t", "response": "r", "truncated": True}
        self.assertTrue(candidate_from_row(row, {}, {"correct": False, "error": "format"}, 3).truncated)
        # Mock fixtures carry no flag; on hf the entries refuse such a pool (test below).
        old = {"id": "x", "task_id": "t", "response": "r"}
        self.assertFalse(candidate_from_row(old, {}, {"correct": True, "error": None}, 3).truncated)


class TrainingGuardTests(unittest.TestCase):
    def test_training_rows_never_hold_a_truncated_answer(self):
        rows = [{"id": "x", "task_id": "t", "response": "r", "truncated": True}]
        with self.assertRaises(ValueError):
            one_step.load_branch_rows(["x"], {"t": {"prompt": "p"}}, rows, {"x": {"correct": True, "error": None}},
                                      {"x": 2})

    def test_an_hf_entry_refuses_a_pool_from_before_the_flag(self):
        """Such a pool's rows capped at max_new_tokens would pass as finished answers."""
        import hashlib
        from run_matched_experiment import read_block_pool
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b00.jsonl"
            for rows, refused in (([{"id": "x", "task_id": "t", "response": "r", "truncated": False}], False),
                                  ([{"id": "x", "task_id": "t", "response": "r"}], True)):
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                Path(str(path) + ".meta.json").write_text(json.dumps({
                    "hash": hashlib.sha256(path.read_bytes()).hexdigest(), "backend": "hf", "model": "m",
                    "revision": "r"}))
                if refused:
                    with self.assertRaises(SystemExit) as stop:
                        read_block_pool(path, "hf", {"t": {}}, base={"model": "m", "revision": "r"})
                    self.assertIn("do not record whether they were cut", str(stop.exception))
                else:
                    read_block_pool(path, "hf", {"t": {}}, base={"model": "m", "revision": "r"})


class GenerationFlagTests(unittest.TestCase):
    def test_a_row_without_a_stop_token_is_marked_truncated(self):
        import torch

        class Tokenizer:
            pad_token_id, eos_token_id = 0, 9

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(i) for i in ids if i != 9)

        class Model:
            generation_config = types.SimpleNamespace(eos_token_id=9)
            config = types.SimpleNamespace(use_cache=False)

            def eval(self):
                pass

            def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
                new = torch.full((input_ids.shape[0], max_new_tokens), 5)
                new[0, 2], new[0, 3:] = 9, 0   # the first sample stops; the second runs to the limit
                return torch.cat([input_ids, new], dim=1)

        model = backends.HFBackend.__new__(backends.HFBackend)
        model.torch, model.device, model.tokenizer, model.model = torch, "cpu", Tokenizer(), Model()
        model.config = {"generation": {"batch_size": 2, "max_new_tokens": 6, "max_sequence_length": 16,
                                       "temperature": 1.0, "top_p": 1.0, "top_k": None}}
        model.prompt_ids = lambda prompt: [1, 2]
        rows = model.generate([{"id": "a", "prompt": "p"}], 2, seed=0)
        self.assertEqual([(r["truncated"], r["completion_tokens"]) for r in rows], [(False, 3), (True, 6)])


class LegacyLoopTests(unittest.TestCase):
    def test_the_legacy_loop_and_evaluation_handle_truncated_answers(self):
        from evaluate import evaluate
        from rsi.experiment import run

        original = backends.MockBackend.generate

        def cut_every_third(self, tasks, count, seed):
            rows = original(self, tasks, count, seed)
            for row in rows[::3]:
                row["truncated"] = True
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate(root / "data", "graph", 9, 1, 18, 6, 8, 6, 6)
            cfg = copy.deepcopy(DEFAULTS)
            cfg.update(backend="mock", rounds=1)
            cfg["training"]["examples"] = 8
            with mock.patch.object(backends.MockBackend, "generate", cut_every_third):
                run(cfg, root / "data", root / "run")
                results = evaluate(root / "run", rounds=[1], full_rounds=[])
            done = read_json(root / "run" / "round_001" / "complete.json")
            attempt = root / "run" / done["attempt"]
            cut = {r["id"] for r in read_jsonl(attempt / "candidates.jsonl") if r.get("truncated")}
            self.assertEqual(done["truncated_candidates_excluded"], len(cut))
            self.assertTrue(cut)
            self.assertFalse({r["id"] for r in read_jsonl(attempt / "selection.jsonl")} & cut)
            self.assertFalse({r["id"] for r in read_jsonl(attempt / "retained.jsonl")} & cut)
        for result in results:
            self.assertEqual(result["truncated_answers"], -(-result["answers"] // 3))


class PoolSizeTests(unittest.TestCase):
    def config(self, root, generation):
        path = Path(root) / "config.json"
        path.write_text(json.dumps({"backend": "mock", "generation": generation}))
        return path

    def test_prompts_per_pool_is_a_positive_integer_or_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (0, -1, True, 1.5, "2"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_config(self.config(tmp, {"prompts_per_pool": value}))
            for value in (None, 3):
                self.assertEqual(load_config(self.config(tmp, {"prompts_per_pool": value}))["generation"]
                                 ["prompts_per_pool"], value)

    def test_a_pool_samples_the_first_prompts_per_pool_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            generate(data, "arithmetic", 11, rounds=1, per_round=5, dev=2, calibration=1, eval_id=1, eval_ood=1,
                     gradient_reference=2)
            first = [t["id"] for t in read_jsonl(data / "train_001.jsonl")][:3]
            out = Path(tmp) / "pool.jsonl"
            sample_candidates.main(["--config", str(self.config(tmp, {"candidates": 2, "prompts_per_pool": 3})),
                                    "--data", str(data), "--out", str(out)])
            self.assertEqual(sorted({r["task_id"] for r in read_jsonl(out)}), sorted(first))
            self.assertEqual(read_json(str(out) + ".meta.json")["split"]["prompts_sampled"], 3)
            with self.assertRaises(SystemExit) as refused:
                sample_candidates.main(["--config", str(self.config(tmp, {"candidates": 2, "prompts_per_pool": 6})),
                                        "--data", str(data), "--out", str(Path(tmp) / "big.jsonl")])
            self.assertIn("has only 5 prompts", str(refused.exception))

    def test_the_lora_alias_counts_as_stating_the_field(self):
        raw = json.loads((Path(__file__).resolve().parents[1] / "configs" / "matched_pool.json").read_text())
        raw["training"]["lora_target_modules"] = raw["training"].pop("target_modules")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alias.json"
            path.write_text(json.dumps(raw))
            require_explicit(path, "one_step")


if __name__ == "__main__":
    unittest.main()
