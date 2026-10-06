"""CPU fixtures for arithmetic matching and its CLI; not model-study results."""
import copy
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import rsi.arithmetic_intervention as entry
from rsi.common import DEFAULTS, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from rsi.matching import (ARITHMETIC_TARGET_SIGNATURES, TARGET_SIGNATURES, Candidate,
                          MatchingFailure, construct_matched_subsets, select_k_for_blocks)
from rsi.split_selection import dataset_binding, read_split
from rsi.tasks import judge

REPO = Path(__file__).resolve().parents[1]
PIN = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def candidates(n, target="ignore_parentheses", target_tokens=3, other_tokens=3):
    rows = []
    for i in range(n):
        tid = "t%03d" % i
        rows.extend([Candidate(tid + "-c", tid, "20", True, "correct", 3),
                     Candidate(tid + "-t", tid, "14", False, target, target_tokens),
                     Candidate(tid + "-o", tid, "21", False, "other", other_tokens)])
    return rows


class ArithmeticMatchingTests(unittest.TestCase):
    def test_arithmetic_matches_both_arms_and_records_target_hits(self):
        pool = candidates(64)
        by_id = {r.candidate_id: r for r in pool}
        result = construct_matched_subsets(pool, 64, seed=0, target_signatures=ARITHMETIC_TARGET_SIGNATURES)
        self.assertTrue(result["feasible"])
        audit = result["audit"]
        for name in ("K_equal", "same_task_ids", "same_correct_ids", "task_token_counts_within_tolerance",
                     "ratio_C_to_E", "final_TPR_equal", "final_FPR_equal"):
            self.assertTrue(audit[name], name)
        self.assertEqual((audit["C_R"], audit["E_R"], audit["C_S"], audit["E_S"]), (48, 16, 48, 16))
        self.assertEqual(audit["S_target_hits"], 16)
        self.assertEqual(audit["token_length_tolerance"], 0)
        self.assertTrue(all(by_id[cid].error == "ignore_parentheses" for cid in result["S"]
                            if not by_id[cid].correct))
        self.assertEqual(audit["final_TPR"], 48 / 64)
        self.assertEqual(audit["final_FPR"], 16 / 128)

    def test_random_arm_can_select_other_errors_and_keeps_coincidences(self):
        result = construct_matched_subsets(candidates(64), 64, seed=0,
                                          target_signatures=ARITHMETIC_TARGET_SIGNATURES)
        pairs = result["audit"]["error_signature_pairs"]
        self.assertTrue(any(p["R"] == "other" for p in pairs))
        self.assertTrue(any(p["same_candidate"] for p in pairs))

    def test_graph_default_is_unchanged_and_arithmetic_is_explicit(self):
        graph = candidates(64, target="nonshortest")
        self.assertEqual(construct_matched_subsets(graph, 64, seed=5),
                         construct_matched_subsets(graph, 64, seed=5, target_signatures=TARGET_SIGNATURES))
        self.assertFalse(construct_matched_subsets(candidates(64), 16)["feasible"])

    def test_all_blocks_use_the_largest_common_k(self):
        result = select_k_for_blocks({"b00": candidates(64), "b01": candidates(32)}, seed=3,
                                     target_signatures=ARITHMETIC_TARGET_SIGNATURES)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["K"], 32)
        self.assertEqual([a["K"] for a in result["attempts"]], [64, 32])

    def test_missing_target_or_equal_length_bucket_stops_at_16(self):
        for pool in (candidates(64, target="sign"), candidates(64, other_tokens=4)):
            result = select_k_for_blocks({"b00": pool}, target_signatures=ARITHMETIC_TARGET_SIGNATURES)
            self.assertFalse(result["feasible"])
            self.assertEqual([a["K"] for a in result["attempts"]], [64, 32, 16])

    def test_ignoring_parentheses_is_not_an_error_when_the_value_is_unchanged(self):
        verdict = judge({"task": "arithmetic", "expression": "(2 + 3)"}, "5")
        self.assertEqual(verdict, {"correct": True, "error": "correct"})

    def test_invalid_target_collections_are_refused(self):
        for targets in ((), ("",), ("ignore_parentheses", "ignore_parentheses"), "ignore_parentheses"):
            with self.subTest(targets=targets), self.assertRaises(MatchingFailure):
                select_k_for_blocks({"b00": candidates(16)}, target_signatures=targets)


def fixture(root, count=64, blocks=3, backend="mock", split="train_001"):
    """Hand-built candidate responses solely for CLI correctness tests.

    Independent expected answers for (2+3)*(4+i) are 20+5*i (correct),
    14+i (ignoring parentheses), and 21+5*i (another error).
    """
    data, pools = root / "data", root / "pools"
    files, counts = {}, {}
    for name, size, start in (("train_001", count, 0), ("dev", count, 100)):
        tasks = []
        for i in range(start, start + size):
            expression = "(2 + 3) * (4 + %d)" % i
            # Without parentheses: 2 + 3 * 4 + i = 14+i.
            tasks.append({"task": "arithmetic", "expression": expression,
                          "prompt": "Evaluate exactly: " + expression, "difficulty": str(i % 3),
                          "id": digest({"task": "arithmetic", "expression": expression}), "split": name})
        path = data / (name + ".jsonl")
        write_jsonl(path, tasks)
        files[path.name], counts[name] = file_hash(path), size
    manifest = {"version": 1, "task": "arithmetic", "seed": 2027, "rounds": 1,
                "counts": counts, "files": files, "unique_instances": sum(counts.values())}
    write_json(data / "manifest.json", manifest)
    task_rows, binding = read_split(data, manifest, split)
    config = copy.deepcopy(DEFAULTS)
    config.update(backend=backend, revision=PIN)
    config["generation"]["candidates"] = 3
    for b in range(blocks):
        rows = []
        for i, task in enumerate(task_rows, start=0 if split == "train_001" else 100):
            # All three are saved input fixtures, never generated by the CLI.
            for sample, response in enumerate((str(20 + 5 * i), str(14 + i), str(21 + 5 * i))):
                rows.append({"id": digest([task["id"], sample]), "task_id": task["id"], "sample": sample,
                             "response": response, "completion_tokens": 999, "prompt_tokens": 999,
                             "truncated": False})
        path = pools / ("b%02d.jsonl" % b)
        write_jsonl(path, rows)
        write_json(str(path) + ".meta.json", {
            "hash": file_hash(path), "backend": backend, "model": config["model"], "revision": PIN,
            "generation": config["generation"], "generation_hash": digest(config["generation"]),
            "config": config, "config_hash": digest(config), "split": binding,
            "data": dataset_binding(data, manifest),
            "identity": {"study_id": "arithmetic-test", "phase": "main" if split.startswith("train_") else "pilot"},
            "seed": {"cli": b}})
    return ["--data", str(data), "--pools", str(pools), "--split", split,
            "--study-id", "arithmetic-test", "--phase", "main" if split.startswith("train_") else "pilot",
            "--out", str(root / "out"), "--backend", backend]


class ArithmeticEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_entry(self, argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return entry.main(argv)

    def update_meta_hash(self, path):
        meta_path = Path(str(path) + ".meta.json")
        meta = read_json(meta_path)
        meta["hash"] = file_hash(path)
        write_json(meta_path, meta)

    def test_exports_original_responses_with_no_oracle_labels(self):
        argv = fixture(self.root)
        self.assertEqual(self.run_entry(argv), 0)
        out = self.root / "out"
        report = read_json(out / "intervention.json")
        matched = read_json(out / "matched_subsets.json")
        self.assertEqual(report["K"], 64)
        self.assertTrue(report["DEMO_ONLY"])
        self.assertTrue(matched["DEMO_ONLY"])
        self.assertFalse(report["trained"])
        self.assertEqual(report["target_signatures"], ["ignore_parentheses"])
        self.assertEqual(len(report["outputs"]), 6)
        for block in ("b00", "b01", "b02"):
            original = {r["id"]: r for r in read_jsonl(self.root / "pools" / (block + ".jsonl"))}
            for arm in ("R", "S"):
                rows = read_jsonl(out / block / arm / "training.jsonl")
                self.assertEqual(len(rows), 64)
                self.assertEqual(len({r["task_id"] for r in rows}), 64)
                self.assertEqual({r["id"] for r in rows}, set(matched["per_block"][block][arm]))
                for row in rows:
                    self.assertEqual(set(row), {"id", "task_id", "block_id", "prompt", "response", "weight", "tokens"})
                    self.assertEqual(row["response"], original[row["id"]]["response"])
                    self.assertEqual(row["weight"], 1.0)
                    self.assertEqual(row["tokens"], 2)  # whitespace response + EOS, not the fake 999

    def test_cli_smoke(self):
        argv = fixture(self.root, count=16, blocks=1)
        result = subprocess.run([sys.executable, "-m", "rsi.arithmetic_intervention"] + argv, cwd=REPO,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("K=16", result.stdout)
        self.assertIn("DEMO_ONLY=True", result.stdout)

    def test_hf_recounts_real_training_encoding_including_eos(self):
        argv = fixture(self.root, count=16, blocks=1, backend="hf")

        class Tokenizer:
            eos_token_id = 10
            chat_template = "test"

            def encode(self, response, add_special_tokens=False):
                self.assert_no_special = not add_special_tokens
                return [ord(c) for c in response]

        tokenizer = Tokenizer()
        encoder = SimpleNamespace(tokenizer=tokenizer, prompt_ids=lambda prompt: [7, 8])
        with mock.patch.object(entry, "load_encoder", return_value=(encoder, {"method": "stub test", "eos_token_id": 10})):
            self.assertEqual(self.run_entry(argv), 0)
        report = read_json(self.root / "out" / "intervention.json")
        self.assertFalse(report["DEMO_ONLY"])
        for arm in ("R", "S"):
            for row in read_jsonl(self.root / "out" / "b00" / arm / "training.jsonl"):
                self.assertEqual(row["tokens"], len(row["response"]) + 1)
        self.assertTrue(tokenizer.assert_no_special)

    def test_hf_tokenizer_loader_pins_revision_without_loading_weights(self):
        tokenizer = SimpleNamespace(eos_token_id=42, chat_template="test", apply_chat_template=mock.Mock(return_value=[1]))
        factory = mock.Mock(return_value=tokenizer)
        stub = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory))
        with mock.patch.dict(sys.modules, {"transformers": stub}):
            encoder, record = entry.load_encoder({"model": "Qwen/Qwen3-1.7B", "revision": PIN}, "hf")
        factory.assert_called_once_with("Qwen/Qwen3-1.7B", revision=PIN)
        self.assertEqual(encoder.prompt_ids("question"), [1])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "question"}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, date_string="26 Jul 2024")
        self.assertEqual(record["eos_token_id"], 42)

    def test_hf_tokenizer_without_eos_or_template_is_refused(self):
        for eos, template in ((None, "template"), (42, None)):
            tokenizer = SimpleNamespace(eos_token_id=eos, chat_template=template)
            stub = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
            with mock.patch.dict(sys.modules, {"transformers": stub}), self.assertRaises(ValueError):
                entry.load_encoder({"model": "Qwen/Qwen3-1.7B", "revision": PIN}, "hf")

    def test_hf_overlong_training_sequences_are_not_truncated(self):
        argv = fixture(self.root, count=16, blocks=1, backend="hf")
        path = self.root / "pools" / "b00.jsonl.meta.json"
        meta = read_json(path)
        meta["generation"]["max_sequence_length"] = 2
        meta["config"]["generation"] = copy.deepcopy(meta["generation"])
        meta["generation_hash"], meta["config_hash"] = digest(meta["generation"]), digest(meta["config"])
        write_json(path, meta)
        encoder, _ = entry.load_encoder(meta, "mock")
        with mock.patch.object(entry, "load_encoder", return_value=(encoder, {})), \
                self.assertRaisesRegex(ValueError, "refusing truncation"):
            self.run_entry(argv)
        self.assertFalse((self.root / "out").exists())

    def test_actual_mock_sampling_metadata_is_accepted_without_inventing_target_errors(self):
        import sample_candidates

        argv = fixture(self.root, count=16, blocks=1)
        path = self.root / "pools" / "b00.jsonl"
        # Use a new pool path: sample_candidates refuses to overwrite a pool.
        config = read_json(Path(str(path) + ".meta.json"))["config"]
        config_path = self.root / "config.json"
        write_json(config_path, config)
        new_pool_dir = self.root / "sampled"
        with redirect_stdout(io.StringIO()):
            sample_candidates.main(["--config", str(config_path), "--data", str(self.root / "data"),
                                    "--out", str(new_pool_dir / "b00.jsonl"), "--backend", "mock",
                                    "--split", "train_001", "--study-id", "arithmetic-test", "--phase", "main"])
        argv[argv.index("--pools") + 1] = str(new_pool_dir)
        self.assertEqual(self.run_entry(argv), 2)
        report = read_json(self.root / "out" / "intervention.json")
        self.assertEqual(report["original_error_counts"]["b00"].get("ignore_parentheses", 0), 0)
        self.assertTrue(report["DEMO_ONLY"])

    def test_mismatched_generation_settings_between_blocks_are_refused(self):
        argv = fixture(self.root, count=16, blocks=2)
        path = self.root / "pools" / "b01.jsonl.meta.json"
        meta = read_json(path)
        meta["generation"]["temperature"] = 0.8
        meta["config"]["generation"] = copy.deepcopy(meta["generation"])
        meta["generation_hash"], meta["config_hash"] = digest(meta["generation"]), digest(meta["config"])
        write_json(path, meta)
        with self.assertRaisesRegex(ValueError, "Blocks must share"):
            self.run_entry(argv)
        self.assertFalse((self.root / "out").exists())

    def test_missing_target_writes_certificate_but_no_training_rows(self):
        argv = fixture(self.root, count=16, blocks=1)
        path = self.root / "pools" / "b00.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            if row["sample"] == 1:
                row["response"] = "99999"
        write_jsonl(path, rows)
        self.update_meta_hash(path)
        self.assertEqual(self.run_entry(argv), 2)
        out = self.root / "out"
        self.assertFalse(read_json(out / "ladder.json")["feasible"])
        self.assertFalse((out / "matched_subsets.json").exists())
        self.assertEqual(list(out.glob("*/R/training.jsonl")), [])
        self.assertEqual([a["K"] for a in read_json(out / "ladder.json")["attempts"]], [64, 32, 16])

    def test_repeated_ids_in_different_blocks_do_not_share_counts(self):
        argv = fixture(self.root, count=16, blocks=2)
        path = self.root / "pools" / "b01.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            if row["sample"] == 2:
                row["response"] = "two words"  # target has 2 supervised mock tokens, this has 3
        write_jsonl(path, rows)
        self.update_meta_hash(path)
        self.assertEqual(self.run_entry(argv), 2)
        last = read_json(self.root / "out" / "ladder.json")["attempts"][-1]
        self.assertTrue(last["certificates"]["b00"]["condition_error_tasks"])
        self.assertFalse(last["certificates"]["b01"]["feasible"])

    def test_held_out_splits_and_phase_mismatch_never_write(self):
        argv = fixture(self.root, count=16, blocks=1)
        for split in ("calibration", "eval_id", "eval_ood", "gradient_reference", "dev"):
            bad = list(argv)
            bad[bad.index("--split") + 1] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.run_entry(bad)
            self.assertFalse((self.root / "out").exists())

    def test_development_pilot_is_explicit_and_keeps_its_split(self):
        argv = fixture(self.root, count=16, blocks=1, split="dev")
        self.assertEqual(self.run_entry(argv), 0)
        report = read_json(self.root / "out" / "intervention.json")
        self.assertEqual(report["split"]["name"], "dev")
        self.assertEqual(report["phase"], "pilot")

    def test_hash_mismatch_refuses_before_tokenizer_load(self):
        argv = fixture(self.root, count=16, blocks=1)
        path = self.root / "pools" / "b00.jsonl"
        with path.open("a") as stream:
            stream.write("\n")
        with mock.patch.object(entry, "load_encoder") as load, self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_entry(argv)
        load.assert_not_called()
        self.assertFalse((self.root / "out").exists())

    def test_mismatched_pool_identity_is_refused(self):
        argv = fixture(self.root, count=16, blocks=1)
        meta_path = self.root / "pools" / "b00.jsonl.meta.json"
        original = read_json(meta_path)
        for field, key, value in (("split", "name", "dev"), ("data", "manifest_sha256", "bad"),
                                  ("identity", "study_id", "another-study"), ("identity", "phase", "pilot")):
            meta = copy.deepcopy(original)
            meta[field][key] = value
            write_json(meta_path, meta)
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                self.run_entry(argv)
            self.assertFalse((self.root / "out").exists())

    def test_mock_pool_cannot_enter_hf_run(self):
        argv = fixture(self.root, count=16, blocks=1)
        argv[argv.index("--backend") + 1] = "hf"
        with self.assertRaisesRegex(ValueError, "backend mismatch"):
            self.run_entry(argv)

    def test_incomplete_duplicate_unknown_and_reference_answer_pools_are_refused(self):
        argv = fixture(self.root, count=16, blocks=1)
        path = self.root / "pools" / "b00.jsonl"
        original = read_jsonl(path)
        cases = [original[:-1], original + [original[0]], copy.deepcopy(original), copy.deepcopy(original)]
        cases[2][0]["task_id"] = "unknown"
        cases[3][0]["reference_answer"] = cases[3][0].pop("response")
        for rows in cases:
            write_jsonl(path, rows)
            self.update_meta_hash(path)
            with self.assertRaises(ValueError):
                self.run_entry(argv)
            self.assertFalse((self.root / "out").exists())

    def test_existing_results_are_preserved(self):
        argv = fixture(self.root, count=16, blocks=1)
        self.assertEqual(self.run_entry(argv), 0)
        path = self.root / "out" / "intervention.json"
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.run_entry(argv)
        self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
