"""run_matched_experiment.py on the hf backend, with the GPU work stubbed.

What is real here: the entry's preflight (base pin, shared adapter record, the
reference gradient h and what it is bound to), the joint matching, the rows each
arm receives, and the records written.  What is stubbed: the tokenizer (a word
tokenizer, so the supervised count is the fixture's word count + 1 EOS and
differs from `completion_tokens` by exactly the EOS), and `one_step._run_hf_arm`
(it records its call and writes the files a real arm writes).  The real arm runs
on the GPU in the end-to-end record, not here.
"""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import one_step
import run_matched_experiment
from generate_data import generate
from rsi.common import digest, file_hash, load_config, read_json, read_jsonl, verify_dataset, write_json, write_jsonl
from rsi.experiment import pin_config
from rsi.shared_adapter import lora_protocol
from test_run_matched_experiment import BLOCKS, block_rows

# An exported RSI_BASE_PIN must not leak into these tests (tests/pin_isolation.py).
from pin_isolation import isolate as setUpModule, restore as tearDownModule  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "matched_pool.json"
SEED = 0
STUDY = "md-hf-test-v01"
ARMS = ("R", "S", "null")
PARAMETER_HASH = "a" * 64
# The base the fixture pools were "sampled" from: the config's, which the hf entries require of a pool.
BASE = {key: read_json(CONFIG)[key] for key in ("model", "revision")}


class WordTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1] * len(text.split())


def pool_meta(path, **base_changes):
    """A fixture pool's meta, sampled from the config's base unless `base_changes` says otherwise."""
    return dict({"fingerprint": "fixture", "hash": file_hash(path), "backend": "hf"}, **dict(BASE, **base_changes))


def write_hf_pools(directory, tasks, **base_changes):
    directory = Path(directory)
    for name, spec in BLOCKS.items():
        path = directory / (name + ".jsonl")
        write_jsonl(path, block_rows(tasks, **spec))
        write_json(str(path) + ".meta.json", pool_meta(path, **base_changes))
    return directory


class HFFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.data = cls.root / "data"
        generate(cls.data, "graph", 7, 1, 40, 4, 4, 8, 8)
        cls.tasks = read_jsonl(cls.data / "train_001.jsonl")
        cls.pools = write_hf_pools(cls.root / "pools", cls.tasks)
        cls.resolved = pin_config(load_config(CONFIG, SEED, "hf"))
        cls.adapter, cls.reference = cls.write_inputs(cls.root / "inputs")
        cls.calls = []
        cls.out = cls.root / "out"
        cls.code, cls.stdout, cls.message = cls.run_hf(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def write_inputs(cls, directory, **changes):
        """A shared adapter record and a reference gradient that belong together; `changes` breaks one."""
        adapter, reference = Path(directory) / "shared_adapter", Path(directory) / "reference"
        adapter.mkdir(parents=True)
        reference.mkdir(parents=True)
        write_json(adapter / "shared_adapter.json", {
            "parameter_hash": PARAMETER_HASH, "base_model": cls.resolved["model"],
            "base_revision": changes.get("adapter_revision", cls.resolved["revision"]),
            "protocol": lora_protocol(cls.resolved["training"])})
        torch.save({"lora": torch.zeros(2)}, reference / "h.pt")
        write_json(reference / "h.json", {
            "h": {"file": "h.pt", "sha256": file_hash(reference / "h.pt")},
            "bindings": {"base_model": cls.resolved["model"], "base_revision": cls.resolved["revision"],
                         "shared_adapter_parameter_hash": changes.get("h_adapter", PARAMETER_HASH),
                         "config_hash": changes.get("h_config", digest(cls.resolved)),
                         "data": {"dataset_hash": changes.get("h_data", digest(verify_dataset(cls.data)))}}})
        if changes.get("tamper_h"):
            torch.save({"lora": torch.ones(2)}, reference / "h.pt")
        return adapter, reference

    @classmethod
    def fake_arm(cls, branch, rows, config, adapter_dir, output, seed, recorder_kwargs):
        cls.calls.append({"branch": branch, "adapter_dir": adapter_dir, "recorder_kwargs": recorder_kwargs,
                          "tokens": [r["tokens"] for r in rows]})
        write_json(Path(output) / "diagnostics.json", {"branch": branch, "stub": True})
        if branch != "null":
            (Path(output) / "adapter").mkdir()
        return {"trained": branch != "null", "steps": 0 if branch == "null" else 1, "mean_loss": 0.5,
                "parameter_hash": "b" * 64, "determinism": {"deterministic_algorithms": True}}

    @classmethod
    def run_hf(cls, out, arm=None, **overrides):
        args = argparse.Namespace(config=str(CONFIG), data=str(cls.data), shared_pool=str(cls.pools), out=str(out),
                                  seed=SEED, study_id=STUDY, phase="pilot", backend="hf",
                                  shared_adapter=str(cls.adapter), reference_gradient=str(cls.reference))
        for key, value in overrides.items():
            setattr(args, key, value)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                mock.patch.object(run_matched_experiment, "load_tokenizer", return_value=WordTokenizer()), \
                mock.patch.object(one_step, "_run_hf_arm", arm or cls.fake_arm):
            try:
                code = run_matched_experiment.main(args)
            except SystemExit as exit:
                code = exit.code
        message = stderr.getvalue()
        if isinstance(code, str):
            code, message = 1, code
        return code, stdout.getvalue(), message

    def scratch(self):
        return Path(tempfile.mkdtemp(dir=self.root))


class HFRunTests(HFFixture):
    def test_every_arm_runs_from_the_shared_adapter_with_h(self):
        self.assertEqual(self.code, 0, self.message)
        experiment = read_json(self.out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["backend"], experiment["DEMO_ONLY"]),
                         ("complete", "hf", False))
        self.assertEqual([c["branch"] for c in self.calls[:9]], list(ARMS) * 3)
        for call in self.calls[:9]:
            self.assertEqual(Path(call["adapter_dir"]), self.adapter)
            self.assertEqual(call["recorder_kwargs"], {"reference_gradient_path": str(self.reference / "h.pt")})

    def test_matching_and_training_use_the_supervised_token_count(self):
        for block in BLOCKS:
            pool = {r["id"]: r for r in read_jsonl(self.pools / (block + ".jsonl"))}
            for arm in ARMS:
                rows = read_jsonl(self.out / block / arm / "round_001" / "training.jsonl")
                for row in rows:
                    self.assertEqual(row["tokens"], pool[row["id"]]["completion_tokens"] + 1, (block, arm))
            run = read_json(self.out / block / "R" / "run.json")
            self.assertEqual(run["bindings"]["token_counts"]["source"], "supervised_token_count")
            self.assertEqual(run["bindings"]["tokenizer"], {"name": self.resolved["model"],
                                                            "revision": self.resolved["revision"]})

    def test_each_arm_records_its_diagnostics(self):
        for block in BLOCKS:
            lines = read_jsonl(self.out / block / "diagnostics" / "gradients.jsonl")
            self.assertEqual([l["branch"] for l in lines], list(ARMS))
            for line, arm in zip(lines, ARMS):
                self.assertEqual(line["status"], "recorded")
                self.assertEqual(line["diagnostics"], "%s/%s/round_001/diagnostics.json" % (block, arm))
                self.assertTrue((self.out / line["diagnostics"]).is_file())
                complete = read_json(self.out / block / arm / "round_001" / "complete.json")
                self.assertEqual(complete["adapter"], None if arm == "null" else "round_001/adapter")

    def test_the_inputs_are_bound(self):
        bindings = read_json(self.out / "b00" / "S" / "run.json")["bindings"]
        self.assertEqual(bindings["shared_adapter"]["status"], "used")
        self.assertEqual(bindings["shared_adapter"]["parameter_hash"], PARAMETER_HASH)
        self.assertEqual(bindings["shared_adapter"]["record_sha256"], file_hash(self.adapter / "shared_adapter.json"))
        self.assertEqual(bindings["reference_gradient"]["status"], "used")
        self.assertEqual(bindings["reference_gradient"]["h_sha256"], file_hash(self.reference / "h.pt"))
        self.assertEqual(bindings["reference_gradient"]["h_json_sha256"], file_hash(self.reference / "h.json"))
        self.assertEqual(bindings["base_pin"]["resolved_revision"], self.resolved["revision"])
        self.assertTrue(bindings["model"]["revision_resolved"])


class HFRefusalTests(HFFixture):
    def assertRefused(self, fragment, **overrides):
        out = self.scratch() / "out"
        code, _, message = self.run_hf(out, **overrides)
        self.assertEqual(code, 1, message)
        self.assertIn(fragment, message)
        self.assertIn("Nothing was written", message)
        self.assertFalse(out.exists())

    def broken(self, **changes):
        adapter, reference = self.write_inputs(self.scratch(), **changes)
        return {"shared_adapter": str(adapter), "reference_gradient": str(reference)}

    def test_hf_needs_both_inputs(self):
        self.assertRefused("--backend hf needs --shared-adapter", shared_adapter=None)
        self.assertRefused("--backend hf needs --reference-gradient", reference_gradient=None)

    def test_mock_takes_neither_input(self):
        self.assertRefused("the mock backend loads no model", backend="mock")

    def test_an_adapter_on_another_base_is_refused(self):
        self.assertRefused("shared adapter", **self.broken(adapter_revision="0" * 40))

    def test_a_pool_sampled_from_another_base_is_refused(self):
        llama = write_hf_pools(self.scratch() / "pools", self.tasks, model="meta-llama/Llama-3.2-1B-Instruct")
        self.assertRefused("was sampled from model", shared_pool=str(llama))
        other_commit = write_hf_pools(self.scratch() / "pools", self.tasks, revision="0" * 40)
        self.assertRefused("was sampled from revision", shared_pool=str(other_commit))
        mixed = write_hf_pools(self.scratch() / "pools", self.tasks)  # one block of three from another base
        b01 = mixed / "b01.jsonl"
        write_json(str(b01) + ".meta.json", pool_meta(b01, model="meta-llama/Llama-3.2-3B-Instruct"))
        self.assertRefused("b01.jsonl was sampled from model", shared_pool=str(mixed))

    def test_a_pool_drawn_for_other_data_or_settings_is_refused(self):
        # Task ids leave the prompt out, so a pool from before a prompt change matches train_001's ids.
        hotter = dict(self.resolved["generation"], temperature=0.7)
        for change, fragment in (({"data": {"dataset_hash": "f" * 64}}, "data (a prompt or the questions"),
                                 ({"split": {"name": "dev"}}, "split (dev, not train_001)"),
                                 ({"generation": hotter}, "sampling settings")):
            pools = write_hf_pools(self.scratch() / "pools", self.tasks, **change)
            self.assertRefused(fragment, shared_pool=str(pools))
        # The same fields recorded as this run's are accepted.
        pools = write_hf_pools(self.scratch() / "pools", self.tasks,
                               data={"dataset_hash": digest(verify_dataset(self.data))},
                               split={"name": "train_001"}, generation=self.resolved["generation"])
        code, _, message = self.run_hf(self.scratch() / "out", shared_pool=str(pools))
        self.assertEqual(code, 0, message)

    def test_an_h_that_does_not_belong_to_this_run_is_refused(self):
        self.assertRefused("shared_adapter_parameter_hash", **self.broken(h_adapter="c" * 64))
        self.assertRefused("config_hash", **self.broken(h_config="d" * 64))
        self.assertRefused("dataset_hash", **self.broken(h_data="e" * 64))

    def test_an_h_changed_after_its_record_is_refused(self):
        self.assertRefused("changed after h.json was written", **self.broken(tamper_h=True))

    def test_an_arm_without_diagnostics_fails_the_run(self):
        def silent(branch, rows, config, adapter_dir, output, seed, recorder_kwargs):
            return {"trained": True, "steps": 1, "mean_loss": 0.5, "parameter_hash": "b" * 64}

        out = self.scratch() / "out"
        with self.assertRaises(RuntimeError):
            self.run_hf(out, arm=silent)
        experiment = read_json(out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["completed"]), ("failed", []))
        self.assertIn("diagnostics.json", experiment["failure_reason"])


if __name__ == "__main__":
    unittest.main()


class HFArithmeticTests(HFFixture):
    """The one-step run on an arithmetic dataset: S's errors are ignore_parentheses."""

    @classmethod
    def setUpClass(cls):
        from test_run_iterative_experiment import arithmetic_pool
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.data = cls.root / "data"
        generate(cls.data, "arithmetic", 7, 1, 40, 4, 4, 8, 8)
        cls.tasks = read_jsonl(cls.data / "train_001.jsonl")
        cls.pools = cls.root / "pools"
        for b, name in enumerate(BLOCKS):
            path = cls.pools / (name + ".jsonl")
            write_jsonl(path, arithmetic_pool(cls.tasks, 4, b, "round-1"))
            write_json(str(path) + ".meta.json", pool_meta(path))
        cls.resolved = pin_config(load_config(CONFIG, SEED, "hf"))
        cls.adapter, cls.reference = cls.write_inputs(cls.root / "inputs")
        cls.calls = []
        cls.out = cls.root / "out"
        cls.code, cls.stdout, cls.message = cls.run_hf(cls.out)

    def test_s_errors_are_the_arithmetic_target(self):
        from rsi.tasks import judge
        self.assertEqual(self.code, 0, self.message)
        by_task = {t["id"]: t for t in self.tasks}
        for block in BLOCKS:
            rows = read_jsonl(self.out / block / "S" / "round_001" / "training.jsonl")
            errors = {judge(by_task[r["task_id"]], r["response"])["error"] for r in rows} - {"correct"}
            self.assertEqual(errors, {"ignore_parentheses"}, block)
        self.assertEqual(read_json(self.out / "experiment.json")["target_signatures"], ["ignore_parentheses"])
