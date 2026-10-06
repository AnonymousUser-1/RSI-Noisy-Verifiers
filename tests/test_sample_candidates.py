"""sample_candidates.py (P): the split is named, hf's phase fixes it, the meta records what was read.

science-aux 5d124fd6: on hf a pilot reads dev only and a main run train_NNN
only, refused before anything is resolved, loaded or written; calibration, the
eval splits and gradient_reference never enter a pool; mock keeps its existing
calls; and the pool meta records the data actually read, the configuration
actually in effect and the argv -- never values copied from a plan.

Where it matters the oracles do not come from the code under test: file hashes
are recomputed with hashlib from the bytes on disk, the generation seed from its
written rule with hashlib, the rows a pool was drawn from by reading the split
files with the json module, and the decoding overrides by running the real
HFBackend.generate against a stub model.  Two components are shared and named
where used: rsi.common.source_hash (the fingerprint's own input) and the base
pin file (read by the real check_base_pin).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import sample_candidates
from generate_data import generate
from rsi.common import load_config, source_hash
from rsi.experiment import sample_pool
from rsi.split_selection import check_selection, dataset_binding

# An exported RSI_BASE_PIN must not leak into these tests (tests/pin_isolation.py).
from pin_isolation import isolate as setUpModule, restore as tearDownModule  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
PIN = REPO / "configs" / "pins" / "base_pin.json"
ANCHOR = json.loads(PIN.read_text(encoding="utf-8"))["revision"]
SYNTHETIC = "0123456789abcdef0123456789abcdef01234567"
MOCK_CONFIG = {"backend": "mock", "generation": {"candidates": 2, "batch_size": 4}}
# Every field the hf pool stage reads is stated (rsi.common.require_explicit).
HF_CONFIG = {"backend": "hf", "model": "Qwen/Qwen3-1.7B", "revision": ANCHOR, "dtype": "bfloat16", "device": "cuda",
             "generation": {"candidates": 2, "prompts_per_pool": None, "batch_size": 4, "max_new_tokens": 2048,
                            "max_sequence_length": 4096, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_stop": None}}
NEVER_SAMPLED = ("calibration", "eval_id", "eval_ood", "gradient_reference")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    """rsi.common.digest's rule, written out: sha256 of the canonical JSON."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ids(path):
    with open(path, "rb") as stream:
        return [json.loads(line)["id"] for line in stream if line.strip()]


class StubGenerationConfig:
    eos_token_id = [151645, 151643]

    def to_dict(self):
        return {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "eos_token_id": [151645, 151643], "repetition_penalty": 1.0}


class StubHF:
    """What sample_pool reads off a loaded HFBackend, and nothing else."""

    def __init__(self, config):
        self.config = config
        self.model = types.SimpleNamespace(generation_config=StubGenerationConfig())
        self.tokenizer = types.SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
        self.seeds = []

    def generate(self, tasks, candidates, seed):
        self.seeds.append(seed)
        return [{"id": "%s-%d" % (t["id"], i), "task_id": t["id"], "sample": i, "response": "0",
                 "completion_tokens": 1, "prompt_tokens": 1} for t in tasks for i in range(candidates)]

    def close(self):
        pass


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._data = tempfile.TemporaryDirectory()
        cls.data = Path(cls._data.name) / "data"
        generate(cls.data, "arithmetic", 11, rounds=2, per_round=3, dev=2, calibration=1, eval_id=1,
                 eval_ood=1, gradient_reference=2)

    @classmethod
    def tearDownClass(cls):
        cls._data.cleanup()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def config(self, values, name="config.json"):
        path = self.tmp / name
        path.write_text(json.dumps(values), encoding="utf-8")
        return path

    def copy_data(self):
        return Path(shutil.copytree(self.data, self.tmp / "data"))

    def refusal(self, argv):
        """The SystemExit message main(argv) stops with; any other outcome fails the test."""
        try:
            sample_candidates.main([str(a) for a in argv])
        except SystemExit as exc:
            return str(exc)
        except Exception as exc:  # the defect surfaced as a crash, not as the refusal
            self.fail("stopped with %s: %s, not with the refusal" % (type(exc).__name__, exc))
        self.fail("main(%s) was not refused" % argv)

    def meta(self, out):
        return json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))


class SelectionRuleTests(unittest.TestCase):
    """rsi.split_selection.check_selection: the one rule P and the runner apply."""

    def test_on_hf_a_pilot_reads_dev_and_a_main_run_reads_train(self):
        for split, phase in (("dev", "pilot"), ("train_001", "main"), ("train_008", "main")):
            selected = check_selection(split, "s1", phase, hf=True)
            self.assertEqual(selected["split"], {"name": split, "source": "cli"})
            self.assertEqual(selected["identity"], {"study_id": "s1", "phase": phase, "missing": {}})

    def test_on_hf_a_phase_that_does_not_fit_the_split_is_refused(self):
        for split, phase, words in (("train_001", "pilot", "pilot reads dev only"),
                                    ("dev", "main", "main reads train_NNN only")):
            with self.assertRaises(ValueError) as caught:
                check_selection(split, "s1", phase, hf=True)
            self.assertIn(words, str(caught.exception))

    def test_on_hf_split_study_and_phase_are_each_required(self):
        full = {"split": "dev", "study_id": "s1", "phase": "pilot"}
        for key, flag in (("split", "--split"), ("study_id", "--study-id"), ("phase", "--phase")):
            values = dict(full, **{key: None})
            with self.assertRaises(ValueError) as caught:
                check_selection(values["split"], values["study_id"], values["phase"], hf=True)
            self.assertIn("%s is required on hf" % flag, str(caught.exception))

    def test_no_backend_samples_calibration_eval_reference_or_an_unlisted_name(self):
        odd = ("train_1", "train_0001", "TRAIN_001", "train_001.jsonl", "dev2", "../dev", "")
        for hf in (True, False):
            for split in NEVER_SAMPLED + odd:
                with self.assertRaises(ValueError, msg="%r on hf=%s" % (split, hf)) as caught:
                    check_selection(split, "s1", "pilot" if split == "dev" else "main", hf=hf)
                self.assertIn("cannot be sampled", str(caught.exception), "%r on hf=%s" % (split, hf))
                if split in NEVER_SAMPLED:
                    self.assertIn("never enter a pool", str(caught.exception), split)

    def test_a_phase_other_than_pilot_or_main_is_refused_without_argparse(self):
        """argparse's choices guard the CLI; a caller of the rule (the runner) gets the same answer."""
        for phase in ("formal", "Pilot", "", "dev"):
            for hf in (True, False):
                with self.assertRaises(ValueError, msg="%r on hf=%s" % (phase, hf)):
                    check_selection("dev", "s1", phase, hf=hf)

    def test_a_study_id_that_is_empty_or_padded_is_refused(self):
        for study in ("", " ", " s1", "s1 "):
            with self.assertRaises(ValueError):
                check_selection("dev", study, "pilot", hf=True)
            with self.assertRaises(ValueError):
                check_selection("dev", study, None, hf=False)

    def test_mock_without_a_split_draws_from_train_001_and_says_so(self):
        selected = check_selection(None, None, None, hf=False)
        self.assertEqual(selected["split"], {"name": "train_001", "source": "mock default"})
        self.assertEqual(sorted(selected["identity"]["missing"]), ["phase", "study_id"])
        self.assertIsNone(selected["identity"]["study_id"])

    def test_mock_does_not_apply_the_hf_phase_interlock(self):
        """Ruled for hf (5d124fd6); mock's existing calls keep working."""
        selected = check_selection("train_001", "s1", "pilot", hf=False)
        self.assertEqual(selected["split"], {"name": "train_001", "source": "cli"})


class RefusalTests(Fixture):
    """The entry refuses before it resolves, reads or loads anything, and writes nothing."""

    def test_an_hf_phase_split_mismatch_stops_before_the_pin_the_data_and_the_model(self):
        config = self.config(HF_CONFIG)
        for split, phase, words in (("train_001", "pilot", "pilot reads dev only"),
                                    ("dev", "main", "main reads train_NNN only")):
            out = self.tmp / ("pool-%s.jsonl" % phase)
            with mock.patch.object(sample_candidates, "pin_config") as pin, \
                    mock.patch.object(sample_candidates, "check_base_pin") as check, \
                    mock.patch.object(sample_candidates, "verify_dataset") as verify, \
                    mock.patch("rsi.experiment.backend") as load:
                message = self.refusal(["--config", config, "--data", self.data, "--out", out,
                                        "--split", split, "--study-id", "s1", "--phase", phase])
            self.assertIn(words, message)
            self.assertIn("Nothing was written", message)
            self.assertFalse(pin.called or check.called or verify.called or load.called,
                             "the mismatch was found only after the pin, data or model was touched")
            self.assertFalse(out.exists())

    def test_an_hf_run_without_split_study_or_phase_stops_before_the_pin(self):
        config = self.config(HF_CONFIG)
        full = {"--split": "dev", "--study-id": "s1", "--phase": "pilot"}
        for flag in full:
            argv = ["--config", config, "--data", self.data, "--out", self.tmp / "pool.jsonl"]
            for other, value in full.items():
                if other != flag:
                    argv += [other, value]
            with mock.patch.object(sample_candidates, "pin_config") as pin:
                message = self.refusal(argv)
            self.assertIn("%s is required on hf" % flag, message)
            self.assertFalse(pin.called, "%s: refused only after the pin was resolved" % flag)

    def test_a_split_the_manifest_does_not_list_is_refused(self):
        """train_009.jsonl is on disk and well named, but no manifest hash vouches for it."""
        data = self.copy_data()
        shutil.copyfile(data / "train_001.jsonl", data / "train_009.jsonl")
        out = self.tmp / "pool.jsonl"
        message = self.refusal(["--config", self.config(MOCK_CONFIG), "--data", data, "--out", out,
                                "--split", "train_009"])
        self.assertIn("not listed in the manifest", message)
        self.assertFalse(out.exists())

    def test_a_split_file_that_changes_after_verification_is_refused(self):
        """The rows parsed are the bytes hashed; a file changed after verify_dataset is not sampled."""
        data = self.copy_data()
        real = sample_candidates.verify_dataset

        def verify_then_change(root):
            manifest = real(root)
            with open(Path(root) / "dev.jsonl", "ab") as stream:
                stream.write(open(Path(root) / "train_001.jsonl", "rb").read())
            return manifest

        out = self.tmp / "pool.jsonl"
        with mock.patch.object(sample_candidates, "verify_dataset", verify_then_change):
            message = self.refusal(["--config", self.config(MOCK_CONFIG), "--data", data, "--out", out,
                                    "--split", "dev"])
        self.assertIn("changed after the manifest was verified", message)
        self.assertFalse(out.exists())

    def test_a_wrong_manifest_is_refused(self):
        """Another dataset's manifest beside these files: verify_dataset refuses, nothing is drawn."""
        data = self.copy_data()
        other = self.tmp / "other"
        generate(other, "arithmetic", 12, rounds=2, per_round=3, dev=2, calibration=1, eval_id=1,
                 eval_ood=1, gradient_reference=2)
        shutil.copyfile(other / "manifest.json", data / "manifest.json")
        out = self.tmp / "pool.jsonl"
        with self.assertRaises(ValueError) as caught:
            sample_candidates.main(["--config", str(self.config(MOCK_CONFIG)), "--data", str(data),
                                    "--out", str(out), "--split", "dev"])
        self.assertIn("Dataset changed", str(caught.exception))
        self.assertFalse(out.exists())

    def test_a_manifest_that_changes_after_verification_is_refused(self):
        manifest = json.loads((self.data / "manifest.json").read_text())
        changed = dict(manifest, seed=manifest["seed"] + 1)
        with self.assertRaises(ValueError) as caught:
            dataset_binding(self.data, changed)
        self.assertIn("changed after it was verified", str(caught.exception))

    def test_inputs_may_not_replace_the_meta_fields_and_are_refused_before_the_load(self):
        config = load_config(self.config(MOCK_CONFIG))
        out = self.tmp / "pool.jsonl"
        with mock.patch("rsi.experiment.backend") as load:
            with self.assertRaises(ValueError) as caught:
                sample_pool(config, [], out, 0, inputs={"seed": {"cli": 99}})
        self.assertIn("seed", str(caught.exception))
        self.assertFalse(load.called)
        self.assertFalse(out.exists())

    def test_a_value_the_meta_cannot_hold_is_refused_before_the_load(self):
        """Not after the pool is drawn: a NaN would otherwise fail in write_json, past the generation."""
        config = load_config(self.config(MOCK_CONFIG))
        out = self.tmp / "pool.jsonl"
        with mock.patch("rsi.experiment.backend") as load:
            with self.assertRaises(ValueError):
                sample_pool(config, [], out, 0, inputs={"note": float("nan")})
        self.assertFalse(load.called, "the model was loaded for a meta that could not be written")
        self.assertFalse(out.exists())


class MetaTests(Fixture):
    """The meta records the rows actually read and the configuration actually in effect."""

    def test_the_mock_call_without_new_flags_still_draws_from_train_001_and_says_so(self):
        out = self.tmp / "pool.jsonl"
        sample_candidates.main(["--config", str(self.config(MOCK_CONFIG)), "--data", str(self.data),
                                "--out", str(out)])
        meta = self.meta(out)
        train = self.data / "train_001.jsonl"
        self.assertEqual(meta["split"], {"name": "train_001", "file": "train_001.jsonl", "sha256": sha256(train),
                                         "rows": len(ids(train)), "prompts_sampled": len(ids(train)),
                                         "source": "mock default"})
        self.assertEqual(sorted({r["task_id"] for r in map(json.loads, out.read_text().splitlines())}),
                         sorted(ids(train)))
        self.assertIsNone(meta["identity"]["study_id"])
        self.assertEqual(sorted(meta["identity"]["missing"]), ["phase", "study_id"])
        self.assertEqual(meta["base_pin"]["status"], "not read")

    def test_the_pool_is_drawn_from_the_named_split_and_from_no_other(self):
        out = self.tmp / "pool.jsonl"
        sample_candidates.main(["--config", str(self.config(MOCK_CONFIG)), "--data", str(self.data),
                                "--out", str(out), "--split", "dev", "--study-id", "s1", "--phase", "pilot"])
        drawn = {json.loads(line)["task_id"] for line in out.read_text().splitlines()}
        self.assertEqual(drawn, set(ids(self.data / "dev.jsonl")))
        for other in ("train_001", "train_002") + NEVER_SAMPLED:
            self.assertFalse(drawn & set(ids(self.data / (other + ".jsonl"))), other)

    def test_the_meta_records_the_split_file_and_the_manifest_actually_read(self):
        out = self.tmp / "pool.jsonl"
        sample_candidates.main(["--config", str(self.config(MOCK_CONFIG)), "--data", str(self.data),
                                "--out", str(out), "--split", "train_002"])
        meta = self.meta(out)
        split = self.data / "train_002.jsonl"
        self.assertEqual(meta["split"], {"name": "train_002", "file": "train_002.jsonl", "sha256": sha256(split),
                                         "rows": len(ids(split)), "prompts_sampled": len(ids(split)),
                                         "source": "cli"})
        manifest = json.loads((self.data / "manifest.json").read_text())
        self.assertEqual(meta["data"]["manifest_sha256"], sha256(self.data / "manifest.json"))
        self.assertEqual(meta["data"]["dataset_hash"], canonical_sha256(manifest))
        self.assertEqual(meta["data"]["root"], str(self.data.resolve()))
        self.assertEqual((meta["data"]["task"], meta["data"]["data_seed"]), ("arithmetic", 11))

    def test_the_meta_records_the_seed_generate_actually_received(self):
        """Written rule seed_for(cli, 1, 'generation'), recomputed here with hashlib."""
        stub = {}

        def load(config):
            stub["backend"] = StubHF(config)
            return stub["backend"]

        out = self.tmp / "pool.jsonl"
        with mock.patch("rsi.experiment.backend", load):
            sample_candidates.main(["--config", str(self.config(MOCK_CONFIG)), "--data", str(self.data),
                                    "--out", str(out), "--seed", "2"])
        expected = int(hashlib.sha256(b'[2,1,"generation"]').hexdigest()[:8], 16)
        meta = self.meta(out)
        self.assertEqual(stub["backend"].seeds, [expected])
        self.assertEqual(meta["seed"], {"cli": 2, "generation": expected,
                                        "rule": "seed_for(cli_seed, 1, 'generation')"})

    def test_an_hf_meta_records_the_configuration_in_effect_the_pin_and_the_decoding(self):
        """The hf path end to end with the model stubbed: everything before the load is real but
        the alias resolution, which a seam replaces as tests/test_base_pin.py does."""
        loaded = {}

        def load(config):
            loaded["config"] = config
            loaded["backend"] = StubHF(config)
            return loaded["backend"]

        config = self.config(dict(HF_CONFIG, revision="main"))
        out = self.tmp / "b00.jsonl"
        # The transformers module a real load imports: its version is what the meta must name.
        transformers = types.SimpleNamespace(__version__="0.0.0-stub")
        argv = ["--config", str(config), "--data", str(self.data), "--out", str(out), "--seed", "1",
                "--backend", "hf", "--split", "train_002", "--study-id", "md-test", "--phase", "main"]
        with mock.patch.object(sample_candidates, "pin_config", lambda c: dict(c, revision=ANCHOR)), \
                mock.patch("rsi.experiment.backend", load), mock.patch.dict(sys.modules, transformers=transformers):
            sample_candidates.main(argv)
        meta = self.meta(out)
        self.assertEqual(meta["requested_config"]["revision"], "main")
        self.assertEqual(meta["config"]["revision"], ANCHOR, "the meta must hold the resolved configuration")
        self.assertEqual(meta["config"], loaded["config"], "the meta must hold the configuration the model got")
        self.assertEqual(meta["config_hash"], canonical_sha256(meta["config"]))
        self.assertEqual(meta["revision"], ANCHOR)
        binding = meta["base_pin"]
        self.assertEqual({k: binding.get(k) for k in ("pin_sha256", "model", "pin_revision", "resolved_revision",
                                                      "status")},
                         {"pin_sha256": sha256(PIN), "model": "Qwen/Qwen3-1.7B", "pin_revision": ANCHOR,
                          "resolved_revision": ANCHOR, "status": "checked"})
        self.assertEqual(os.path.normcase(binding.get("pin_path", "")), os.path.normcase(str(PIN)))
        self.assertEqual(meta["generation"], meta["config"]["generation"])
        self.assertEqual(meta["generation_hash"], canonical_sha256(meta["config"]["generation"]))
        self.assertEqual(meta["identity"], {"study_id": "md-test", "phase": "main", "missing": {}})
        self.assertEqual((meta["split"]["name"], meta["split"]["source"]), ("train_002", "cli"))
        self.assertEqual(meta["config_file"], {"path": str(config.resolve()), "sha256": sha256(config)})
        self.assertEqual(meta["argv"]["args"], argv)
        self.assertEqual(meta["argv"]["parsed"]["split"], "train_002")
        decoding = meta["decoding"]
        self.assertTrue(decoding["composition"].startswith("defaults + overrides"))
        self.assertEqual(decoding["model_defaults"], StubGenerationConfig().to_dict())
        self.assertEqual(decoding["overrides"], {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                                 "max_new_tokens": 2048, "pad_token_id": 151643})
        self.assertEqual(decoding["stop_ids"], {"generation_config_eos_token_id": [151645, 151643],
                                                "tokenizer_eos_token_id": 151645})
        self.assertEqual(decoding["batch_size"], 4)
        self.assertEqual(decoding["transformers_version"], "0.0.0-stub")

    def test_the_recorded_overrides_are_what_hfbackend_generate_passes(self):
        """The real HFBackend.generate against a stub model: the two copies of the kwargs agree."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from rsi.backends import HFBackend
        from rsi.experiment import decoding_record

        config = load_config(self.config(dict(HF_CONFIG, generation={
            "candidates": 2, "batch_size": 4, "temperature": 0.55, "top_p": 0.85, "top_k": 7, "max_new_tokens": 5})))
        passed = {}

        class Model:
            generation_config = StubGenerationConfig()
            config = types.SimpleNamespace(use_cache=False)

            def eval(self):
                return self

            def generate(self, input_ids, attention_mask, **kwargs):
                passed.update(kwargs)
                return torch.cat([input_ids, torch.full((input_ids.shape[0], 2), 151645, dtype=torch.long)], 1)

        class Tokenizer:
            chat_template, eos_token_id, pad_token_id = "stub", 151645, 151643

            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]

            def decode(self, tokens, skip_special_tokens):
                return "0"

        hf = HFBackend.__new__(HFBackend)
        hf.torch, hf.config, hf.device, hf.model, hf.tokenizer = torch, config, "cpu", Model(), Tokenizer()
        hf.generate([{"id": "t1", "prompt": "p"}], 2, 0)
        self.assertEqual(decoding_record(hf, config)["overrides"], passed)

    def test_the_old_fields_keep_their_meaning(self):
        """fingerprint over the CLI seed (not the derived one) and hash over the pool file, as before."""
        out = self.tmp / "pool.jsonl"
        sample_candidates.main(["--config", str(self.config(MOCK_CONFIG)), "--data", str(self.data),
                                "--out", str(out), "--seed", "3", "--split", "dev"])
        meta = self.meta(out)
        tasks = [json.loads(line) for line in (self.data / "dev.jsonl").read_text().splitlines() if line.strip()]
        config = meta["config"]
        # source_hash is the fingerprint's own input; it is shared with the code, not re-derived.
        expected = canonical_sha256({"model": config["model"], "revision": config["revision"],
                                     "backend": config["backend"], "generation": config["generation"], "seed": 3,
                                     "tasks": tasks, "source_hash": source_hash()})
        self.assertEqual(meta["fingerprint"], expected)
        self.assertEqual(meta["hash"], sha256(out))
        self.assertEqual(meta["backend"], "mock")
        self.assertEqual(meta["decoding"]["composition"], None)

    def test_the_meta_names_the_commit_it_ran(self):
        out = self.tmp / "pool.jsonl"
        sample_candidates.main(["--config", str(self.config(MOCK_CONFIG)), "--data", str(self.data),
                                "--out", str(out)])
        code = self.meta(out)["code"]
        self.assertEqual(code.get("source_hash"), source_hash())
        try:
            head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            self.assertIsNone(code.get("git_commit"))
            self.assertTrue(code.get("git_reason"))
        else:
            self.assertEqual(code.get("git_commit"), head)

    def test_two_pools_for_different_studies_differ_where_the_runner_compares_them(self):
        """The carrier the runner's cross-pool check needs: identity and split are in each meta."""
        config = self.config(MOCK_CONFIG)
        metas = []
        for name, argv in (("a.jsonl", ["--split", "dev", "--study-id", "s-a", "--phase", "pilot"]),
                           ("b.jsonl", ["--split", "train_001", "--study-id", "s-b", "--phase", "main"])):
            sample_candidates.main(["--config", str(config), "--data", str(self.data),
                                    "--out", str(self.tmp / name)] + argv)
            metas.append(self.meta(self.tmp / name))
        a, b = metas
        self.assertNotEqual(a["identity"], b["identity"])
        self.assertNotEqual(a["split"]["sha256"], b["split"]["sha256"])
        self.assertEqual(a["data"]["manifest_sha256"], b["data"]["manifest_sha256"])


class CliTests(Fixture):
    """The same through the real command line, in a subprocess."""

    def run_cli(self, argv):
        # PYTHON_COLORS/NO_COLOR: Python 3.14's argparse may colour --help even into a pipe.
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                   PYTHON_COLORS="0", NO_COLOR="1")
        return subprocess.run([sys.executable, str(REPO / "sample_candidates.py")] + [str(a) for a in argv],
                              cwd=str(REPO), capture_output=True, text=True, env=env, timeout=300)

    def test_help_lists_the_new_options(self):
        result = self.run_cli(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for words in ("--split SPLIT", "--study-id STUDY_ID", "--phase {pilot,main}"):
            self.assertIn(words, result.stdout)

    def test_a_phase_argparse_does_not_know_is_refused_by_argparse(self):
        result = self.run_cli(["--config", self.config(HF_CONFIG), "--data", self.data,
                               "--out", self.tmp / "pool.jsonl", "--phase", "formal"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_an_hf_pilot_on_train_001_is_refused_before_the_pin(self):
        """The config's revision is not the anchor, so a run that reached the pin would say "base pin"."""
        out = self.tmp / "pool.jsonl"
        result = self.run_cli(["--config", self.config(dict(HF_CONFIG, revision=SYNTHETIC)), "--data", self.data,
                               "--out", out, "--split", "train_001", "--study-id", "s1", "--phase", "pilot"])
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pilot reads dev only", output)
        self.assertNotIn("base pin", output)
        self.assertFalse(out.exists())

    def test_a_mock_dev_pool_end_to_end_records_its_argv(self):
        out = self.tmp / "pool.jsonl"
        argv = ["--config", str(self.config(MOCK_CONFIG)), "--data", str(self.data), "--out", str(out),
                "--seed", "1", "--split", "dev", "--study-id", "s1", "--phase", "pilot"]
        result = self.run_cli(argv)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        meta = self.meta(out)
        self.assertEqual(meta["argv"]["args"], argv)
        self.assertEqual(os.path.normcase(meta["argv"]["script"]),
                         os.path.normcase(str((REPO / "sample_candidates.py").resolve())))
        self.assertEqual(meta["split"]["sha256"], sha256(self.data / "dev.jsonl"))
        drawn = {json.loads(line)["task_id"] for line in out.read_text().splitlines()}
        self.assertEqual(drawn, set(ids(self.data / "dev.jsonl")))


if __name__ == "__main__":
    unittest.main()
