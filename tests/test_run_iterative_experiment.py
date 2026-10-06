"""run_iterative_experiment.py: the multi-round R/S extension, with the GPU work stubbed.

Real here: the preflight, round 1's joint matching, the per-arm selection of
rounds 2+, the adapter chain, the records and resume.  Stubbed: the tokenizer
(words, so a supervised count is the word count + 1 EOS), `generate_pool` (a
deterministic fixture pool whose answers depend on the arm's adapter) and
`train_arm` (records its call, writes an adapter, returns the steps a real
trainer would take).  The real trainer and sampler run on the GPU in the
end-to-end record.
"""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import run_iterative_experiment
from generate_data import generate
from rsi.common import digest, file_hash, load_config, read_json, read_jsonl, verify_dataset, write_json, write_jsonl
from rsi.experiment import pin_config
from rsi.shared_adapter import lora_protocol
from rsi.tasks import judge, shortest_path
from rsi.matching import quotas
from test_run_matched_experiment import BLOCKS, block_rows

# An exported RSI_BASE_PIN must not leak into these tests (tests/pin_isolation.py).
from pin_isolation import isolate as setUpModule, restore as tearDownModule  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "matched_iterative.json"
SEED = 0
STUDY = "md-iter-test-v01"
PARAMETER_HASH = "a" * 64
# The base the fixture pools were "sampled" from: the config's, which the hf entries require of a pool.
BASE = {key: read_json(CONFIG)[key] for key in ("model", "revision")}
LATER_PROMPTS, LATER_SAMPLES = 30, 4


def iterative_config(directory, prompts=LATER_PROMPTS, samples=LATER_SAMPLES, base=CONFIG):
    """`base` with this fixture's later-round pool: the entry reads it from the config
    (generation.prompts_per_pool prompts x generation.candidates answers), never from its command line."""
    raw = read_json(base)
    raw["generation"].update(prompts_per_pool=prompts, candidates=samples)
    # The fixture's own training setting (fake_train takes len(rows) // 4 steps), so a change to the
    # shipped config's values does not break these tests.
    raw["training"].update(epochs=1, batch_size=2, effective_batch_size=4)
    path = Path(directory) / ("iterative_%s_%s_%s.json" % (Path(base).stem, prompts, samples))
    write_json(path, raw)
    return path


def pool_meta(path, **base_changes):
    """A fixture pool's meta, sampled from the config's base unless `base_changes` says otherwise."""
    return dict({"fingerprint": "fixture", "hash": file_hash(path), "backend": "hf"}, **dict(BASE, **base_changes))


class WordTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


def fixture_pool(tasks, samples, seed, adapter):
    """A pool with correct, nonshortest, endpoint and format answers; which sample
    carries which depends on the adapter path, so R's and S's pools differ."""
    shift = int(digest([str(adapter), seed])[:2], 16)
    rows = []
    for index, task in enumerate(tasks):
        path = shortest_path(task)
        walk = [path[0], path[1], path[0]] + path[1:]
        options = [json.dumps(path), json.dumps(walk), json.dumps(walk[:-1] + [path[0]]), "[]"]
        for sample in range(samples):
            response = options[(sample + index + shift) % 4]
            rows.append({"id": digest([task["id"], sample]), "task_id": task["id"], "sample": sample,
                         "response": response, "completion_tokens": len(response.split()), "prompt_tokens": 1,
                         "truncated": False})
    return rows


class IterFixture(unittest.TestCase):
    later_prompts = LATER_PROMPTS

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.data = cls.root / "data"
        generate(cls.data, "graph", 7, 4, 40, 4, 4, 8, 8)
        cls.tasks = {split: read_jsonl(cls.data / ("train_%03d.jsonl" % split)) for split in range(1, 5)}
        cls.pools = cls.root / "pools"
        for name, spec in BLOCKS.items():
            path = cls.pools / (name + ".jsonl")
            write_jsonl(path, block_rows(cls.tasks[1], **spec))
            write_json(str(path) + ".meta.json", pool_meta(path))
        cls.config = iterative_config(cls.root, cls.later_prompts)
        cls.resolved = pin_config(load_config(cls.config, SEED, "hf"))
        cls.adapter, cls.reference = cls.write_inputs(cls.root / "inputs")
        cls.generated, cls.trained = [], []
        cls.out = cls.root / "out"
        cls.code, cls.message = cls.run_entry(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def write_inputs(cls, directory, h_config=None):
        adapter, reference = Path(directory) / "shared_adapter", Path(directory) / "reference"
        adapter.mkdir(parents=True)
        reference.mkdir(parents=True)
        write_json(adapter / "shared_adapter.json", {"parameter_hash": PARAMETER_HASH, "base_model": cls.resolved["model"],
                                                     "base_revision": cls.resolved["revision"],
                                                     "protocol": lora_protocol(cls.resolved["training"])})
        torch.save({"lora": torch.zeros(2)}, reference / "h.pt")
        write_json(reference / "h.json", {
            "h": {"file": "h.pt", "sha256": file_hash(reference / "h.pt")},
            "bindings": {"base_model": cls.resolved["model"], "base_revision": cls.resolved["revision"],
                         "shared_adapter_parameter_hash": PARAMETER_HASH,
                         "config_hash": h_config or digest(cls.resolved),
                         "data": {"dataset_hash": digest(verify_dataset(cls.data))}}})
        return adapter, reference

    @classmethod
    def fake_generate(cls, config, adapter, tasks, samples, seed):
        cls.generated.append({"adapter": str(adapter), "task_ids": [t["id"] for t in tasks], "samples": samples})
        return fixture_pool(tasks, samples, seed, adapter)

    @classmethod
    def fake_train(cls, config, adapter_in, rows, out_adapter, seed, h_path):
        cls.trained.append({"adapter_in": str(adapter_in), "out": str(out_adapter), "rows": [r["id"] for r in rows],
                            "h_path": h_path})
        Path(out_adapter).mkdir(parents=True)
        (Path(out_adapter) / "adapter_model.safetensors").write_text("stub")
        return {"trained": True, "steps": len(rows) // 4, "mean_loss": 0.5, "delta_theta_norm": 0.1,
                "h_T_delta_theta": 0.01, "determinism": {"deterministic_algorithms": True}}

    @classmethod
    def bound_to(cls, directory, config):
        """Run overrides for `config`: the config, and a shared adapter and h bound to it."""
        resolved = pin_config(load_config(config, SEED, "hf"))
        adapter, reference = cls.write_inputs(directory, h_config=digest(resolved))
        return {"config": str(config), "shared_adapter": str(adapter), "reference_gradient": str(reference)}

    @classmethod
    def run_entry(cls, out, generate_pool=None, train_arm=None, **overrides):
        args = argparse.Namespace(config=str(cls.config), data=str(cls.data), shared_pool=str(cls.pools), out=str(out),
                                  seed=SEED, study_id=STUDY, phase="pilot", shared_adapter=str(cls.adapter),
                                  reference_gradient=str(cls.reference), resume=False, extend_rounds=False)
        for key, value in overrides.items():
            setattr(args, key, value)
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr), \
                mock.patch.object(run_iterative_experiment, "load_tokenizer", return_value=WordTokenizer()), \
                mock.patch.object(run_iterative_experiment, "generate_pool", generate_pool or cls.fake_generate), \
                mock.patch.object(run_iterative_experiment, "train_arm", train_arm or cls.fake_train):
            try:
                code = run_iterative_experiment.main(args)
            except SystemExit as exit:
                code = exit.code
        message = stderr.getvalue()
        if isinstance(code, str):
            code, message = 1, code
        return code, message

    def scratch(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def arm_dir(self, block, arm, out=None):
        return (out or self.out) / block / arm


class IterRunTests(IterFixture):
    def test_the_configured_ratio_and_tolerance_reach_every_selection(self):
        directory = self.scratch()
        raw = read_json(self.config)
        raw["matching"] = {"error_fraction": 0.125, "token_tolerance": 3}
        config = directory / "matching.json"
        write_json(config, raw)
        out = directory / "out"
        code, message = self.run_entry(out, **self.bound_to(directory, config))
        self.assertIn(code, (0, 3), message)
        experiment = read_json(out / "experiment.json")
        self.assertEqual(experiment["matching_settings"], {"error_fraction": 0.125, "token_tolerance": 3})
        c, e = quotas(experiment["K"], 0.125)
        matched = read_json(out / "matching" / "matched_subsets.json")
        self.assertEqual((matched["error_fraction"], matched["token_tolerance"]), (0.125, 3))
        for record in matched["per_block"].values():
            self.assertEqual((record["certificate"]["C"], record["certificate"]["E"]), (c, e))
            self.assertEqual(record["audit"]["token_length_tolerance"], 3)
        later = sorted(out.glob("b*/*/round_002/selection.json"))
        self.assertTrue(later)
        for path in later:
            certificate = read_json(path)["certificate"]
            self.assertEqual((certificate["C"], certificate["E"]), (c, e), path)

    def test_every_block_runs_four_rounds_per_arm_in_block_order(self):
        self.assertEqual(self.code, 0, self.message)
        experiment = read_json(self.out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["rounds"], experiment["K"]), ("complete", 4, 16))
        expected = ["%s/%s/round_%03d" % (b, a, t) for b in BLOCKS for t in range(1, 5) for a in ("R", "S")]
        self.assertEqual(experiment["completed"], expected)

    def test_each_arm_continues_from_its_own_previous_round(self):
        calls = {(Path(c["out"]).parts[-4], Path(c["out"]).parts[-3], Path(c["out"]).parts[-2]): c
                 for c in self.trained[:24]}
        for block in BLOCKS:
            for arm in ("R", "S"):
                self.assertEqual(Path(calls[block, arm, "round_001"]["adapter_in"]), self.adapter)
                for t in range(2, 5):
                    self.assertEqual(Path(calls[block, arm, "round_%03d" % t]["adapter_in"]),
                                     self.arm_dir(block, arm) / ("round_%03d" % (t - 1)) / "adapter")
                for c in calls.values():
                    self.assertEqual(c["h_path"], str(self.reference / "h.pt"))

    def test_round_one_is_the_joint_matched_construction(self):
        for block in BLOCKS:
            r = read_jsonl(self.arm_dir(block, "R") / "round_001" / "training.jsonl")
            s = read_jsonl(self.arm_dir(block, "S") / "round_001" / "training.jsonl")
            by_task = {x["id"]: x for x in self.tasks[1]}

            def correct(rows):
                return {x["id"] for x in rows if judge(by_task[x["task_id"]], x["response"])["correct"]}

            self.assertEqual([x["task_id"] for x in r], [x["task_id"] for x in s])
            self.assertEqual(correct(r), correct(s))
            self.assertEqual([x["tokens"] for x in r], [x["tokens"] for x in s])
            self.assertFalse({"correct", "error"} & set(r[0]))  # no label enters the training file
            self.assertEqual(read_json(self.arm_dir(block, "R") / "round_001" / "selection.json")["rule"],
                             "joint matched construction over the shared round-1 pool")

    def test_later_rounds_sample_each_arm_from_its_own_model_on_common_prompts(self):
        for block in BLOCKS:
            for t in range(2, 5):
                prompts = [x["id"] for x in self.tasks[t][:LATER_PROMPTS]]
                for arm in ("R", "S"):
                    pool = read_jsonl(self.arm_dir(block, arm) / ("round_%03d" % t) / "pool.jsonl")
                    self.assertEqual(sorted({r["task_id"] for r in pool}), sorted(prompts))
                    self.assertEqual(len(pool), LATER_PROMPTS * LATER_SAMPLES)
                    selection = read_json(self.arm_dir(block, arm) / ("round_%03d" % t) / "selection.json")
                    self.assertEqual((selection["audit"]["K"], selection["audit"]["C"]), (16, 12))
                    rows = read_jsonl(self.arm_dir(block, arm) / ("round_%03d" % t) / "training.jsonl")
                    self.assertEqual([r["id"] for r in rows], selection["ids"])
                    by_task = {x["id"]: x for x in self.tasks[t]}
                    verdicts = [judge(by_task[r["task_id"]], r["response"]) for r in rows]
                    self.assertEqual(sum(v["correct"] for v in verdicts), 12)
                    if arm == "S":
                        self.assertEqual({v["error"] for v in verdicts if not v["correct"]}, {"nonshortest"})
        adapters = {g["adapter"] for g in self.generated[:18]}
        self.assertNotIn(str(self.adapter), adapters)
        self.assertEqual(len(adapters), 18)

    def test_every_arm_reads_as_an_evaluate_py_run(self):
        for block in BLOCKS:
            for arm in ("R", "S"):
                run = read_json(self.arm_dir(block, arm) / "run.json")
                self.assertEqual(run["config"]["rounds"], 4)
                self.assertEqual(run["dataset_hash"], digest(verify_dataset(self.data)))
                self.assertEqual(Path(run["data_path"]), self.data)
                self.assertEqual(read_json(self.arm_dir(block, arm) / "finished.json")["rounds"], 4)
                for t in range(1, 5):
                    complete = read_json(self.arm_dir(block, arm) / ("round_%03d" % t) / "complete.json")
                    self.assertEqual(complete["adapter"], "round_%03d/adapter" % t)
                    self.assertTrue((self.arm_dir(block, arm) / complete["adapter"]).is_dir())
                    self.assertEqual(complete["training"]["steps"], 4)


class IterStopTests(IterFixture):
    def test_an_arm_that_cannot_fill_its_quota_stops_and_the_other_continues(self):
        def no_targets_for_s(config, adapter, tasks, samples, seed):
            rows = self.fake_generate(config, adapter, tasks, samples, seed)
            if "/S/" not in Path(adapter).as_posix():
                return rows
            by_task = {t["id"]: t for t in tasks}
            return [dict(r, response="[]") if judge(by_task[r["task_id"]], r["response"])["error"] == "nonshortest"
                    else r for r in rows]

        out = self.scratch() / "out"
        code, message = self.run_entry(out, generate_pool=no_targets_for_s)
        self.assertEqual(code, 3, message)
        experiment = read_json(out / "experiment.json")
        self.assertEqual(experiment["status"], "complete_with_stopped_arms")
        self.assertEqual(sorted(experiment["stopped"]), sorted("%s/S" % b for b in BLOCKS))
        for block in BLOCKS:
            selection = read_json(out / block / "S" / "round_002" / "selection.json")
            self.assertFalse(selection["certificate"]["feasible"])
            self.assertFalse((out / block / "S" / "round_003").exists())
            self.assertFalse((out / block / "S" / "finished.json").exists())
            self.assertTrue((out / block / "R" / "finished.json").exists())
        # A resumed run still reports the arms an earlier attempt stopped.
        code, message = self.run_entry(out, generate_pool=no_targets_for_s, resume=True)
        self.assertEqual(code, 3, message)
        resumed = read_json(out / "experiment.json")
        self.assertEqual(resumed["status"], "complete_with_stopped_arms")
        self.assertEqual(sorted(resumed["stopped"]), sorted("%s/S" % b for b in BLOCKS))

    def test_a_trainer_that_takes_another_step_count_fails_the_run(self):
        def wrong_steps(*args):
            return dict(self.fake_train(*args), steps=1)

        out = self.scratch() / "out"
        with self.assertRaises(RuntimeError):
            self.run_entry(out, train_arm=wrong_steps)
        self.assertEqual(read_json(out / "experiment.json")["status"], "failed")


class IterResumeTests(IterFixture):
    def test_resume_skips_finished_rounds_and_redoes_the_interrupted_one(self):
        out = self.scratch() / "out"
        calls = []

        def crash_on_b01_round_2_s(config, adapter_in, rows, out_adapter, seed, h_path):
            if Path(out_adapter).as_posix().endswith("b01/S/round_002/adapter"):
                Path(out_adapter).mkdir(parents=True)
                raise RuntimeError("killed")
            calls.append(Path(out_adapter).as_posix())
            return self.fake_train(config, adapter_in, rows, out_adapter, seed, h_path)

        with self.assertRaises(RuntimeError):
            self.run_entry(out, train_arm=crash_on_b01_round_2_s)
        first = list(calls)
        calls.clear()

        def counting(config, adapter_in, rows, out_adapter, seed, h_path):
            calls.append(Path(out_adapter).as_posix())
            return self.fake_train(config, adapter_in, rows, out_adapter, seed, h_path)

        code, message = self.run_entry(out, train_arm=counting, resume=True)
        self.assertEqual(code, 0, message)
        self.assertFalse(set(first) & set(calls))
        self.assertTrue(calls[0].endswith("b01/S/round_002/adapter"))
        self.assertTrue(list((out / "b01" / "S").glob("round_002.incomplete-*")))
        self.assertEqual(read_json(out / "experiment.json")["status"], "complete")

    def test_resume_refuses_a_run_with_other_inputs(self):
        out = self.scratch() / "out"
        self.run_entry(out)
        # Another later-round pool size is another config, so another identity.
        other = self.bound_to(self.scratch(), iterative_config(self.scratch(), LATER_PROMPTS - 2))
        code, message = self.run_entry(out, resume=True, **other)
        self.assertEqual(code, 1)
        self.assertIn("cannot resume", message)

    def test_without_resume_an_existing_run_is_never_overwritten(self):
        code, message = self.run_entry(self.out)
        self.assertEqual(code, 1)
        self.assertIn("--resume", message)


def relative_pool(config, adapter, tasks, samples, seed):
    """fixture_pool keyed by the adapter's place in its run (block/arm/round), not its absolute path, so
    two runs in different directories sample alike."""
    return fixture_pool(tasks, samples, seed, "/".join(Path(adapter).parts[-4:]))


class IterExtendTests(IterFixture):
    """--extend-rounds: a complete 4-round run continued to 6 rounds is the 6-round run."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.data = cls.root / "data"
        generate(cls.data, "graph", 7, 6, 40, 4, 4, 8, 8)
        cls.tasks = {split: read_jsonl(cls.data / ("train_%03d.jsonl" % split)) for split in range(1, 7)}
        cls.pools = cls.root / "pools"
        for name, spec in BLOCKS.items():
            path = cls.pools / (name + ".jsonl")
            write_jsonl(path, block_rows(cls.tasks[1], **spec))
            write_json(str(path) + ".meta.json", pool_meta(path))
        cls.config = cls.with_rounds(4)
        cls.config6 = cls.with_rounds(6)
        cls.resolved = pin_config(load_config(cls.config, SEED, "hf"))
        # The shared adapter and h of the 4-round run: h is bound to the config a run begins with.
        cls.adapter, cls.reference = cls.write_inputs(cls.root / "inputs")
        cls.generated, cls.trained = [], []

    @classmethod
    def with_rounds(cls, rounds, audit=None, **training):
        raw = read_json(iterative_config(cls.root, cls.later_prompts))
        raw["rounds"] = rounds
        raw["training"].update(training)
        if audit is not None:
            raw["audit"] = audit
        path = cls.root / ("rounds_%d_%s.json" % (rounds, digest([training, audit])[:8]))
        write_json(path, raw)
        return path

    @classmethod
    def recording_train(cls, config, adapter_in, rows, out_adapter, seed, h_path):
        """fake_train taking the steps the real trainer takes (audited-out rows have weight 0), whose stats
        also hold the seed and the settings it was given, so complete.json shows what every round's
        training received."""
        from rsi.budgeted_auditing import positive_rows
        training = config["training"]
        steps = -(-len(positive_rows(rows)) // training["effective_batch_size"]) * training["epochs"]
        return dict(cls.fake_train(config, adapter_in, rows, out_adapter, seed, h_path), steps=steps, seed=seed,
                    training_config=training, generation_config=config["generation"])

    def four_rounds(self, **overrides):
        out = self.scratch() / "out"
        code, message = self.run_entry(out, generate_pool=relative_pool, **overrides)
        self.assertEqual(code, 0, message)
        return out

    def extend(self, out, **overrides):
        return self.run_entry(out, generate_pool=relative_pool,
                              **dict(dict(config=str(self.config6), extend_rounds=True), **overrides))

    def assertSameRun(self, straight, extended, rounds, files=("selection.json", "training.jsonl")):
        """Every round's records, finished.json and the run's outcome are the same in both runs."""
        for block in BLOCKS:
            for arm in ("R", "S"):
                for t in range(1, rounds + 1):
                    a, b = (run / block / arm / ("round_%03d" % t) for run in (straight, extended))
                    for name in files + (("pool.jsonl",) if t > 1 else ()):
                        self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), (block, arm, t, name))
                    self.assertEqual(dict(read_json(a / "complete.json"), seconds=None),
                                     dict(read_json(b / "complete.json"), seconds=None), (block, arm, t))
                self.assertEqual(read_json(straight / block / arm / "finished.json"),
                                 read_json(extended / block / arm / "finished.json"))
                manifest = read_json(extended / block / arm / "run.json")
                self.assertEqual(manifest["config"]["rounds"], rounds)  # what evaluate_multiround.py evaluates
                self.assertEqual(manifest["bindings"]["config_hash"],
                                 read_json(straight / block / arm / "run.json")["bindings"]["config_hash"])
        a, b = (read_json(run / "experiment.json") for run in (straight, extended))
        for key in ("status", "rounds", "K", "steps_per_round", "completed", "stopped"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(b["identity"]["config_hash"], a["identity"]["config_hash"])
        self.assertEqual(a["extensions"], [])
        return b

    def test_an_extended_run_is_the_run_with_more_rounds_from_the_start(self):
        straight = self.scratch() / "out"
        code, message = self.run_entry(straight, generate_pool=relative_pool, train_arm=self.recording_train,
                                       **self.bound_to(self.scratch(), self.config6))
        self.assertEqual(code, 0, message)
        extended = self.four_rounds(train_arm=self.recording_train)
        code, message = self.extend(extended, train_arm=self.recording_train)
        self.assertEqual(code, 0, message)
        experiment = self.assertSameRun(straight, extended, 6)
        (extension,) = experiment["extensions"]
        self.assertEqual((extension["from_rounds"], extension["to_rounds"], extension["source_changed"]), (4, 6, False))
        self.assertEqual(extension["previous_identity"]["config_hash"], digest(self.resolved))
        for block in BLOCKS:
            for arm in ("R", "S"):
                (entry,) = read_json(extended / block / arm / "run.json")["extensions"]
                self.assertEqual((entry["from_rounds"], entry["to_rounds"], entry["previous_finished"]["rounds"]),
                                 (4, 6, 4))
                self.assertEqual(read_json(extended / block / arm / "finished.4-rounds.json")["rounds"], 4)

    def test_an_extended_audited_run_is_the_audited_run_with_more_rounds(self):
        audit = {"policy": "uniform", "budget": 2, "weighting": True}
        audited4, audited6 = self.with_rounds(4, audit=audit), self.with_rounds(6, audit=audit)
        straight = self.scratch() / "out"
        code, message = self.run_entry(straight, generate_pool=relative_pool, train_arm=self.recording_train,
                                       **self.bound_to(self.scratch(), audited6))
        self.assertEqual(code, 0, message)
        inputs = self.bound_to(self.scratch(), audited4)
        extended = self.four_rounds(train_arm=self.recording_train, **inputs)
        code, message = self.extend(extended, train_arm=self.recording_train, **dict(inputs, config=str(audited6)))
        self.assertEqual(code, 0, message)
        self.assertSameRun(straight, extended, 6, files=("selection.json", "pre_audit.jsonl", "audit.json",
                                                         "training.jsonl"))
        for block in BLOCKS:
            for arm in ("R", "S"):
                rounds = [read_json(extended / block / arm / ("round_%03d" % t) / "complete.json") for t in range(1, 7)]
                # The budget accounting goes on across the extension.
                self.assertEqual(rounds[-1]["cumulative_audit_queries"], sum(r["audit_queries"] for r in rounds))
                self.assertGreater(rounds[-1]["cumulative_audit_queries"], rounds[3]["cumulative_audit_queries"])

    def test_a_run_is_extended_twice(self):
        out = self.four_rounds()
        for rounds in (5, 6):
            code, message = self.extend(out, config=str(self.with_rounds(rounds)))
            self.assertEqual(code, 0, message)
        experiment = read_json(out / "experiment.json")
        self.assertEqual([(e["from_rounds"], e["to_rounds"]) for e in experiment["extensions"]], [(4, 5), (5, 6)])
        self.assertEqual((experiment["status"], len(experiment["completed"])), ("complete", len(BLOCKS) * 2 * 6))
        for block in BLOCKS:
            for arm in ("R", "S"):
                arm_dir = out / block / arm
                self.assertEqual([(e["from_rounds"], e["to_rounds"]) for e in read_json(arm_dir / "run.json")["extensions"]],
                                 [(4, 5), (5, 6)])
                self.assertEqual([read_json(arm_dir / name)["rounds"] for name in
                                  ("finished.4-rounds.json", "finished.5-rounds.json", "finished.json")], [4, 5, 6])

    def test_more_rounds_without_extend_rounds_are_refused(self):
        out = self.four_rounds()
        code, message = self.run_entry(out, generate_pool=relative_pool, config=str(self.config6), resume=True)
        self.assertEqual(code, 1, message)
        self.assertIn("pass --extend-rounds", message)
        self.assertEqual(read_json(out / "experiment.json")["rounds"], 4)

    def test_only_a_complete_run_is_extended(self):
        def crash(config, adapter_in, rows, out_adapter, seed, h_path):
            if Path(out_adapter).as_posix().endswith("b01/S/round_002/adapter"):
                raise RuntimeError("killed")
            return self.fake_train(config, adapter_in, rows, out_adapter, seed, h_path)

        out = self.scratch() / "out"
        with self.assertRaises(RuntimeError):
            self.run_entry(out, generate_pool=relative_pool, train_arm=crash)
        code, message = self.extend(out)
        self.assertEqual(code, 1, message)
        self.assertIn("finish it first with the code and the config it began with", message)
        self.assertEqual(read_json(out / "experiment.json")["rounds"], 4)

    def test_nothing_but_rounds_may_change(self):
        out = self.four_rounds()
        code, message = self.extend(out, study_id="md-iter-other-v01")
        self.assertEqual(code, 1, message)
        self.assertIn("besides rounds, this run differs in study_id", message)
        # Another training setting, with an h bound to it at the run's 4 rounds: the extension refuses it.
        other6 = self.with_rounds(6, learning_rate=1e-4)
        inputs = self.bound_to(self.scratch(), self.with_rounds(4, learning_rate=1e-4))
        code, message = self.extend(out, **dict(inputs, config=str(other6)))
        self.assertEqual(code, 1, message)
        self.assertIn("besides rounds, this run differs in config_hash", message)
        # With the run's own h, the h check refuses it first.
        code, message = self.extend(out, config=str(other6))
        self.assertEqual(code, 1, message)
        self.assertIn("h.json in", message)
        self.assertEqual(read_json(out / "experiment.json")["rounds"], 4)

    def test_rounds_are_never_lowered(self):
        out = self.four_rounds()
        code, message = self.extend(out, config=str(self.with_rounds(2)))
        self.assertEqual(code, 1, message)
        self.assertIn("cannot resume", message)
        self.assertEqual(read_json(out / "experiment.json")["rounds"], 4)

    def test_a_code_change_is_recorded_by_an_extension_and_refused_by_a_resume(self):
        out = self.four_rounds()
        real = run_iterative_experiment.environment()

        def changed():
            return dict(real, source_hash="f" * 64)

        with mock.patch.object(run_iterative_experiment, "environment", changed):
            code, message = self.run_entry(out, generate_pool=relative_pool, resume=True)
            self.assertEqual(code, 1, message)
            self.assertIn("differs in source_hash", message)
            code, message = self.extend(out)
        self.assertEqual(code, 0, message)
        experiment = read_json(out / "experiment.json")
        (extension,) = experiment["extensions"]
        self.assertTrue(extension["source_changed"])
        self.assertEqual((extension["previous_code"]["source_hash"], extension["code"]["source_hash"]),
                         (real["source_hash"], "f" * 64))
        self.assertEqual(experiment["identity"]["source_hash"], "f" * 64)

    def test_code_that_matches_round_one_otherwise_is_refused_before_anything_is_written(self):
        real_env, real_select = run_iterative_experiment.environment(), run_iterative_experiment.select_k_for_blocks

        def other_draw(candidates, seed=0, **kwargs):
            return real_select(candidates, seed=seed + 1, **kwargs)

        for name, change in (("another K", mock.patch.object(run_iterative_experiment, "K_LADDER", (8,))),
                             ("other subsets", mock.patch.object(run_iterative_experiment, "select_k_for_blocks",
                                                                 other_draw))):
            with self.subTest(name):
                out = self.four_rounds()
                before = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*.json"))}
                with mock.patch.object(run_iterative_experiment, "environment",
                                       lambda: dict(real_env, source_hash="f" * 64)), change:
                    code, message = self.extend(out)
                self.assertEqual(code, 1, message)
                self.assertIn("round-1 matching is not the run's", message)
                self.assertEqual({p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*.json"))}, before)

    def test_an_interrupted_extension_resumes_like_any_run(self):
        def crash(config, adapter_in, rows, out_adapter, seed, h_path):
            if Path(out_adapter).as_posix().endswith("b01/S/round_005/adapter"):
                raise RuntimeError("killed")
            return self.fake_train(config, adapter_in, rows, out_adapter, seed, h_path)

        out = self.four_rounds()
        with self.assertRaises(RuntimeError):
            self.extend(out, train_arm=crash)
        experiment = read_json(out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["rounds"], len(experiment["extensions"])), ("failed", 6, 1))
        self.assertEqual(read_json(out / "b00" / "S" / "finished.json")["rounds"], 6)  # b00 ran its 6 rounds
        self.assertFalse((out / "b01" / "S" / "finished.json").exists())  # b01 has not
        # Now a 6-round run: a plain resume continues it, with h still checked at the 4 rounds it began with.
        code, message = self.run_entry(out, generate_pool=relative_pool, config=str(self.config6), resume=True)
        self.assertEqual(code, 0, message)
        experiment = read_json(out / "experiment.json")
        self.assertEqual((experiment["status"], len(experiment["extensions"])), ("complete", 1))
        self.assertEqual(len(experiment["completed"]), len(BLOCKS) * 2 * 6)
        for block in BLOCKS:
            for arm in ("R", "S"):
                self.assertEqual(len(read_json(out / block / arm / "run.json")["extensions"]), 1)
                self.assertEqual(read_json(out / block / arm / "finished.json")["rounds"], 6)


class IterRefusalTests(IterFixture):
    def assertRefused(self, fragment, **overrides):
        out = self.scratch() / "out"
        code, message = self.run_entry(out, **overrides)
        self.assertEqual(code, 1, message)
        self.assertIn(fragment, message)
        self.assertFalse(out.exists())

    def test_inputs_must_belong_to_this_run(self):
        self.assertRefused("--backend hf needs --shared-adapter", shared_adapter=None)
        adapter, reference = self.write_inputs(self.scratch(), h_config="d" * 64)
        self.assertRefused("config_hash", shared_adapter=str(adapter), reference_gradient=str(reference))

    def test_later_pools_must_fit_the_data(self):
        too_many = iterative_config(self.scratch(), 41)
        self.assertRefused("generation.prompts_per_pool is 41", **self.bound_to(self.scratch(), too_many))
        with self.assertRaises(ValueError):  # load_config: a positive integer or null
            self.run_entry(self.scratch() / "out", config=str(iterative_config(self.scratch(), 0)))

    def test_the_later_round_size_comes_from_the_config_alone(self):
        experiment = read_json(self.out / "experiment.json")
        self.assertEqual((experiment["later_pool"]["prompts"], experiment["later_pool"]["samples"]),
                         (LATER_PROMPTS, LATER_SAMPLES))
        self.assertIn("config", experiment["later_pool"]["source"])
        import subprocess
        import sys
        usage = subprocess.run([sys.executable, str(ROOT / "run_iterative_experiment.py"), "--help"],
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(usage.returncode, 0, usage.stderr)
        self.assertNotIn("--later-", usage.stdout)

    def test_a_round_one_pool_from_another_base_is_refused(self):
        for change in ({"model": "meta-llama/Llama-3.2-1B-Instruct"}, {"revision": "0" * 40}):
            pools = self.scratch() / "pools"
            for name, spec in BLOCKS.items():
                path = pools / (name + ".jsonl")
                write_jsonl(path, block_rows(self.tasks[1], **spec))
                write_json(str(path) + ".meta.json", pool_meta(path, **change))
            self.assertRefused("was sampled from %s" % next(iter(change)), shared_pool=str(pools))

    def test_a_round_one_pool_drawn_for_other_data_is_refused(self):
        pools = self.scratch() / "pools"
        for name, spec in BLOCKS.items():
            path = pools / (name + ".jsonl")
            write_jsonl(path, block_rows(self.tasks[1], **spec))
            write_json(str(path) + ".meta.json", pool_meta(path, data={"dataset_hash": "f" * 64}))
        self.assertRefused("data (a prompt or the questions changed since)", shared_pool=str(pools))

    def pools_with(self, generations):
        """Fixture pools whose metas record these generation settings, one per block."""
        pools = self.scratch() / "pools"
        for (name, spec), generation in zip(BLOCKS.items(), generations):
            path = pools / (name + ".jsonl")
            write_jsonl(path, block_rows(self.tasks[1], **spec))
            write_json(str(path) + ".meta.json", pool_meta(path, generation=generation))
        return pools

    def round_one_generation(self, **changes):
        """A pool config's generation: this run's sampling, with a round-1 pool's own size."""
        return dict(dict(self.resolved["generation"], candidates=8, prompts_per_pool=None), **changes)

    def test_a_round_one_pool_drawn_with_other_sampling_is_refused(self):
        for change in ({"temperature": 0.7}, {"top_p": 0.95}, {"top_k": 20}, {"max_new_tokens": 256},
                       {"max_sequence_length": 2048}):
            with self.subTest(change=change):
                pools = self.pools_with([self.round_one_generation(**change)] * len(BLOCKS))
                self.assertRefused("sampling settings", shared_pool=str(pools))

    def test_round_one_pools_must_be_drawn_alike(self):
        generations = [self.round_one_generation()] * len(BLOCKS)
        generations[-1] = self.round_one_generation(batch_size=4)
        self.assertRefused("not drawn alike", shared_pool=str(self.pools_with(generations)))

    def test_round_one_pools_of_their_own_size_with_this_sampling_are_read(self):
        pools = self.pools_with([self.round_one_generation()] * len(BLOCKS))
        code, message = self.run_entry(self.scratch() / "out", shared_pool=str(pools))
        self.assertEqual(code, 0, message)


if __name__ == "__main__":
    unittest.main()


def arithmetic_options(task):
    """Correct, ignore_parentheses (when it differs), off-by-one and an unparseable answer."""
    from rsi.tasks import arithmetic_value
    truth = arithmetic_value(task["expression"])
    no_parens = arithmetic_value(task["expression"].replace("(", "").replace(")", ""))
    return [str(truth), str(no_parens if no_parens != truth else truth + 2), str(truth + 1), "answer"]


def arithmetic_pool(tasks, samples, seed, adapter):
    shift = int(digest([str(adapter), seed])[:2], 16)
    rows = []
    for index, task in enumerate(tasks):
        options = arithmetic_options(task)
        for sample in range(samples):
            response = options[(sample + index + shift) % 4]
            rows.append({"id": digest([task["id"], sample]), "task_id": task["id"], "sample": sample,
                         "response": response, "completion_tokens": 1, "prompt_tokens": 1, "truncated": False})
    return rows


class ArithmeticIterTests(IterFixture):
    """The same run on an arithmetic dataset: S's target is ignore_parentheses in every round."""

    # K is 32 on these pools, so a round needs 32 distinct prompts: give it all 40.
    later_prompts = 40

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.data = cls.root / "data"
        generate(cls.data, "arithmetic", 7, 4, 40, 4, 4, 8, 8)
        cls.tasks = {split: read_jsonl(cls.data / ("train_%03d.jsonl" % split)) for split in range(1, 5)}
        cls.pools = cls.root / "pools"
        for b, name in enumerate(BLOCKS):
            path = cls.pools / (name + ".jsonl")
            write_jsonl(path, arithmetic_pool(cls.tasks[1], 4, b, "round-1"))
            write_json(str(path) + ".meta.json", pool_meta(path))
        cls.config = iterative_config(cls.root, cls.later_prompts)
        cls.resolved = pin_config(load_config(cls.config, SEED, "hf"))
        cls.adapter, cls.reference = cls.write_inputs(cls.root / "inputs")
        cls.generated, cls.trained = [], []
        cls.out = cls.root / "out"
        cls.code, cls.message = cls.run_entry(cls.out, generate_pool=cls.fake_arithmetic_generate)

    @classmethod
    def fake_arithmetic_generate(cls, config, adapter, tasks, samples, seed):
        cls.generated.append({"adapter": str(adapter)})
        return arithmetic_pool(tasks, samples, seed, adapter)

    def test_s_trains_on_ignore_parentheses_errors_in_every_round(self):
        self.assertEqual(self.code, 0, self.message)
        experiment = read_json(self.out / "experiment.json")
        self.assertEqual((experiment["task"], experiment["target_signatures"]), ("arithmetic", ["ignore_parentheses"]))
        for block in BLOCKS:
            for t in range(1, 5):
                by_task = {x["id"]: x for x in self.tasks[t]}
                for arm in ("R", "S"):
                    rows = read_jsonl(self.arm_dir(block, arm) / ("round_%03d" % t) / "training.jsonl")
                    errors = [judge(by_task[r["task_id"]], r["response"])["error"] for r in rows]
                    errors = [e for e in errors if e != "correct"]
                    self.assertEqual(len(errors), experiment["K"] // 4)
                    if arm == "S":
                        self.assertEqual(set(errors), {"ignore_parentheses"}, (block, t))
