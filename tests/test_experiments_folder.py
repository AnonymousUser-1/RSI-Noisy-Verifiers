"""experiments/: the experiments, their configs and the pool check.

The configs are checked as committed: every field every stage reads is stated, 2,048 prompts x 8
answers in every round, auditing on only in the audited ones, and within a task the two configs
differ only in auditing, so by default the audited run reuses the unaudited run's round-1 pools.
graph_unaudited_qwen3-4b has no audited sibling. It generates its own inputs by default and also
permits explicit import of the original inputs for one-step replay.
"""
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_multiround import read_evaluation_config
from experiments import pool_check
from generate_data import generate
from rsi.base_pin import read_base_pin
from rsi.common import digest, file_hash, load_config, require_explicit, verify_dataset, write_json, write_jsonl

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("graph_unaudited", "graph_audited", "arithmetic_unaudited", "arithmetic_audited",
               "graph_unaudited_qwen3-4b",
               "graph_unaudited_llama3.2-3b", "graph_audited_llama3.2-3b",
               "arithmetic_unaudited_llama3.2-3b", "arithmetic_audited_llama3.2-3b",
               "arithmetic_unaudited_llama3.2-1b", "arithmetic_audited_llama3.2-1b")
STAGES = ("pool", "pilot", "adapter", "reference", "iterative")
UNPAIRED = ("graph_unaudited_qwen3-4b",)    # no audited sibling in this release


def settings(name):
    """settings.sh's KEY=VALUE lines (values unquoted)."""
    found = {}
    for line in (REPO / "experiments" / name / "settings.sh").read_text().splitlines():
        m = re.match(r'^([A-Z_]+)=(.*)$', line)
        if m:
            found[m.group(1)] = m.group(2).strip().strip('"')
    return found


class ExperimentConfigTests(unittest.TestCase):
    def test_every_experiment_has_its_own_three_files(self):
        for name in EXPERIMENTS:
            for f in ("settings.sh", "config.json", "evaluation.json"):
                self.assertTrue((REPO / "experiments" / name / f).is_file(), (name, f))

    def test_settings(self):
        studies = set()
        for name in EXPERIMENTS:
            s = settings(name)
            self.assertEqual(s["TASK"], name.split("_")[0], name)
            self.assertEqual(s["SEEDS"].split(), ["0", "1", "2", "3", "4"], name)
            sibling = name.replace("unaudited", "X").replace("audited", "unaudited").replace("X", "audited")
            self.assertEqual(s["SHARE_POOLS_WITH"], "" if name in UNPAIRED else sibling, name)
            studies.add(s["STUDY"])
        self.assertEqual(len(studies), len(EXPERIMENTS))

    def test_configs_state_every_field_of_every_stage(self):
        for name in EXPERIMENTS:
            path = REPO / "experiments" / name / "config.json"
            load_config(path)
            for stage in STAGES:
                require_explicit(path, stage)

    def test_configs_carry_every_generation_field_the_main_configs_do(self):
        # A field main adds to the matched configs (as #25 added repetition_stop) must reach these too,
        # or every stage refuses them.
        main = json.loads((REPO / "configs" / "matched_iterative.json").read_text())["generation"]
        for name in EXPERIMENTS:
            ours = json.loads((REPO / "experiments" / name / "config.json").read_text())["generation"]
            self.assertEqual(set(ours), set(main), name)
            if "qwen3-4b" not in name:   # imported pools, drawn before the field (the test below)
                self.assertEqual(ours["repetition_stop"], main["repetition_stop"], name)

    def test_2048_prompts_x_8_answers_in_every_round(self):
        for name in EXPERIMENTS:
            config = load_config(REPO / "experiments" / name / "config.json")
            # Round t samples train_00t: the study's data has train_001 ... train_008.
            self.assertIn(config["rounds"], range(1, 9), name)
            self.assertEqual((config["generation"]["prompts_per_pool"], config["generation"]["candidates"]), (None, 8))

    def test_auditing_only_in_the_audited_experiments(self):
        for name in EXPERIMENTS:
            audit = load_config(REPO / "experiments" / name / "config.json")["audit"]
            expected = ({"policy": "adaptive", "budget": 16, "weighting": True} if "_audited" in name
                        else {"policy": "none", "budget": 0, "weighting": True})
            self.assertEqual(audit, expected, name)

    def test_an_experiment_and_its_audited_sibling_differ_only_in_auditing(self):
        for name in (n for n in EXPERIMENTS if "_unaudited" in n and n not in UNPAIRED):
            sibling = name.replace("_unaudited", "_audited")
            self.assertIn(sibling, EXPERIMENTS)
            a, b = (json.loads((REPO / "experiments" / n / "config.json").read_text()) for n in (name, sibling))
            self.assertEqual(dict(a, audit=None), dict(b, audit=None), name)

    def test_imported_pools_are_stated_as_drawn(self):
        # The Qwen3-4B pools were drawn before repetition_stop existed: their config states it off, and
        # their evaluation states the current rule.  Every other experiment draws its own pools.
        for name in EXPERIMENTS:
            s = settings(name)
            config = load_config(REPO / "experiments" / name / "config.json")
            evaluation = json.loads((REPO / "experiments" / name / "evaluation.json").read_text())
            if "qwen3-4b" in name:
                self.assertNotIn("POOLS_IMPORTED", s, name)
                self.assertEqual((s["POOLS_IMPORT_ALLOWED"], s["DATA_NAME"]), ("yes", "graph-qwen3-4b-import"), name)
                self.assertIsNone(config["generation"]["repetition_stop"], name)
                self.assertEqual(evaluation["repetition_stop"], {"span": 128, "max_period": 32}, name)
            else:
                self.assertNotIn("POOLS_IMPORTED", s, name)
                self.assertEqual(config["generation"]["repetition_stop"], {"span": 128, "max_period": 32}, name)
                self.assertNotIn("repetition_stop", evaluation, name)

    def test_the_pin_names_the_config_s_base(self):
        for name in EXPERIMENTS:
            pin = read_base_pin(REPO / settings(name)["PIN_FILE"])
            config = load_config(REPO / "experiments" / name / "config.json")
            self.assertEqual((pin["model"], pin["revision"]), (config["model"], config["revision"]), name)

    def test_evaluation_configs(self):
        for name in EXPERIMENTS:
            config = read_evaluation_config(REPO / "experiments" / name / "evaluation.json")
            self.assertEqual(config["splits"], ["eval_id", "eval_ood"], name)


class PoolCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = cls.root / "data"
        generate(cls.data, "graph", 7, 2, 8, 4, 4, 4, 4)
        cls.config = REPO / "experiments" / "graph_unaudited" / "config.json"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pool(self, **changes):
        """A pool and its meta as sample_candidates.py writes them for seed 3, with `changes` to the meta."""
        config = load_config(self.config)
        path = Path(tempfile.mkdtemp(dir=self.root)) / "b03.jsonl"
        write_jsonl(path, [{"id": "x", "task_id": "t", "response": "[]"}])
        meta = {"hash": file_hash(path), "backend": "hf", "model": config["model"], "revision": config["revision"],
                "generation": config["generation"], "config": config, "seed": {"cli": 3},
                "split": {"name": "train_001"}, "data": {"dataset_hash": digest(verify_dataset(self.data))}}
        for key, value in changes.items():
            meta[key] = value
        write_json(str(path) + ".meta.json", meta)
        return path

    def check(self, path, seed=3, config=None):
        return pool_check.differences(config or self.config, self.data, path, seed)

    def test_a_pool_drawn_with_the_config_matches(self):
        self.assertEqual(self.check(self.pool()), [])

    def test_the_audited_config_matches_the_same_pool(self):
        self.assertEqual(self.check(self.pool(), config=REPO / "experiments" / "graph_audited" / "config.json"), [])

    def test_each_difference_is_named(self):
        config = load_config(self.config)
        cases = {
            "generation temperature": dict(generation=dict(config["generation"], temperature=1.0)),
            "generation candidates": dict(generation=dict(config["generation"], candidates=4)),
            "model": dict(model="Qwen/Qwen3-4B"),
            "dtype": dict(config=dict(config, dtype="float16")),
            "backend": dict(backend="mock"),
            "seed": dict(seed={"cli": 4}),
            "split": dict(split={"name": "dev"}),
            "another dataset": dict(data={"dataset_hash": "0" * 64}),
        }
        for words, changes in cases.items():
            found = self.check(self.pool(**changes))
            self.assertTrue(any(words in f for f in found), (words, found))

    def test_a_pool_from_before_repetition_stop_matches_a_config_that_states_it_off(self):
        config = load_config(self.config)
        legacy = {k: v for k, v in config["generation"].items() if k != "repetition_stop"}
        path = self.pool(generation=legacy)
        self.assertTrue(any("repetition_stop None" in f for f in self.check(path)))   # the current rule: refused
        off = Path(tempfile.mkdtemp(dir=self.root)) / "config.json"
        write_json(off, dict(json.loads(self.config.read_text()),
                             generation=dict(config["generation"], repetition_stop=None)))
        self.assertTrue(any("model" in f or "dtype" in f for f in self.check(path, config=off)) is False)
        self.assertEqual([f for f in self.check(path, config=off) if "generation" in f], [])

    def test_a_changed_pool_file_or_a_missing_meta(self):
        path = self.pool()
        path.write_text("{}\n")
        self.assertTrue(any("changed after" in f for f in self.check(path)))
        self.assertEqual(self.check(self.root / "none.jsonl"), ["no pool or no meta file"])

    def test_cli_exit_codes(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ok = pool_check.main(["--config", str(self.config), "--data", str(self.data),
                                  "--pool", str(self.pool()), "--seed", "3"])
            bad = pool_check.main(["--config", str(self.config), "--data", str(self.data),
                                   "--pool", str(self.pool()), "--seed", "0"])
        self.assertEqual((ok, bad), (0, 1))


class ScriptTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash") and sys.platform != "win32", "needs a POSIX bash (on Windows, bash may be WSL)")
    def test_scripts_parse(self):
        import subprocess
        for script in ("lib.sh", "run.sh", "pilot.sh", "compare.sh", "one_step.sh"):
            subprocess.run(["bash", "-n", str(REPO / "experiments" / script)], check=True)

    @unittest.skipUnless(shutil.which("bash") and sys.platform != "win32", "needs a POSIX bash (on Windows, bash may be WSL)")
    def test_the_train_stage_extends_a_run_to_the_config_s_larger_rounds(self):
        import os
        import subprocess
        name = "graph_unaudited"
        rounds = load_config(REPO / "experiments" / name / "config.json")["rounds"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calls = tmp / "calls.txt"
            # The Python of lib.sh: runs its -c snippets, records every other command and succeeds.
            python = tmp / "python"
            python.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then exec "%s" "$@"; fi\necho "$*" >> "%s"\n'
                              % (sys.executable, calls))
            python.chmod(0o755)
            for ran, extends in ((rounds - 1, True), (rounds, False)):
                write_json(tmp / "experiments" / name / "out" / "experiment.json",
                           {"study_id": settings(name)["STUDY"], "rounds": ran})
                calls.write_text("")
                result = subprocess.run(["bash", "-c", "source experiments/lib.sh && load_experiment %s && stage_train" % name],
                                        cwd=REPO, env=dict(os.environ, PY=str(python), RSI_ROOT=str(tmp)),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                train = [c for c in calls.read_text().splitlines() if "run_iterative_experiment.py" in c]
                self.assertEqual(len(train), 1, calls.read_text())
                self.assertIn("--resume", train[0])
                self.assertEqual("--extend-rounds" in train[0], extends, train[0])

    @unittest.skipUnless(shutil.which("bash") and sys.platform != "win32", "needs a POSIX bash (on Windows, bash may be WSL)")
    def test_an_unknown_experiment_or_stage_is_refused(self):
        import subprocess
        run = REPO / "experiments" / "run.sh"
        missing = subprocess.run(["bash", str(run), "no_such_experiment"], capture_output=True, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("No experiment", missing.stderr)


if __name__ == "__main__":
    unittest.main()
