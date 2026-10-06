"""run_matched_experiment.py on the mock backend: the R, S and null arms per block.

The fixture pools are built here, because real mock pools never match:
MockBackend's errors are "[]" (one token) and a two-step detour, so no task
has a target and a non-target error at one length.  Each fixture task has a
correct path, a nonshortest walk, an endpoint error as long as the walk, and
"[]".  Candidate ids are digest([task_id, sample]) as MockBackend writes them,
so every block holds the same ids; the blocks differ in which response each id
carries, so one id has different token counts and correctness in different
blocks.

Expected values are computed here from each block's own pool rows with
rsi.tasks.judge and rsi.matching.construct_matched_subsets, the judge and
matching library the entry calls.  What these tests check is the entry's
wiring around them -- which rows reach which block, at which K, in which order,
bound to which hashes -- not judge or matching themselves.
"""
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import one_step
import rsi.matching
import run_matched_experiment
from generate_data import generate
from rsi.common import (digest, load_config, read_json, read_jsonl, seed_for, source_hash, verify_dataset,
                        write_json, write_jsonl)
from rsi.experiment import pin_config, sample_pool
from rsi.matching import candidate_from_row, construct_matched_subsets, quotas
from rsi.provenance import source_files
from rsi.tasks import judge, shortest_path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "controlled.json"
SEED = 0
STUDY = "md-mock-test-v01"
ARMS = ("R", "S", "null")
FIXTURE_NOTE = "test fixture, not sampled by any backend"
# b00 and b01 alone match at K=32.  b02 has only 6 tasks with a target and a
# non-target error at one length, so it fails K=32 (E=8) and the joint K is 16.
BLOCKS = {"b00": {"detour": 1, "shift": 0, "eligible": None},
          "b01": {"detour": 2, "shift": 1, "eligible": None},
          "b02": {"detour": 1, "shift": 2, "eligible": 6}}
# b00 kept; b01 and b02 changed, with the joint K still 16.
VARIANT = {"b00": BLOCKS["b00"],
           "b01": {"detour": 3, "shift": 3, "eligible": None},
           "b02": {"detour": 2, "shift": 1, "eligible": 5}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_rows(tasks, detour, shift, eligible):
    rows = []
    for index, task in enumerate(tasks):
        path = shortest_path(task)
        walk = [path[0]] + [path[1], path[0]] * detour + path[1:]
        options = [json.dumps(path), json.dumps(walk), json.dumps(walk[:-1] + [path[0]]), "[]"]
        if eligible is not None and index >= eligible:
            options = [options[0], "[]", "[]", "[]"]
        for sample in range(4):
            response = options[(sample + index + shift) % 4]
            rows.append({"id": digest([task["id"], sample]), "task_id": task["id"], "sample": sample,
                         "response": response, "completion_tokens": len(response.split()),
                         "prompt_tokens": len(task["prompt"].split()), "truncated": False})
    return rows


def write_pool(directory, name, rows):
    path = Path(directory) / (name + ".jsonl")
    write_jsonl(path, rows)
    write_json(str(path) + ".meta.json", {"fingerprint": "fixture:" + digest(rows), "hash": sha256(path),
                                          "backend": "mock", "fixture": FIXTURE_NOTE})
    return path


def write_pools(directory, tasks, specs):
    for name, spec in specs.items():
        write_pool(directory, name, block_rows(tasks, **spec))
    return Path(directory)


def run_entry(data, pools, out, **overrides):
    """main() with the CLI's arguments; returns (exit code, stdout, stderr or the refusal)."""
    args = argparse.Namespace(config=str(CONFIG), data=str(data), shared_pool=str(pools), out=str(out),
                              seed=SEED, study_id=None, phase=None, backend="mock",
                              shared_adapter=None, reference_gradient=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = run_matched_experiment.main(args)
        except SystemExit as exit:
            code = exit.code
    message = stderr.getvalue()
    if isinstance(code, str):
        code, message = 1, code
    return code, stdout.getvalue(), message


def expected_subsets(tasks, rows, k):
    by_task = {t["id"]: t for t in tasks}
    candidates = [candidate_from_row(r, by_task[r["task_id"]], judge(by_task[r["task_id"]], r["response"]),
                                     r["completion_tokens"]) for r in rows]
    return construct_matched_subsets(candidates, k, SEED)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # Resolved, as the entry resolves --out, so a path comparison cannot pass on a spelling difference.
        cls.root = Path(cls.tmp.name).resolve()
        cls.data = cls.root / "data"
        generate(cls.data, "graph", 7, 1, 40, 4, 4, 8, 8)
        cls.tasks = read_jsonl(cls.data / "train_001.jsonl")
        cls.by_task = {t["id"]: t for t in cls.tasks}
        cls.pools = write_pools(cls.root / "pools", cls.tasks, BLOCKS)
        cls.rows = {name: read_jsonl(cls.pools / (name + ".jsonl")) for name in BLOCKS}
        cls.out = cls.root / "out"
        cls.code, cls.stdout, cls.message = run_entry(cls.data, cls.pools, cls.out, study_id=STUDY, phase="pilot")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scratch(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def load(self, path):
        """read_json, with a missing output a test failure that names it, not an error."""
        self.assertTrue(Path(path).is_file(), "missing %s" % path)
        return read_json(path)

    def load_rows(self, path):
        """read_jsonl, with a missing output a test failure that names it, not an error."""
        self.assertTrue(Path(path).is_file(), "missing %s" % path)
        return read_jsonl(path)

    def copy_pools(self, names=None):
        target = self.scratch() / "pools"
        target.mkdir()
        for name in names or BLOCKS:
            for suffix in (".jsonl", ".jsonl.meta.json"):
                shutil.copyfile(self.pools / (name + suffix), target / (name + suffix))
        return target


class FixtureTests(Fixture):
    def test_the_configured_ratio_and_tolerance_reach_the_matching(self):
        directory = self.scratch()
        raw = read_json(CONFIG)
        raw["matching"] = {"error_fraction": 0.125, "token_tolerance": 3}
        config = directory / "matching.json"
        write_json(config, raw)
        out = directory / "out"
        code, _, message = run_entry(self.data, self.copy_pools(), out, config=str(config), study_id=STUDY, phase="pilot")
        self.assertEqual(code, 0, message)
        experiment = self.load(out / "experiment.json")
        self.assertEqual(experiment["matching_settings"], {"error_fraction": 0.125, "token_tolerance": 3})
        c, e = quotas(experiment["K"], 0.125)
        matched = self.load(out / "matching" / "matched_subsets.json")
        self.assertEqual((matched["error_fraction"], matched["token_tolerance"]), (0.125, 3))
        for record in matched["per_block"].values():
            self.assertEqual((record["certificate"]["C"], record["certificate"]["E"]), (c, e))
            self.assertEqual(record["audit"]["token_length_tolerance"], 3)

    def test_fixture_blocks_share_ids_not_rows(self):
        """Precondition: the same ids carry different token counts and correctness in b00 and b01."""
        b00 = {r["id"]: r for r in self.rows["b00"]}
        b01 = {r["id"]: r for r in self.rows["b01"]}
        self.assertEqual(set(b00), set(b01))
        self.assertTrue(any(b00[i]["completion_tokens"] != b01[i]["completion_tokens"] for i in b00))
        correct = {name: {r["id"]: judge(self.by_task[r["task_id"]], r["response"])["correct"] for r in rows}
                   for name, rows in self.rows.items()}
        self.assertTrue(any(correct["b00"][i] != correct["b01"][i] for i in b00))

    def test_fixture_needs_the_joint_ladder(self):
        """Precondition: b00 and b01 alone would match at K=32, b02 cannot."""
        self.assertTrue(expected_subsets(self.tasks, self.rows["b00"], 32)["feasible"])
        self.assertTrue(expected_subsets(self.tasks, self.rows["b01"], 32)["feasible"])
        self.assertFalse(expected_subsets(self.tasks, self.rows["b02"], 32)["feasible"])
        self.assertTrue(expected_subsets(self.tasks, self.rows["b02"], 16)["feasible"])


class MatchedRunTests(Fixture):
    def setUp(self):
        self.assertEqual(self.code, 0, self.message)

    def test_tree(self):
        top = sorted(p.name for p in self.out.iterdir())
        expected_top = ["b00", "b01", "b02", "experiment.json", "matching"]
        self.assertEqual([n for n in top if n != "source_snapshot"], expected_top)
        self.assertEqual(sorted(p.name for p in (self.out / "matching").iterdir()),
                         ["ladder.json", "matched_subsets.json"])
        for block in BLOCKS:
            self.assertEqual(sorted(p.name for p in (self.out / block).iterdir()), ["R", "S", "diagnostics", "null"])
            for arm in ARMS:
                arm_dir = self.out / block / arm
                self.assertEqual(sorted(p.name for p in arm_dir.iterdir()), ["finished.json", "round_001", "run.json"])
                self.assertEqual(sorted(p.name for p in (arm_dir / "round_001").iterdir()),
                                 ["arm.json", "complete.json", "training.jsonl"])
        experiment = self.load(self.out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["exit_code"]), ("complete", 0))
        self.assertEqual(experiment["completed"], ["%s/%s" % (b, a) for b in BLOCKS for a in ARMS])
        self.assertTrue(experiment["DEMO_ONLY"])

    def test_joint_k(self):
        ladder = self.load(self.out / "matching" / "ladder.json")
        self.assertEqual([(a["K"], a["feasible"]) for a in ladder["attempts"]], [(64, False), (32, False), (16, True)])
        self.assertEqual(self.load(self.out / "experiment.json")["K"], 16)
        for block in BLOCKS:
            for arm in ARMS:
                self.assertEqual(len(self.load_rows(self.out / block / arm / "round_001" / "training.jsonl")), 16)

    def test_each_block_is_matched_and_trained_from_its_own_pool(self):
        subsets = self.load(self.out / "matching" / "matched_subsets.json")["per_block"]
        for block in BLOCKS:
            expected = expected_subsets(self.tasks, self.rows[block], 16)
            self.assertEqual((subsets[block]["R"], subsets[block]["S"]), (expected["R"], expected["S"]), block)
            pool = {r["id"]: r for r in self.rows[block]}
            for arm, key in (("R", "R"), ("S", "S"), ("null", "R")):
                rows = [{"id": i, "task_id": pool[i]["task_id"], "prompt": self.by_task[pool[i]["task_id"]]["prompt"],
                         "response": pool[i]["response"], "weight": 1.0, "tokens": pool[i]["completion_tokens"],
                         "block_id": block}
                        for i in sorted(expected[key], key=lambda i: (pool[i]["task_id"], i))]
                self.assertEqual(self.load_rows(self.out / block / arm / "round_001" / "training.jsonl"), rows,
                                 "%s/%s" % (block, arm))

    def test_null_arm_reads_r_and_takes_no_step(self):
        for block in BLOCKS:
            r_rows = self.load_rows(self.out / block / "R" / "round_001" / "training.jsonl")
            self.assertEqual(self.load_rows(self.out / block / "null" / "round_001" / "training.jsonl"), r_rows)
            arm = self.load(self.out / block / "null" / "round_001" / "arm.json")
            self.assertEqual((arm["branch"], arm["updates"]), ("null", 0))
            self.assertEqual(self.load(self.out / block / "null" / "run.json")["training_rows"]["subset"], "R")
            for name in ARMS:
                self.assertIsNone(self.load(self.out / block / name / "round_001" / "complete.json")["adapter"])

    def test_arms_share_one_task_order(self):
        subsets = self.load(self.out / "matching" / "matched_subsets.json")["per_block"]
        # Precondition: in id order, R and S put some block's tasks in different orders.
        id_order_differs = []
        for block in BLOCKS:
            task_of = {r["id"]: r["task_id"] for r in self.rows[block]}
            id_order_differs.append([task_of[i] for i in sorted(subsets[block]["R"])]
                                    != [task_of[i] for i in sorted(subsets[block]["S"])])
        self.assertTrue(any(id_order_differs))
        for block in BLOCKS:
            orders = [[r["task_id"] for r in self.load_rows(self.out / block / arm / "round_001" / "training.jsonl")]
                      for arm in ARMS]
            self.assertEqual(orders[0], sorted(orders[0]), block)
            self.assertEqual(orders[0], orders[1], block)
            self.assertEqual(orders[0], orders[2], block)

    def test_mock_records_skips_not_numbers(self):
        for block in BLOCKS:
            lines = self.load_rows(self.out / block / "diagnostics" / "gradients.jsonl")
            self.assertEqual([(l["block_id"], l["branch"]) for l in lines], [(block, a) for a in ARMS])
            for line, arm in zip(lines, ARMS):
                self.assertEqual(set(line), {"block_id", "branch", "run_id", "step", "status", "reason", "arm_record"})
                self.assertEqual(line["status"], "skipped")
                self.assertTrue(line["reason"])
                self.assertEqual(line["arm_record"], "%s/%s/round_001/arm.json" % (block, arm))
                record = self.load(self.out / line["arm_record"])
                self.assertEqual((record["diagnostics"], record["trained"]), ("skipped", False))
                self.assertFalse({"mean_loss", "parameter_hash", "steps"} & set(record))
                self.assertFalse(list((self.out / block / arm / "round_001").glob("diagnostics*")))
                self.assertFalse((self.out / block / arm / "round_001" / "adapter").exists())

    def test_identity(self):
        experiment = self.load(self.out / "experiment.json")
        self.assertEqual((experiment["study_id"], experiment["phase"], experiment["identity_missing"]),
                         (STUDY, "pilot", {}))
        self.assertEqual((experiment["block_id"], experiment["branch"], experiment["run_id"]),
                         ("shared", "shared", STUDY + "--shared--shared"))
        for block in BLOCKS:
            lines = self.load_rows(self.out / block / "diagnostics" / "gradients.jsonl")
            for arm, line in zip(ARMS, lines):
                run = self.load(self.out / block / arm / "run.json")
                self.assertEqual((run["study_id"], run["phase"], run["block_id"], run["branch"], run["run_id"]),
                                 (STUDY, "pilot", block, arm, "%s--%s--%s" % (STUDY, block, arm)))
                self.assertEqual(run["schema_version"], "rsi-run-cost/1")
                self.assertEqual(line["run_id"], run["run_id"])

    def test_bindings(self):
        experiment = self.load(self.out / "experiment.json")
        code = experiment["code"]
        self.assertEqual(code["source_hash"], source_hash())
        if shutil.which("git"):
            head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
            if head.returncode == 0:
                self.assertEqual(code["git_commit"], head.stdout.strip())
                status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all",
                                         "--", ":(glob)*.py", ":(glob)rsi/**/*.py", ":(glob)tests/**/*.py"],
                                        capture_output=True, text=True, check=True).stdout
                self.assertEqual(code["git_dirty"], bool(status.strip()))
        if code["git_commit"] is None:
            self.assertTrue(code["git_reason"])
        resolved = pin_config(load_config(CONFIG, SEED, "mock"))
        dataset = verify_dataset(self.data)
        matching = experiment["matching"]
        self.assertEqual(matching["ladder_sha256"], sha256(self.out / "matching" / "ladder.json"))
        self.assertEqual(matching["subsets_sha256"], sha256(self.out / "matching" / "matched_subsets.json"))
        subsets = self.load(self.out / "matching" / "matched_subsets.json")["per_block"]
        for block in BLOCKS:
            pool = experiment["pools"][block]
            self.assertEqual(pool["pool_id"], sha256(self.pools / (block + ".jsonl")))
            self.assertEqual(pool["meta_sha256"], sha256(self.pools / (block + ".jsonl.meta.json")))
            for arm in ARMS:
                run = self.load(self.out / block / arm / "run.json")
                bindings = run["bindings"]
                self.assertEqual(bindings["git_commit"], code["git_commit"])
                self.assertEqual(bindings["source_hash"], code["source_hash"])
                self.assertEqual(bindings["git_dirty"], code["git_dirty"])
                self.assertEqual(bindings["config_hash"], digest(resolved))
                self.assertEqual(bindings["config_file"]["sha256"], sha256(CONFIG))
                self.assertEqual(bindings["data"]["manifest_sha256"], sha256(self.data / "manifest.json"))
                self.assertEqual(bindings["pool"], pool)
                self.assertEqual(bindings["matching"]["block_hash"], subsets[block]["hash"])
                self.assertEqual(bindings["matching"]["subsets_sha256"], matching["subsets_sha256"])
                self.assertEqual(bindings["token_counts"]["source"], "completion_tokens")
                for name in ("shared_adapter", "reference_gradient"):
                    self.assertEqual(bindings[name]["status"], "not_used")
                training = run["training_rows"]
                self.assertEqual(training["path"], "%s/%s/round_001/training.jsonl" % (block, arm))
                self.assertEqual(training["sha256"], sha256(self.out / training["path"]))
                # What evaluate.py reads from an arm directory.
                self.assertEqual(run["config"], resolved)
                self.assertEqual(run["dataset_hash"], digest(dataset))
                self.assertEqual(Path(run["data_path"]), self.data.resolve())
                self.assertIsNone(run["calibration"])
                self.assertTrue(run["DEMO_ONLY"])
                self.assertEqual(self.load(self.out / block / arm / "finished.json")["rounds"], 1)

    def test_paths_under_out_are_relative(self):
        text = "\n".join(p.read_text() for p in self.out.rglob("*.json") if "source_snapshot" not in p.parts)
        text += "\n".join(p.read_text() for p in self.out.rglob("*.jsonl"))
        self.assertNotIn(json.dumps(str(self.out))[1:-1], text)
        self.assertNotIn(self.out.as_posix(), text)


class IndependenceTests(Fixture):
    def test_a_block_is_unchanged_when_the_other_blocks_change(self):
        pools = write_pools(self.scratch() / "pools", self.tasks, VARIANT)
        out = self.scratch() / "out"
        code, _, message = run_entry(self.data, pools, out, study_id=STUDY, phase="pilot")
        self.assertEqual((self.code, code), (0, 0), message)
        self.assertEqual(self.load(out / "experiment.json")["K"], self.load(self.out / "experiment.json")["K"])
        first = self.load(self.out / "matching" / "matched_subsets.json")["per_block"]
        second = self.load(out / "matching" / "matched_subsets.json")["per_block"]
        self.assertNotEqual(first["b01"]["hash"], second["b01"]["hash"])  # precondition: the change mattered
        self.assertEqual(first["b00"], second["b00"])
        for arm in ARMS:
            self.assertEqual(self.load_rows(out / "b00" / arm / "round_001" / "training.jsonl"),
                             self.load_rows(self.out / "b00" / arm / "round_001" / "training.jsonl"), arm)


class SeedTests(Fixture):
    def test_recorded_seeds_are_the_ones_matching_drew_from(self):
        # Each block's pool_key, computed as the entry computes it on mock.  The pools differ, so
        # every block draws from its own streams.
        keys = {name: rsi.matching.pool_key([candidate_from_row(r, self.by_task[r["task_id"]],
                                                                judge(self.by_task[r["task_id"]], r["response"]),
                                                                r["completion_tokens"]) for r in rows])
                for name, rows in self.rows.items()}
        self.assertEqual(len(set(keys.values())), len(BLOCKS))
        for seed in (0, 5):
            drawn = {}
            original = rsi.matching.rng_for

            def spy(*parts):
                # (seed, pool_key, label): one block's draws are kept apart from another's.
                drawn.setdefault(parts[1:], set()).add(seed_for(*parts))
                return original(*parts)

            out = self.scratch() / "out"
            with mock.patch("rsi.matching.rng_for", spy):
                code, _, message = run_entry(self.data, self.pools, out, seed=seed)
            self.assertEqual(code, 0, message)
            experiment = self.load(out / "experiment.json")
            recorded = experiment["seed_derivation"]
            self.assertEqual(recorded["cli_seed"], seed)
            per_block = recorded["match_preflight"]["per_block"]
            self.assertEqual(sorted(per_block), sorted(BLOCKS))
            self.assertEqual({key for key, _ in drawn}, set(keys.values()))
            for block in BLOCKS:
                self.assertEqual(per_block[block]["pool_key"], keys[block], block)
                for label, field in (("matched_subset_task_order", "task_order"),
                                     ("matched_subset_assignment", "assignment"),
                                     ("matched_subset_selection", "selection")):
                    self.assertEqual(drawn[keys[block], label], {per_block[block][field]}, (block, label))
            self.assertEqual(recorded["update_with_diagnostics"]["value"], seed)
            for block in BLOCKS:
                for arm in ARMS:
                    run = self.load(out / block / arm / "run.json")
                    self.assertEqual(run["seeds"]["match_preflight"], dict(seed=seed, **per_block[block]))
                    self.assertEqual(run["seeds"]["update_with_diagnostics"], seed)
                    self.assertEqual(self.load(out / block / arm / "round_001" / "arm.json")["seed"], seed)


class IdentityTests(Fixture):
    def test_missing_identity_is_null_with_reasons(self):
        out = self.scratch() / "out"
        code, _, message = run_entry(self.data, self.pools, out)
        self.assertEqual(code, 0, message)
        experiment = self.load(out / "experiment.json")
        self.assertEqual((experiment["study_id"], experiment["phase"], experiment["run_id"]), (None, None, None))
        self.assertEqual(set(experiment["identity_missing"]), {"study_id", "phase", "run_id"})
        run = self.load(out / "b00" / "R" / "run.json")
        self.assertEqual((run["study_id"], run["phase"], run["run_id"]), (None, None, None))
        self.assertTrue(all(run["identity_missing"].values()))

    def test_bad_identity_is_refused(self):
        for overrides in ({"study_id": ""}, {"study_id": " padded"}, {"phase": "final"}):
            out = self.scratch() / "out"
            code, _, message = run_entry(self.data, self.pools, out, **overrides)
            self.assertEqual(code, 1, overrides)
            self.assertIn("Nothing was written", message)
            self.assertFalse(out.exists())


class RefusalTests(Fixture):
    def assertRefused(self, pools, fragment, **overrides):
        out = self.scratch() / "out"
        code, _, message = run_entry(self.data, pools, out, **overrides)
        self.assertEqual(code, 1, message)
        self.assertIn(fragment, message)
        self.assertFalse(out.exists())

    def test_hf_without_its_inputs_is_refused_before_anything_is_written(self):
        # The hf path itself is tested in test_run_matched_experiment_hf.py.
        identity = {"study_id": STUDY, "phase": "pilot"}
        self.assertRefused(self.pools, "--backend hf needs --shared-adapter", backend="hf", **identity)
        self.assertRefused(self.pools, "--backend hf needs --shared-adapter", backend=None, **identity)  # controlled.json: hf
        self.assertRefused(self.pools, "--backend hf needs --study-id and --phase", backend="hf")

    def test_bad_pools_are_refused(self):
        pools = self.copy_pools()
        with open(pools / "b01.jsonl", "a") as stream:
            stream.write("\n")
        self.assertRefused(pools, "changed after its meta was written")

        pools = self.copy_pools()
        meta = self.load(pools / "b02.jsonl.meta.json")
        write_json(pools / "b02.jsonl.meta.json", dict(meta, backend="hf"))
        self.assertRefused(pools, "was sampled with backend 'hf'")

        pools = self.copy_pools()
        (pools / "b00.jsonl.meta.json").unlink()
        self.assertRefused(pools, "has no b00.jsonl.meta.json")

        pools = self.copy_pools(["b00"])
        rows = self.rows["b01"][:4] + [dict(self.rows["b01"][4], task_id="no-such-task")]
        write_pool(pools, "b01", rows)
        self.assertRefused(pools, "not in train_001")

        pools = self.copy_pools(["b00"])
        write_pool(pools, "b01", self.rows["b01"][:4] + self.rows["b01"][:1])
        self.assertRefused(pools, "repeats a candidate id")

        empty = self.scratch() / "empty"
        empty.mkdir()
        self.assertRefused(empty, "holds no <block>.jsonl pools")
        self.assertRefused(self.scratch() / "missing", "is not a directory")

    def test_block_ids_that_cannot_name_a_directory_are_refused(self):
        for name, fragment in (("Matching", "is a name this entry writes"),
                               ("source_snapshot", "is a name this entry writes"),
                               ("b0--1", "single '-'"), ("b 0", "single '-'"), ("b0.x", "single '-'")):
            pools = self.copy_pools(["b00"])
            write_pool(pools, name, self.rows["b01"])
            self.assertRefused(pools, fragment)

    def test_out_that_is_not_empty_is_never_overwritten(self):
        out = self.scratch() / "out"
        out.mkdir()
        (out / "keep.txt").write_text("keep")
        code, _, message = run_entry(self.data, self.pools, out, study_id=STUDY, phase="pilot")
        self.assertEqual(code, 1)
        self.assertIn("is not empty", message)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["keep.txt"])

        before = {p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        code, _, message = run_entry(self.data, self.pools, self.out, study_id="md-mock-other-v01", phase="main")
        self.assertEqual(code, 1)
        self.assertIn("study_id %r, this run %r" % (STUDY, "md-mock-other-v01"), message)
        self.assertIn("phase 'pilot', this run 'main'", message)
        self.assertIn("Nothing was overwritten", message)
        code, _, message = run_entry(self.data, self.pools, self.out, study_id=STUDY, phase="pilot")
        self.assertEqual(code, 1)
        self.assertIn("does not resume", message)
        self.assertNotIn("another study", message)
        self.assertEqual({p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}, before)


class BlockIdTests(Fixture):
    def test_mock_records_the_names_it_is_given(self):
        pools = self.scratch() / "pools"
        pools.mkdir()
        write_pool(pools, "alpha", self.rows["b00"])
        write_pool(pools, "beta", self.rows["b01"])
        out = self.scratch() / "out"
        code, _, message = run_entry(self.data, pools, out, study_id=STUDY, phase="pilot")
        self.assertEqual(code, 0, message)
        experiment = self.load(out / "experiment.json")
        self.assertEqual(experiment["blocks"], ["alpha", "beta"])
        self.assertIs(experiment["block_id_source"]["conforming"], False)
        self.assertEqual(self.load(out / "alpha" / "S" / "run.json")["run_id"], STUDY + "--alpha--S")
        self.assertEqual(self.load(out / "experiment.json")["K"], 32)

    def test_hf_requires_exactly_b00_to_b02_or_b00_to_b04(self):
        paths, source = run_matched_experiment.find_blocks(self.pools, "hf")
        self.assertEqual((sorted(paths), source["conforming"]), (["b00", "b01", "b02"], True))
        five = self.copy_pools()
        for extra in ("b03", "b04"):
            write_pool(five, extra, self.rows["b00"])
        paths, source = run_matched_experiment.find_blocks(five, "hf")
        self.assertEqual((sorted(paths), source["conforming"]), (["b00", "b01", "b02", "b03", "b04"], True))
        for names in (["b00", "b01"], ["b00", "b01", "b02", "b03"], ["b00", "b01", "b02", "b03", "b04", "b05"]):
            pools = self.copy_pools([n for n in names if n in BLOCKS])
            for extra in set(names) - set(BLOCKS):
                write_pool(pools, extra, self.rows["b00"])
            with self.assertRaises(SystemExit) as refused:
                run_matched_experiment.find_blocks(pools, "hf")
            self.assertIn("--backend hf needs exactly the pools b00.jsonl, b01.jsonl, b02.jsonl or b00.jsonl",
                          str(refused.exception))
            self.assertIs(run_matched_experiment.find_blocks(pools, "mock")[1]["conforming"], False)


class InfeasibleTests(Fixture):
    def test_real_mock_pools_stop_before_training(self):
        pools = self.scratch() / "pools"
        config = load_config(CONFIG, SEED, "mock")
        for seed, name in enumerate(BLOCKS):
            sample_pool(config, self.tasks, pools / (name + ".jsonl"), seed)
        out = self.scratch() / "out"
        code, _, message = run_entry(self.data, pools, out)
        self.assertEqual(code, 2)
        self.assertIn("INFEASIBLE", message)
        experiment = self.load(out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["exit_code"], experiment["K"]), ("infeasible", 2, None))
        self.assertEqual(sorted(p.name for p in (out / "matching").iterdir()), ["ladder.json"])
        self.assertFalse([p for p in out.iterdir() if p.name in BLOCKS])


class SourceSnapshotTests(Fixture):
    def run_with_git(self, dirty):
        state = {"git_commit": "f" * 40, "git_dirty": dirty, "changes": [] if dirty is False else None,
                 "reason": None}
        out = self.scratch() / "out"
        with mock.patch.object(run_matched_experiment, "git_state", return_value=state):
            code, _, message = run_entry(self.data, self.pools, out)
        self.assertEqual(code, 0, message)
        return out, self.load(out / "experiment.json")["code"]

    def test_clean_tree_has_no_snapshot(self):
        out, code = self.run_with_git(False)
        self.assertIsNone(code["source_snapshot"])
        self.assertFalse((out / "source_snapshot").exists())

    def test_dirty_or_unknown_tree_keeps_a_snapshot(self):
        for dirty in (True, None):
            out, code = self.run_with_git(dirty)
            self.assertEqual(code["git_dirty"], dirty)
            self.assertEqual(code["source_snapshot"], {"path": "source_snapshot", "source_hash": source_hash()})
            snapshot = out / "source_snapshot"
            listed = sorted(p.relative_to(ROOT).as_posix() for p in source_files())
            self.assertEqual(sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()),
                             listed)
            for name in listed:
                self.assertEqual((snapshot / name).read_bytes(), (ROOT / name).read_bytes(), name)

    def test_a_snapshot_that_is_not_the_source_fails_the_run(self):
        out = self.scratch() / "out"
        state = {"git_commit": "f" * 40, "git_dirty": True, "changes": ["x.py"], "reason": None}
        with mock.patch.object(run_matched_experiment, "git_state", return_value=state), \
                mock.patch.object(run_matched_experiment, "snapshot_source", return_value="0" * 64):
            with self.assertRaises(RuntimeError):
                run_entry(self.data, self.pools, out)
        experiment = self.load(out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["exit_code"], experiment["completed"]), ("failed", 1, []))
        self.assertIn("source changed", experiment["failure_reason"])


class FailureTests(Fixture):
    def test_a_failed_arm_is_recorded_not_left_running(self):
        out = self.scratch() / "out"
        original = one_step.run_arm

        def fail_on_s(branch, *args, **kwargs):
            if branch == "S":
                raise RuntimeError("arm S failed")
            return original(branch, *args, **kwargs)

        with mock.patch.object(one_step, "run_arm", fail_on_s):
            with self.assertRaises(RuntimeError):
                run_entry(self.data, self.pools, out)
        experiment = self.load(out / "experiment.json")
        self.assertEqual((experiment["status"], experiment["exit_code"]), ("failed", 1))
        self.assertEqual(experiment["completed"], ["b00/R"])
        self.assertEqual(experiment["failure_reason"], "RuntimeError: arm S failed")


class CommandLineTests(unittest.TestCase):
    def test_help(self):
        result = subprocess.run([sys.executable, str(ROOT / "run_matched_experiment.py"), "--help"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--config", "--data", "--shared-pool", "--out", "--seed", "--study-id", "--phase", "--backend",
                     "--shared-adapter", "--reference-gradient"):
            self.assertIn(flag, result.stdout)


if __name__ == "__main__":
    unittest.main()
