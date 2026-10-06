"""experiments/one_step.sh, one_step_check.py, one_step_figures.py, one_step_cost.py and rsi/cost.py.

one_step.sh derives the one-step config from an unaudited experiment's config.json (rounds 1, the
one-step learning rate, the same base, sampling, LoRA and matching), so the experiment's pools and
shared adapter check out against it; one_step_check.py checks a finished run and its evaluation and
writes its tables; one_step_figures.py and one_step_cost.py draw the figures and the cost table.
"""
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_multiround import adapter_hash
from experiments import import_pools, one_step_check, one_step_cost, one_step_figures
from generate_data import generate
from rsi import cost
from rsi.common import digest, file_hash, load_config, require_explicit, verify_dataset, write_json, write_jsonl

REPO = Path(__file__).resolve().parents[1]
UNAUDITED = ("graph_unaudited", "graph_unaudited_qwen3-4b", "graph_unaudited_llama3.2-3b",
             "arithmetic_unaudited_llama3.2-3b")
POSIX_BASH = shutil.which("bash") and sys.platform != "win32"


def bash(script, tmp, **env):
    """script in bash, with RSI_ROOT=tmp and PY this Python unless env sets them."""
    return subprocess.run(["bash", "-c", script], cwd=REPO, capture_output=True, text=True,
                          env=dict(os.environ, **dict({"RSI_ROOT": str(tmp), "PY": sys.executable}, **env)))


@unittest.skipUnless(POSIX_BASH, "needs a POSIX bash (on Windows, bash may be WSL)")
class OneStepConfigTests(unittest.TestCase):
    def write(self, name, tmp, **env):
        """one_step.sh with no stage after the config: writes one_step_config.json and returns it."""
        done = bash("bash experiments/one_step.sh %s data_none_" % name, tmp, **env)
        path = Path(tmp) / "experiments" / name / "one_step_config.json"
        return done, path

    def test_the_one_step_config_is_the_experiment_s_with_one_step_training(self):
        for name in UNAUDITED:
            with tempfile.TemporaryDirectory() as tmp:
                done, path = self.write(name, tmp)
                self.assertIn("Unknown stage", done.stderr)        # the config is written before the stages
                ours = json.loads(path.read_text())
                theirs = json.loads((REPO / "experiments" / name / "config.json").read_text())
                self.assertEqual(ours["rounds"], 1)
                self.assertEqual(ours["training"]["learning_rate"], 5e-5)
                for key in ("model", "revision", "dtype", "device", "generation", "matching", "audit"):
                    self.assertEqual(ours[key], theirs[key], (name, key))
                for key in ("lora_rank", "lora_alpha", "lora_dropout", "target_modules"):
                    self.assertEqual(ours["training"][key], theirs["training"][key], (name, key))
                load_config(path)
                for stage in ("one_step", "reference"):
                    require_explicit(path, stage)

    def test_the_rate_can_be_set_and_a_changed_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path = self.write("graph_unaudited", tmp, ONE_STEP_LR="2e-4")
            self.assertEqual(json.loads(path.read_text())["training"]["learning_rate"], 2e-4)
            done, _ = self.write("graph_unaudited", tmp)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("differs from", done.stderr)

    def test_an_audited_experiment_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            done, path = self.write("graph_audited", tmp)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("audited experiment", done.stderr)
            self.assertFalse(path.exists())


def fake_run(root, blocks, multiround_same=True, null_moved=False, evaluated=("eval_id", "eval_ood"),
             skip=(), questions=4):
    """A finished one-step run's files, as run_matched_experiment.py and evaluate_multiround.py write them,
    and a multi-round run's matching.  The splits in `evaluated` hold every checkpoint's result except the
    names in `skip` (e.g. "b01_S_round_001"); the base's Pass@1 is 0.25, R's 0.3 and S's 0.35 + 0.01 per
    block index."""
    run, multi = Path(root) / "out_one_step", Path(root) / "out"
    per_block = {}
    for i, block in enumerate(blocks):
        ids = {"R": ["%s-r%d" % (block, j) for j in range(4)], "S": ["%s-s%d" % (block, j) for j in range(4)]}
        per_block[block] = dict(ids, certificate={"K": 4, "C": 3, "E": 1, "N_plus": 30, "N_minus": 10,
                                                  "truncated_excluded": 0})
        for arm in ("R", "S", "null"):
            rd = run / block / arm / "round_001"
            steps = 0 if arm == "null" else 1
            if arm != "null":
                write_json(rd / "adapter" / "mock.json", {"arm": arm, "block": block})
            write_json(rd / "complete.json", {"adapter": None if arm == "null" else "round_001/adapter",
                                              "training": {"steps": steps, "mean_loss": 0.1}})
            write_json(run / block / arm / "run.json", {"config": {"model": "Qwen/Qwen3-1.7B"}})
            norm = 0.5 if (arm == "null" and null_moved) else (0.0 if arm == "null" else 0.06)
            projection = 0.0 if arm == "null" else (-0.01 if arm == "R" else -0.008) * (1 + i)
            write_json(rd / "diagnostics.json", {
                "delta_theta": {"norm": norm}, "main_diagnostic": {"h_T_delta_theta": projection},
                "clipping": {"fired": False},
                "validation": {"composition_valid": True, "relative_error": 1e-7},
                "per_sample": [{"id": "%s-c%d" % (block, j), "correct": j < 3, "norm": 0.01 * (j + 1),
                                "loss": 0.5} for j in range(4)]})
            write_jsonl(rd / "training.jsonl", [{"task_id": "%s-t%d" % (block, j)} for j in range(3)] + [{"task_id": "shared"}])
    write_json(run / "matching" / "matched_subsets.json", {"per_block": per_block})
    write_json(run / "experiment.json", {"status": "complete", "task": "graph"})
    theirs = {b: dict(v) for b, v in per_block.items()}
    if not multiround_same:
        theirs[blocks[0]] = dict(theirs[blocks[0]], S=["other"])
    write_json(multi / "matching" / "matched_subsets.json", {"per_block": theirs})
    for split in evaluated:
        directory = run / ("evaluation_greedy_" + split)
        write_json(directory / "protocol.json", {"split": split, "questions": questions})
        results = {"round_000": (0.25, None)}
        for i, block in enumerate(blocks):
            results["%s_R_round_001" % block] = (0.3, (block, "R"))
            results["%s_S_round_001" % block] = (0.35 + 0.01 * i, (block, "S"))
        for name, (value, arm) in results.items():
            if name in skip:
                continue
            adapter = None if arm is None else adapter_hash(run / arm[0] / arm[1] / "round_001" / "adapter")
            write_json(directory / (name + ".json"), {"pass1": value, "questions": questions, "adapter_hash": adapter})
    return run, multi


@unittest.skipUnless(POSIX_BASH, "needs a POSIX bash (on Windows, bash may be WSL)")
class OneStepGateTests(unittest.TestCase):
    """A one-step run that fails its checks is refused by train, evaluate and figures, also when its
    experiment.json says complete; a run is never started without all of its inputs."""

    def test_nothing_runs_without_the_round1_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "experiments" / "graph_unaudited"
            write_json(tmp / "data" / "graph" / "manifest.json", {})
            write_json(work / "shared_adapter" / "shared_adapter.json", {})
            for seed in range(5):
                write_jsonl(work / "pools" / ("b%02d.jsonl" % seed), [])
                write_json(work / "pools" / ("b%02d.jsonl.meta.json" % seed), {})
            done = bash("bash experiments/one_step.sh graph_unaudited reference", tmp)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("out/matching/matched_subsets.json", done.stderr)
            self.assertFalse((work / "reference_one_step").exists())

    def test_a_complete_run_that_fails_its_checks_is_never_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, _ = fake_run(tmp / "experiments" / "graph_unaudited", ["b00", "b01", "b02", "b03", "b04"],
                              multiround_same=False, null_moved=True)
            (run / "experiment.json").write_text(json.dumps({"status": "complete"}, indent=2) + "\n")
            calls = tmp / "calls.txt"
            python = tmp / "python"
            python.write_text("\n".join([
                "#!/bin/bash",
                'case "$1" in experiments/pool_check.py) exit 0;; '
                'experiments/one_step_check.py|-|-c) exec "%s" "$@";; esac' % sys.executable,
                'echo "$*" >> "%s"' % calls, ""]))
            python.chmod(0o755)
            for stage in ("train", "evaluate", "figures"):
                done = bash("bash experiments/one_step.sh graph_unaudited %s" % stage, tmp, PY=str(python))
                self.assertNotEqual(done.returncode, 0, stage)
                self.assertIn("fails its checks", done.stderr, stage)
            self.assertFalse(calls.exists() and calls.read_text().strip(), "something ran after a failed check")

    def test_figures_need_a_complete_evaluation_and_done_means_every_stage_succeeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "experiments" / "graph_unaudited"
            run, multi = fake_run(work, BLOCKS5, skip=("b03_R_round_001",))
            calls = tmp / "calls.txt"
            python = tmp / "python"
            python.write_text("\n".join([
                "#!/bin/bash",
                'case "$1" in experiments/one_step_check.py|-|-c) exec "%s" "$@";; esac' % sys.executable,
                'echo "$*" >> "%s"' % calls, ""]))
            python.chmod(0o755)
            done = bash("bash experiments/one_step.sh graph_unaudited figures", tmp, PY=str(python))
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("evaluation of", done.stderr)
            self.assertIn("not complete", done.stderr)
            self.assertNotIn("DONE", done.stdout)
            self.assertFalse(calls.exists() and calls.read_text().strip(), "something was drawn")
            self.assertFalse(json.loads((run / "checks.json").read_text())["passed"])
            self.assertIn("b03_R_round_001", (run / "checks.json").read_text())

            fake_run(work, BLOCKS5)                         # the missing checkpoint evaluated
            done = bash("bash experiments/one_step.sh graph_unaudited figures", tmp, PY=str(python))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("DONE graph_unaudited one-step, every stage succeeded: figures", done.stdout)
            drawn = calls.read_text()
            for script in ("plot_multiround.py", "experiments/one_step_figures.py", "experiments/one_step_cost.py"):
                self.assertIn(script, drawn)

    def test_a_failing_figure_command_stops_before_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake_run(tmp / "experiments" / "graph_unaudited", BLOCKS5)
            python = tmp / "python"
            python.write_text("\n".join([
                "#!/bin/bash",
                'case "$1" in experiments/one_step_check.py|-|-c) exec "%s" "$@";; esac' % sys.executable,
                'case "$2" in experiments/one_step_figures.py) echo "no display" >&2; exit 3;; esac',
                "exit 0", ""]))
            python.chmod(0o755)
            done = bash("bash experiments/one_step.sh graph_unaudited figures", tmp, PY=str(python))
            self.assertNotEqual(done.returncode, 0)
            self.assertNotIn("DONE", done.stdout)


class OneStepCheckTests(unittest.TestCase):
    def test_a_good_run_passes_and_its_table_has_every_block_and_arm(self):
        blocks = ["b00", "b01", "b02", "b03", "b04"]
        with tempfile.TemporaryDirectory() as tmp:
            run, multi = fake_run(tmp, blocks)
            ref = multi / "matching" / "matched_subsets.json"
            self.assertTrue(all(ok for ok, _ in one_step_check.check(run, ref)))
            rows = one_step_check.table(run)
            self.assertEqual([(r["block"], r["arm"]) for r in rows], [(b, a) for b in blocks for a in ("R", "S", "null")])
            first = {r["arm"]: r for r in rows if r["block"] == "b00"}
            self.assertEqual((first["R"]["TPR"], first["R"]["FPR"], first["R"]["precision"]), (0.1, 0.1, 0.75))
            self.assertEqual(first["null"]["pass1_eval_id"], 0.25)      # null is round 0
            self.assertEqual(first["R"]["pass1_eval_id"], 0.3)
            self.assertAlmostEqual(first["R"]["error_eval_id"], 0.7)
            self.assertAlmostEqual(first["R"]["delta_error_eval_id"], -0.05)   # 0.70 - 0.75: R lowered the error
            self.assertAlmostEqual(first["null"]["delta_error_eval_id"], 0.0)

    def test_failures_are_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, multi = fake_run(tmp, ["b00", "b01", "b02"], multiround_same=False, null_moved=True)
            ref = multi / "matching" / "matched_subsets.json"
            failed = [text for ok, text in one_step_check.check(run, ref) if not ok]
            self.assertIn("b00 R/S subsets equal the reference round-1 matching", failed)
            self.assertTrue(any("b00/null |Delta theta|" in t for t in failed), failed)

    def test_a_run_without_the_reference_matching_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = fake_run(tmp, ["b00", "b01", "b02"])
            failed = [text for ok, text in one_step_check.check(run) if not ok]
            self.assertEqual(len(failed), 1, failed)
            self.assertIn("no reference round-1 matching", failed[0])
            record = Path(tmp) / "checks.json"
            self.assertEqual(one_step_check.main(["--run", str(run), "--record", str(record)]), 1)
            self.assertFalse(json.loads(record.read_text())["passed"])


BLOCKS5 = ["b00", "b01", "b02", "b03", "b04"]


def evaluation_json(root, splits=("eval_id", "eval_ood")):
    path = Path(root) / "evaluation.json"
    write_json(path, {"splits": list(splits), "batch_size": 32})
    return path


class EvaluationCompletenessTests(unittest.TestCase):
    """Check 5: nothing is drawn from an evaluation that lacks a checkpoint, a split, a question or
    was taken on another adapter."""

    def failed(self, run, multi, splits=("eval_id", "eval_ood")):
        found = one_step_check.check(run, multi / "matching" / "matched_subsets.json")
        found += one_step_check.check_evaluation(run, list(splits))
        return [text for ok, text in found if not ok]

    def test_a_complete_evaluation_passes_with_11_checkpoints_per_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, multi = fake_run(tmp, BLOCKS5)
            self.assertEqual(self.failed(run, multi), [])
            texts = [t for _, t in one_step_check.check_evaluation(run, ["eval_id", "eval_ood"])]
            self.assertEqual(texts, ["eval_id: 11 of 11 checkpoints evaluated", "eval_ood: 11 of 11 checkpoints evaluated"])

    def test_a_missing_checkpoint_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, multi = fake_run(tmp, BLOCKS5, skip=("b01_S_round_001", "round_000"))
            failed = self.failed(run, multi)
            self.assertEqual(len(failed), 2, failed)          # both splits, each naming both checkpoints
            self.assertIn("9 of 11", failed[0])
            self.assertIn("round_000", failed[0])
            self.assertIn("b01_S_round_001", failed[0])

    def test_a_split_not_evaluated_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, multi = fake_run(tmp, BLOCKS5, evaluated=("eval_id",))
            failed = self.failed(run, multi)
            self.assertEqual(len(failed), 1, failed)
            self.assertIn("eval_ood: not evaluated", failed[0])

    def test_a_result_on_fewer_questions_or_another_adapter_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, multi = fake_run(tmp, BLOCKS5)
            path = run / "evaluation_greedy_eval_id" / "b02_R_round_001.json"
            write_json(path, dict(json.loads(path.read_text()), questions=3))
            write_json(run / "b03" / "S" / "round_001" / "adapter" / "mock.json", {"retrained": True})
            failed = self.failed(run, multi)
            self.assertTrue(any("b02_R_round_001.json answered 3 of 4 questions" in t for t in failed), failed)
            self.assertTrue(any("b03_S_round_001.json was evaluated on another adapter" in t for t in failed), failed)

    def test_the_tables_are_written_only_from_a_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, multi = fake_run(tmp, BLOCKS5, skip=("b04_S_round_001",))
            args = ["--run", str(run), "--reference", str(multi / "matching" / "matched_subsets.json"),
                    "--evaluation", str(evaluation_json(tmp)), "--record", str(tmp / "checks.json"),
                    "--table", str(tmp / "table.csv"), "--table15", str(tmp / "table15.csv")]
            self.assertEqual(one_step_check.main(args), 1)
            self.assertFalse((tmp / "table.csv").exists() or (tmp / "table15.csv").exists())
            record = json.loads((tmp / "checks.json").read_text())
            self.assertEqual((record["passed"], record["evaluation_checked"]), (False, True))


class Table15Tests(unittest.TestCase):
    """The paper's one-step results table: per block and arm, ID/OOD error, the ID error change against
    null, h^T Delta theta and |Delta theta|_2, for all five blocks."""

    def test_rows_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, multi = fake_run(tmp, BLOCKS5)
            args = ["--run", str(run), "--reference", str(multi / "matching" / "matched_subsets.json"),
                    "--evaluation", str(evaluation_json(tmp)), "--table15", str(tmp / "table15.csv")]
            self.assertEqual(one_step_check.main(args), 0)
            with open(tmp / "table15.csv") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), ["task", "model", "block", "arm", "id_error", "ood_error",
                                             "delta_id_error", "h_T_delta_theta", "delta_theta_l2"])
            self.assertEqual([(r["block"], r["arm"]) for r in rows], [(b, a) for b in BLOCKS5 for a in ("R", "S", "null")])
            b01 = {r["arm"]: r for r in rows if r["block"] == "b01"}
            self.assertEqual((b01["S"]["task"], b01["S"]["model"]), ("graph", "Qwen/Qwen3-1.7B"))
            self.assertAlmostEqual(float(b01["S"]["id_error"]), 0.64)        # 1 - 0.36
            self.assertAlmostEqual(float(b01["S"]["delta_id_error"]), -0.11)  # 0.64 - 0.75
            self.assertAlmostEqual(float(b01["null"]["delta_id_error"]), 0.0)
            self.assertAlmostEqual(float(b01["R"]["h_T_delta_theta"]), -0.02)
            self.assertEqual(float(b01["null"]["delta_theta_l2"]), 0.0)


class OneStepFiguresTests(unittest.TestCase):
    def test_every_figure_as_pdf_png_and_its_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = fake_run(tmp, BLOCKS5)
            prefix = Path(tmp) / "figures" / "graph_unaudited_one_step"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(one_step_figures.main(["--run", str(run), "--label", "graph_unaudited",
                                                        "--out", str(prefix)]), 0)
            names = ("projection", "displacement", "gradients", "error_eval_id", "error_eval_ood")
            for name in names:
                for ext in ("pdf", "png", "json"):
                    self.assertTrue(Path("%s_%s.%s" % (prefix, name, ext)).is_file(), (name, ext))
            self.assertTrue(Path("%s_projection.pdf" % prefix).read_bytes().startswith(b"%PDF"))
            projection = json.loads(Path("%s_projection.json" % prefix).read_text())
            self.assertAlmostEqual(projection["per_block"]["b02"]["S"], -0.024)
            paired = projection["paired"]["s_minus_r"]
            self.assertEqual(paired["n"], 5)
            self.assertAlmostEqual(paired["mean"], 0.006)                  # S - R = 0.002 x (1 + i), mean over 5
            error = json.loads(Path("%s_error_eval_id.json" % prefix).read_text())
            self.assertAlmostEqual(error["per_block"]["b00"]["R"], -0.05)
            gradients = json.loads(Path("%s_gradients.json" % prefix).read_text())["groups"]
            self.assertEqual(len(gradients["R_correct"]), 15)              # 3 correct per block x 5 blocks
            self.assertEqual(len(gradients["S_error"]), 5)


class CostTests(unittest.TestCase):
    """rsi/cost.py records and experiments/one_step_cost.py's table."""

    def test_a_record_per_attempt_failed_ones_kept_and_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "costs"
            with cost.stage(records, "reference", "shared") as attempt:
                attempt.generated(0, 0)
            with cost.stage(records, "train", "R", "b00"):
                pass                                                         # generation not measured
            with self.assertRaises(RuntimeError):
                with cost.stage(records, "train", "S", "b00"):
                    raise RuntimeError("out of memory")
            found = cost.read_records(records)
            self.assertEqual(len(found), 3)
            by = {(r["stage"], r["owner"]): r for r in found.values()}
            self.assertEqual(by["train", "S"]["status"], "failed")
            self.assertIn("out of memory", by["train", "S"]["failure"])
            self.assertIsNone(by["train", "R"]["new_responses"])
            self.assertIn("new_responses", by["train", "R"]["missing"])
            self.assertEqual(by["reference", "shared"]["new_responses"], 0)
            self.assertGreaterEqual(by["reference", "shared"]["wall_seconds"], 0)
            self.assertEqual(by["reference", "shared"]["gpus"], 0)            # CPU
            self.assertEqual(by["reference", "shared"]["allocated_gpu_hours"], 0)
            self.assertIsNone(by["reference", "shared"]["memory"])
            with self.assertRaises(ValueError):
                with cost.stage(records, "train", "everyone"):
                    pass

    def test_the_table_sums_each_cost_id_once_and_labels_lower_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / "a", tmp / "b"
            rec = {"schema": cost.SCHEMA, "stage": "train", "block": "b00", "detail": {}, "failure": None,
                   "memory": None, "wall_seconds": 10.0, "gpus": 1, "allocated_gpu_hours": 10 / 3600,
                   "new_responses": 0, "new_output_tokens": 0}
            write_json(a / "x.json", dict(rec, cost_id="x", owner="R", status="complete"))
            write_json(b / "x.json", dict(rec, cost_id="x", owner="R", status="complete"))   # the same attempt twice
            write_json(a / "y.json", dict(rec, cost_id="y", owner="S", status="complete", new_responses=None))
            write_json(a / "z.json", dict(rec, cost_id="z", owner="S", status="failed", failure="killed"))
            write_json(a / "e.json", dict(rec, cost_id="e", owner="evaluation", stage="evaluate", status="complete",
                                          new_responses=3000, new_output_tokens=90000,
                                          memory={"allocated_peak_bytes": 5, "reserved_peak_bytes": 7}))
            pools = tmp / "pools"
            for s in range(5):
                write_json(pools / ("b%02d.jsonl.meta.json" % s), {"hash": "%d" % s})
            out = tmp / "cost.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(one_step_cost.main(["--records", str(a), str(b), "--pools", str(pools),
                                                     "--out", str(out)]), 0)
            with open(out) as f:
                rows = {r["cost_owner"]: r for r in csv.DictReader(f)}
            self.assertEqual(float(rows["R only"]["wall_seconds"]), 10.0)            # x counted once
            self.assertIn("observed lower bound", rows["S only"]["coverage"])        # y lacks new_responses
            self.assertIn("failed attempts not summed: 1", rows["S only"]["coverage"])
            self.assertEqual(float(rows["Null and evaluation"]["new_responses"]), 3000)
            total = rows["Deduplicated joint total"]
            self.assertEqual(float(total["wall_seconds"]), 30.0)                      # x, y, e; not z
            self.assertIn("3 unique cost_id(s)", total["coverage"])
            self.assertTrue((tmp / "cost_failed.csv").is_file())
            self.assertTrue((tmp / "cost_memory.csv").is_file())
            with open(tmp / "cost_pools.csv") as f:
                self.assertEqual(next(csv.DictReader(f))["pools"], "5")

    def test_two_different_records_under_one_cost_id_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rec = {"schema": cost.SCHEMA, "cost_id": "x", "owner": "R", "status": "complete"}
            write_json(tmp / "a" / "x.json", rec)
            write_json(tmp / "b" / "x.json", dict(rec, status="failed"))
            with self.assertRaises(ValueError):
                cost.read_records(tmp / "a", tmp / "b")


class ImportPoolsTests(unittest.TestCase):
    """experiments/import_pools.py on a frozen export shaped like the Qwen3-4B one."""

    def export(self, root, generation=None, adapter_base=None):
        name = "graph_unaudited_qwen3-4b"
        config = load_config(REPO / "experiments" / name / "config.json")
        source = Path(root) / "export"
        generate(source / "graph", "graph", 7, 1, 8, 4, 4, 4, 4)
        data_hash = digest(verify_dataset(source / "graph"))
        drawn = {k: v for k, v in config["generation"].items() if k != "repetition_stop"}   # no such field then
        for seed in range(5):
            pool = source / "pools" / ("b%02d.jsonl" % seed)
            write_jsonl(pool, [{"id": "x%d" % seed, "task_id": "t", "response": "[]", "truncated": False}])
            write_json(str(pool) + ".meta.json", {
                "hash": file_hash(pool), "backend": "hf", "model": config["model"], "revision": config["revision"],
                "config": config, "generation": generation or drawn, "seed": {"cli": seed},
                "split": {"name": "train_001"}, "data": {"dataset_hash": data_hash}})
        write_json(source / "shared_adapter" / "shared_adapter.json", {
            "base_model": adapter_base or config["model"], "base_revision": config["revision"],
            "parameter_hash": "a" * 64, "protocol": {"rank": 8, "alpha": 16, "dropout": 0.0,
                                                     "target_modules": ["q_proj", "v_proj"], "use_rslora": False}})
        write_json(source / "matching" / "matched_subsets.json", {"per_block": {}})
        return name, source

    def run_import(self, root, name, source):
        old = os.environ.get("RSI_ROOT")
        os.environ["RSI_ROOT"] = str(root)
        try:
            return import_pools.main([name, str(source)])
        finally:
            if old is None:
                os.environ.pop("RSI_ROOT")
            else:
                os.environ["RSI_ROOT"] = old

    def test_pools_drawn_before_repetition_stop_import_as_they_are(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name, source = self.export(root)
            self.assertEqual(self.run_import(root, name, source), 0)
            work = root / "experiments" / name
            for seed in range(5):
                pool = "b%02d.jsonl.meta.json" % seed
                self.assertEqual(file_hash(work / "pools" / pool), file_hash(source / "pools" / pool))   # not rewritten
            self.assertTrue((work / "round1_reference" / "matched_subsets.json").is_file())
            self.assertTrue((root / "data" / "graph-qwen3-4b-import" / "manifest.json").is_file())
            self.assertEqual(self.run_import(root, name, source), 0)                                      # again: no-op

    def test_pools_drawn_with_other_settings_or_another_adapter_base_are_refused(self):
        config = load_config(REPO / "experiments" / "graph_unaudited_qwen3-4b" / "config.json")
        cases = ({"generation": dict(config["generation"], repetition_stop={"span": 128, "max_period": 32})},
                 {"generation": dict(config["generation"], temperature=0.7)},
                 {"adapter_base": "Qwen/Qwen3-1.7B"})
        for change in cases:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                name, source = self.export(root, **change)
                with self.assertRaises(SystemExit) as refused:
                    self.run_import(root, name, source)
                self.assertIn("Not imported", str(refused.exception))
                self.assertFalse((root / "experiments" / name / "pools").exists())

    def test_only_an_experiment_that_says_so_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as refused:
                self.run_import(Path(tmp), "graph_unaudited", Path(tmp))
            self.assertIn("POOLS_IMPORTED", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
