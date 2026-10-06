"""GSM8K and DeepMind Mathematics: answer rules, judges, imports, verifier targets, mock runs.

Inline fixtures only -- no network and no raw data -- because evaluation_gpu_smoke.py runs this
suite and refuses skipped tests.  The fixture sources stand in for the pinned files by passing
their own hashes; the pins themselves are checked by import_data.py on the real files.
"""
import argparse
import copy
import importlib.util
import io
import json
import random
import sys
import tarfile
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest import mock

import import_data
import prepare_suite
import run_matched_experiment
from rsi.common import (DEFAULTS, digest, file_hash, load_config, read_json, read_jsonl, verify_dataset, write_json,
                        write_jsonl)
from rsi.external import DMMATH_ROOT, dmmath_task, gsm8k_task, stated_numbers
from rsi.iterative import select_own_subset
from rsi.matching import ARITHMETIC_TARGET_SIGNATURES, TARGET_SIGNATURES, TARGETS_BY_TASK, Candidate, targets_for
from rsi.tasks import (ERROR_SIGNATURES, TASKS, extract_answer, judge, make_instance, mock_wrong_answer,
                       number_text, reference_answer, stratum)
from rsi.verifiers import select_controlled

JANET_QUESTION = ("Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
                  "muffins for her friends every day with four. She sells the remainder at the farmers' market "
                  "daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' "
                  "market?")
JANET_SOLUTION = ("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
                  "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18")


def janet(rule="strict"):
    return gsm8k_task(JANET_QUESTION, JANET_SOLUTION, rule, {"file": "fixture", "line": 0})


def division(answer="-189/2", rule="strict", question="Calculate -378 divided by 4."):
    return dmmath_task(question, answer, "arithmetic__div", "train-easy", rule, {"file": "fixture", "pair": 0})


def gsm8k_record(i):
    a, b, c = 3 + i, 5 + 2 * i, 1 + i % 3
    return {"question": "Item %d: Ann has %d pens and buys %d more. She gives away %d. How many pens are left?"
                        % (i, a, b, c),
            "answer": "Ann has %d + %d = <<%d+%d=%d>>%d pens.\nShe gives away %d, so %d - %d = <<%d-%d=%d>>%d remain.\n"
                      "#### %d" % (a, b, a, b, a + b, a + b, c, a + b, c, a + b, c, a + b - c, a + b - c, a + b - c)}


def write_gsm8k_fixture(raw, train=40, test=12):
    raw.mkdir(parents=True)
    for name, rows in (("train.jsonl", range(train)), ("test.jsonl", range(1000, 1000 + test))):
        (raw / name).write_text("".join(json.dumps(gsm8k_record(i)) + "\n" for i in rows))
    return {"gsm8k": {"origin": "fixture", "license": "test", "files": {
        name: {"url": "fixture:" + name, "sha256": file_hash(raw / name)} for name in ("train.jsonl", "test.jsonl")}}}


def write_dmmath_fixture(raw):
    """Two modules, one with an extrapolate file; a question repeated across levels and one
    interpolate question that also appears in train, as in the real archive."""
    raw.mkdir(parents=True)
    files = {}
    for level in ("train-easy", "train-medium", "train-hard"):
        files["%s/arithmetic__add_or_sub.txt" % level] = [
            ("What is %d plus 0.5? (%s)" % (i, level), "%d.5" % i) for i in range(30)]
        files["%s/numbers__gcd.txt" % level] = [
            ("What is the gcd of %d and %d?" % (6 * (i + 1), 4 * (i + 1)), str(2 * (i + 1))) for i in range(30)]
    files["interpolate/arithmetic__add_or_sub.txt"] = [("Total of %d and -1/4?" % i, "%d/4" % (4 * i - 1))
                                                       for i in range(1, 21)]
    files["interpolate/numbers__gcd.txt"] = [("What is the gcd of 6 and 4?", "2")] + [
        ("Calculate the gcd of %d and %d." % (10 * i, 15 * i), str(5 * i)) for i in range(1, 20)]
    files["extrapolate/arithmetic__add_or_sub_big.txt"] = [("What is %d + -700000?" % (10 ** 6 + i), str(300000 + i))
                                                           for i in range(20)]
    archive = raw / "mathematics_dataset-v1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, pairs in files.items():
            data = "".join(q + "\n" + a + "\n" for q, a in pairs).encode("ascii")
            info = tarfile.TarInfo("%s/%s" % (DMMATH_ROOT, name))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return {"dmmath": {"origin": "fixture", "license": "test", "files": {
        archive.name: {"url": "fixture:" + archive.name, "sha256": file_hash(archive)}}}}


def run_import(argv, sources):
    with redirect_stdout(io.StringIO()):
        return import_data.main([str(a) for a in argv], sources=sources)


class AnswerRuleTests(unittest.TestCase):
    def test_strict_takes_only_a_bare_number(self):
        for text, value in (("18", 18), (" 18 \n", 18), ("$18", 18), ("18.", 18), ("Answer: 18", 18),
                            ("answer:18", 18), ("1,080", 1080), ("-3/4", Fraction(-3, 4)), ("0.5", Fraction(1, 2)),
                            (".5", Fraction(1, 2)), ("50%", 50), (chr(0x2212) + "7", -7), (chr(0x2013) + "7", -7),
                            (chr(0xFF0D) + "0.5", Fraction(-1, 2))):
            self.assertEqual(extract_answer(text, "strict"), value, text)
        # A dash between two numbers is a range, not a minus sign.
        self.assertEqual(extract_answer("3" + chr(0x2013) + "5 hours", "last_number"), 5)
        for text in ("The answer is 18.", "18 dollars", "eighteen", "", "1,08", "18!", "9 then 18"):
            self.assertIsNone(extract_answer(text, "strict"), text)

    def test_first_and_last_number(self):
        prose = "She has 16 eggs, sells 9, and makes $18."
        self.assertEqual(extract_answer(prose, "first_number"), 16)
        self.assertEqual(extract_answer(prose, "last_number"), 18)
        self.assertIsNone(extract_answer("no digits at all", "last_number"))
        # The step list hazard of first_number on prose answers: the '1.' of '1. First ...'.
        self.assertEqual(extract_answer("1. First, add the numbers.\n2. The total is 18.", "first_number"), 1)
        with self.assertRaises(ValueError):
            extract_answer("18", "middle_number")

    def test_fractions_only_where_the_task_allows_them(self):
        self.assertEqual(extract_answer("36/2 = 18", "first_number", fractions=True), 18)
        self.assertEqual(extract_answer("36/2 = 18", "first_number", fractions=False), 36)
        self.assertIsNone(extract_answer("3/4", "strict", fractions=False))

    def test_number_text(self):
        for value, text in ((18, "18"), (Fraction(5, 2), "2.5"), (Fraction(-3, 4), "-0.75"),
                            (Fraction(3, 20), "0.15"), (Fraction(1, 3), "1/3"), (Fraction(-1, 8), "-0.125")):
            self.assertEqual(number_text(value), text)
            self.assertEqual(Fraction(text), Fraction(value))


class Gsm8kTaskTests(unittest.TestCase):
    def test_row_fields(self):
        row = janet()
        self.assertEqual((row["task"], row["answer"], row["steps"], row["difficulty"]), ("gsm8k", "18", 2, "mixed"))
        self.assertEqual(row["intermediates"], ["9"])
        self.assertEqual(set(row["operands"]), {"2", "3", "4", "9", "16"})
        self.assertTrue(row["prompt"].endswith("Problem:\n" + JANET_QUESTION + "\n\nAnswer:"))
        self.assertEqual(stratum(row), "gsm8k:mixed")
        self.assertEqual(reference_answer(row), "18")
        self.assertEqual(row["id"], gsm8k_task("  " + JANET_QUESTION.replace(" ", "  "), JANET_SOLUTION, "last_number",
                                               {"file": "other", "line": 9})["id"])

    def test_labels(self):
        row = janet()
        for response, label in (("18", "correct"), ("$18", "correct"), ("18.0", "correct"), ("9", "intermediate"),
                                ("16", "operand"), ("4", "operand"), ("20", "other"), ("eighteen", "format"),
                                ("The answer is 18.", "format")):
            self.assertEqual(judge(row, response)["error"], label, response)
        prose = janet("first_number")
        self.assertEqual(judge(prose, "The answer is 18.")["error"], "correct")
        self.assertEqual(judge(prose, "She sells 9 eggs, so 18")["error"], "intermediate")

    def test_a_fraction_in_the_question_is_one_stated_number(self):
        """GSM8K test row 396: '1/4' must not state a 4, or Bill's 4 slices stop being an intermediate."""
        question = ("Jenny is dividing up a pizza with 12 slices. She gives 1/3 to Bill and 1/4 to Mark. "
                    "If Jenny eats 2 slices, how many slices are left?")
        solution = ("First find how many slices 1/3 of the pizza is by multiplying 1/3 by the total number of "
                    "slices: 12 slices * 1/3 = <<12*1/3=4>>4 slices\nDo the same thing to find how many slices "
                    "1/4 of the pizza is: 12 slices * 1/4 = <<12*1/4=3>>3 slices\nThen subtract the slices each "
                    "of the three people ate to find the remaining number of slices: 12 slices - 4 slices - 3 "
                    "slices - 2 slices = <<12-4-3-2=3>>3 slices\n#### 3")
        row = gsm8k_task(question, solution, "strict", {"file": "test.jsonl", "index": 396})
        self.assertEqual(row["intermediates"], ["4"])
        self.assertEqual([judge(row, x)["error"] for x in ("4", "1", "3")], ["intermediate", "operand", "correct"])

    def test_results_written_after_equals_are_intermediates(self):
        """GSM8K test row 136 has no <<a=b>> step: its sub-result 9 is written after '='."""
        question = ("There are 27 unicorns left in the world.  One third of them are in the Scottish Highlands.  "
                    "Two thirds of the Scottish unicorns are female.  How many female Scottish unicorns are there?")
        row = gsm8k_task(question, "Scottish Unicorns:27(1/3)=9\nFemale:9(2/3)=6 unicorns\n#### 6", "strict", {})
        self.assertEqual(row["intermediates"], ["9"])
        self.assertEqual(judge(row, "9")["error"], "intermediate")
        row = gsm8k_task("Tom has 5 red cars and 3 blue cars, then twice as many. How many?",
                         "Total = 5 + 3 = 8 so 8*2 = 16.\n#### 16", "strict", {})
        self.assertEqual(row["intermediates"], ["8"])  # '= 5 + 3' is an operand, not a result

    def test_compound_number_words(self):
        self.assertEqual(stated_numbers("She has twenty-five apples, forty two pears and three plums."),
                         {25, 42, 3})

    def test_commas_in_the_gold_and_malformed_solutions(self):
        row = gsm8k_task("How many?", "It is 1,000 + 80 = <<1000+80=1080>>1,080.\n#### 1,080", "strict", {})
        self.assertEqual(row["answer"], "1080")
        self.assertTrue(judge(row, "1,080")["correct"])
        for solution in ("no final line", "Text.\n#### 2.5", "Text.\n####18"):
            with self.assertRaises(ValueError, msg=solution):
                gsm8k_task("How many?", solution, "strict", {})
        with self.assertRaises(ValueError):
            gsm8k_task("How many?", "1 + 1.\n#### 2", "loose", {})


class DmmathTaskTests(unittest.TestCase):
    def test_labels(self):
        row = division()
        self.assertEqual((row["difficulty"], row["level"], row["module"]), ("mixed", "train-easy", "arithmetic__div"))
        for response, label in (("-189/2", "correct"), ("-94.5", "correct"), ("-378/4", "correct"),
                                ("189/2", "sign"), ("-945", "decimal_shift"), ("-9.45", "decimal_shift"),
                                ("-94", "other"), ("x", "format")):
            self.assertEqual(judge(row, response)["error"], label, response)
        self.assertEqual(reference_answer(row), "-189/2")

    def test_a_zero_answer_has_no_sign_or_shift(self):
        row = division("0", question="What is 0*6?")
        for response, label in (("0", "correct"), ("0.0", "correct"), ("-0", "correct"), ("1", "other")):
            self.assertEqual(judge(row, response)["error"], label, response)

    def test_only_numeric_answers(self):
        with self.assertRaises(ValueError):
            division("x**2 + 1")


class TaskRegistryTests(unittest.TestCase):
    def test_imported_and_unknown_kinds_are_not_generated(self):
        for kind in ("gsm8k", "dmmath", "svamp"):
            with self.assertRaises(ValueError, msg=kind):
                make_instance(kind, random.Random(0), "dev", 0)
        with self.assertRaises(ValueError):
            judge({"task": "svamp"}, "1")
        with self.assertRaises(ValueError):
            reference_answer({"task": "svamp"})

    def test_every_task_has_signatures_the_judge_emits(self):
        self.assertEqual(set(ERROR_SIGNATURES), set(TASKS))
        rows = {"gsm8k": janet(), "dmmath": division()}
        for kind, row in rows.items():
            labels = {judge(row, r)["error"] for r in ("18", "9", "16", "20", "x", "189/2", "-945", "-94", "-189/2")}
            self.assertLessEqual(labels - {"correct"}, set(ERROR_SIGNATURES[kind]), kind)


class VerifierAndMockTests(unittest.TestCase):
    def pool(self, row, responses):
        return [{"id": "c%d" % i, "task_id": row["id"], "response": r} for i, r in enumerate(responses)]

    def accepted_errors(self, row, kind, round_number=1):
        candidates = self.pool(row, ["18"] * 4 + ["9", "9", "16", "16", "20", "20"])
        selected = select_controlled(candidates, {row["id"]: row}, {"kind": kind, "tpr": 1.0, "fpr": 0.2},
                                     0, round_number)
        return [c["response"] for c in selected if c["accepted"] and c["response"] != "18"]

    def test_persistent_and_rotating_target_the_gsm8k_signatures(self):
        row = janet()
        self.assertEqual(self.accepted_errors(row, "persistent"), ["9"])
        self.assertEqual(self.accepted_errors(row, "rotating", 2), ["16"])
        self.assertEqual(len(self.accepted_errors(row, "iid")), 1)

    def test_mock_answers_are_wrong_and_mostly_the_target(self):
        for row in (janet(), division(), division("0", question="What is 0*6?")):
            rng = random.Random(7)
            labels = [judge(row, mock_wrong_answer(row, rng))["error"] for _ in range(400)]
            self.assertNotIn("correct", labels)
            self.assertLessEqual(set(labels), set(ERROR_SIGNATURES[row["task"]]))
            if row["answer"] != "0":
                share = labels.count(ERROR_SIGNATURES[row["task"]][0]) / len(labels)
                self.assertTrue(0.35 < share < 0.65, (row["task"], share))


class Gsm8kImportTests(unittest.TestCase):
    def test_layout_disjointness_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = write_gsm8k_fixture(root / "raw")
            argv = ["--task", "gsm8k", "--raw", root / "raw", "--rounds", 2, "--per-round", 10, "--dev", 4,
                    "--calibration", 4, "--gradient-reference", 3]
            manifest = run_import(argv + ["--out", root / "a"], sources)
            self.assertEqual(manifest["counts"], {"eval_id": 12, "train_001": 10, "train_002": 10, "dev": 4,
                                                  "calibration": 4, "gradient_reference": 3})
            self.assertNotIn("eval_ood.jsonl", manifest["files"])
            self.assertEqual(verify_dataset(root / "a"), manifest)
            self.assertEqual(manifest["source"]["files"]["train.jsonl"]["sha256"],
                             sources["gsm8k"]["files"]["train.jsonl"]["sha256"])
            rows = {name: read_jsonl(root / "a" / (name + ".jsonl")) for name in manifest["counts"]}
            ids = [r["id"] for split in rows.values() for r in split]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual({r["source"]["file"] for r in rows["eval_id"]}, {"test.jsonl"})
            for name, split in rows.items():
                for r in split:
                    self.assertEqual((r["split"], r["difficulty"], r["answer_rule"]), (name, "mixed", "strict"))
                    if name != "eval_id":
                        self.assertEqual(r["source"]["file"], "train.jsonl")
            again = run_import(argv + ["--out", root / "b"], sources)
            self.assertEqual(again["files"], manifest["files"])

    def test_disjoint_from_keeps_a_pilot_out_of_the_larger_import(self):
        def train_side(root):
            return {r["id"] for name in ("train_001", "dev", "calibration", "gradient_reference")
                    for r in read_jsonl(root / (name + ".jsonl"))}

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = write_gsm8k_fixture(root / "raw", train=60)
            base = ["--task", "gsm8k", "--raw", root / "raw", "--rounds", 1]
            run_import(base + ["--out", root / "pilot", "--per-round", 6, "--dev", 3, "--calibration", 3,
                               "--gradient-reference", 2], sources)
            larger = base + ["--per-round", 30, "--dev", 5, "--calibration", 5, "--gradient-reference", 4]
            plain = run_import(larger + ["--out", root / "plain"], sources)
            self.assertTrue(train_side(root / "pilot") & train_side(root / "plain"))  # one pool, same seed
            full = run_import(larger + ["--out", root / "full", "--disjoint-from", root / "pilot"], sources)
            self.assertFalse(train_side(root / "pilot") & train_side(root / "full"))
            self.assertEqual(full["import"]["disjoint_from"][0]["train_side_ids"], 14)
            self.assertEqual(plain["import"]["disjoint_from"], [])
            dm_sources = write_dmmath_fixture(root / "dm-raw")
            run_import(["--task", "dmmath", "--raw", root / "dm-raw", "--out", root / "dm", "--modules", "numbers__gcd",
                        "--rounds", 1, "--per-round", 6, "--dev", 2, "--calibration", 2, "--eval-id", 4,
                        "--gradient-reference", 2], dm_sources)
            with self.assertRaises(ValueError):  # another task's dataset
                run_import(larger + ["--out", root / "mixed", "--disjoint-from", root / "dm"], sources)

    def test_refusals_write_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = write_gsm8k_fixture(root / "raw")
            base = ["--task", "gsm8k", "--raw", root / "raw", "--out", root / "out", "--dev", 4,
                    "--calibration", 4, "--gradient-reference", 3]
            with self.assertRaises(ValueError):  # 2 x 30 + 11 rows from 40 train problems
                run_import(base + ["--per-round", 30], sources)
            tampered = copy.deepcopy(sources)
            tampered["gsm8k"]["files"]["test.jsonl"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                run_import(base + ["--per-round", 10], tampered)
            for extra in (["--eval-ood", 5], ["--modules", "numbers__gcd"], ["--train-pairs-per-file", 5]):
                with self.assertRaises(SystemExit, msg=extra):
                    run_import(base + ["--per-round", 10] + extra, sources)
            self.assertFalse((root / "out").exists() and any((root / "out").iterdir()))


class DmmathImportTests(unittest.TestCase):
    def test_pooled_levels_modules_in_turn_and_extrapolate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = write_dmmath_fixture(root / "raw")
            manifest = run_import(["--task", "dmmath", "--raw", root / "raw", "--out", root / "out",
                                   "--modules", "arithmetic__add_or_sub", "numbers__gcd", "--rounds", 1,
                                   "--per-round", 12, "--dev", 4, "--calibration", 4, "--eval-id", 10,
                                   "--eval-ood", 6, "--gradient-reference", 4], sources)
            rows = {name: read_jsonl(root / "out" / (name + ".jsonl")) for name in manifest["counts"]}
            ids = [r["id"] for split in rows.values() for r in split]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual([r["module"] for r in rows["eval_id"]],
                             ["arithmetic__add_or_sub", "numbers__gcd"] * 5)
            self.assertEqual({r["level"] for r in rows["eval_id"]}, {"interpolate"})
            self.assertEqual({(r["module"], r["level"], r["source"]["file"]) for r in rows["eval_ood"]},
                             {("arithmetic__add_or_sub", "extrapolate", "extrapolate/arithmetic__add_or_sub_big.txt")})
            train = [r for name in ("train_001", "dev", "calibration", "gradient_reference") for r in rows[name]]
            self.assertEqual([r["module"] for r in rows["train_001"]], ["arithmetic__add_or_sub", "numbers__gcd"] * 6)
            self.assertLessEqual({r["level"] for r in train}, {"train-easy", "train-medium", "train-hard"})
            self.assertGreater(len({r["level"] for r in train}), 1)
            self.assertEqual({r["difficulty"] for split in rows.values() for r in split}, {"mixed"})
            self.assertEqual(manifest["import"]["eval_ood_files"], {"arithmetic__add_or_sub": "arithmetic__add_or_sub_big"})
            for split in rows.values():
                for r in split:
                    self.assertTrue(judge(r, r["answer"])["correct"])

    def test_gradient_reference_size_leaves_the_other_splits_alone(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = write_dmmath_fixture(root / "raw")
            base = ["--task", "dmmath", "--raw", root / "raw", "--modules", "arithmetic__add_or_sub", "numbers__gcd",
                    "--rounds", 1, "--per-round", 12, "--dev", 4, "--calibration", 4, "--eval-id", 6, "--eval-ood", 4]
            small = run_import(base + ["--out", root / "a", "--gradient-reference", 2], sources)["files"]
            large = run_import(base + ["--out", root / "b", "--gradient-reference", 8], sources)["files"]
            self.assertNotEqual(small.pop("gradient_reference.jsonl"), large.pop("gradient_reference.jsonl"))
            self.assertEqual(small, large)

    def test_module_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = write_dmmath_fixture(root / "raw")
            base = ["--task", "dmmath", "--raw", root / "raw", "--rounds", 1, "--per-round", 6, "--dev", 2,
                    "--calibration", 2, "--eval-id", 4, "--gradient-reference", 2]
            with self.assertRaises(ValueError):  # no extrapolate file for gcd
                run_import(base + ["--out", root / "a", "--modules", "numbers__gcd", "--eval-ood", 2], sources)
            with self.assertRaises(ValueError):  # not a single-number module
                run_import(base + ["--out", root / "b", "--modules", "comparison__sort"], sources)
            manifest = run_import(base + ["--out", root / "c", "--modules", "numbers__gcd"], sources)
            self.assertNotIn("eval_ood.jsonl", manifest["files"])


class MatchedEntryTests(unittest.TestCase):
    """The matched one-step and multi-round entries take the dataset task's target error."""

    def test_targets_follow_the_task(self):
        self.assertEqual(targets_for("graph"), TARGET_SIGNATURES)
        self.assertEqual(targets_for("arithmetic"), ARITHMETIC_TARGET_SIGNATURES)
        self.assertEqual(targets_for("gsm8k"), ("intermediate",))
        self.assertEqual(targets_for("dmmath"), ("sign",))
        with self.assertRaises(ValueError):
            targets_for("svamp")
        # Every task kind has one, and it is its persistent controlled verifier's target.
        self.assertEqual(set(TARGETS_BY_TASK), set(TASKS))
        for task in TASKS:
            self.assertEqual(targets_for(task), (ERROR_SIGNATURES[task][0],), task)

    def test_later_round_selection_takes_the_task_target(self):
        pool = [Candidate("p%d-%s" % (p, kind), "p%d" % p, response, kind == "c", label, 1)
                for p in range(12)
                for kind, response, label in (("c", "18", "correct"), ("t", "9", "intermediate"), ("o", "16", "operand"))]
        chosen = select_own_subset(pool, "S", 8, 0, ("b00", 2, "S"), target_signatures=("intermediate",))
        self.assertTrue(chosen["feasible"])
        self.assertEqual([c.error for c in chosen["rows"] if not c.correct], ["intermediate"] * 2)
        self.assertEqual(chosen["audit"]["target_signatures"], ["intermediate"])
        # graph's nonshortest never occurs here: the old fixed target left S nothing to select.
        self.assertFalse(select_own_subset(pool, "S", 8, 0, ("b00", 2, "S"))["feasible"])

    def matched_run(self, root, tasks, wrong):
        """One mock block: per task the gold, a target error, a non-target error and 'unknown',
        all one word, so every task has a target and a non-target error at one length."""
        rows = []
        for task in tasks:
            target, other = wrong(task)
            for sample, response in enumerate((task["answer"], target, other, "unknown")):
                rows.append({"id": digest([task["id"], sample]), "task_id": task["id"], "sample": sample,
                             "response": response, "completion_tokens": 1, "prompt_tokens": 1})
        pools = root / "pools"
        write_jsonl(pools / "b00.jsonl", rows)
        write_json(pools / "b00.jsonl.meta.json", {"hash": file_hash(pools / "b00.jsonl"), "backend": "mock",
                                                   "fixture": "test fixture, not sampled by any backend"})
        args = argparse.Namespace(config=str(Path(import_data.__file__).parent / "configs" / "controlled.json"),
                                  data=str(root / "data"), shared_pool=str(pools), out=str(root / "out"), seed=0,
                                  study_id=None, phase=None, backend="mock", shared_adapter=None,
                                  reference_gradient=None)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = run_matched_experiment.main(args)
        self.assertEqual(code, 0)
        experiment = read_json(root / "out" / "experiment.json")
        subsets = read_json(root / "out" / "matching" / "matched_subsets.json")["per_block"]["b00"]
        by_id, by_task = {r["id"]: r for r in rows}, {t["id"]: t for t in tasks}
        labels = {arm: [judge(by_task[by_id[i]["task_id"]], by_id[i]["response"])["error"] for i in subsets[arm]]
                  for arm in ("R", "S")}
        return experiment, subsets, labels

    def test_gsm8k_and_dmmath_run_through_the_matched_entry(self):
        for kind, wrong, target in (
                ("gsm8k", lambda t: (t["intermediates"][0], next(x for x in t["operands"]
                                                               if x not in t["intermediates"])), "intermediate"),
                ("dmmath", lambda t: (number_text(-Fraction(t["answer"])), number_text(10 * Fraction(t["answer"]))),
                 "sign")):
            with self.subTest(task=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                if kind == "gsm8k":
                    sources = write_gsm8k_fixture(root / "raw", train=60)
                    argv = ["--task", "gsm8k", "--rounds", 1, "--per-round", 40]
                else:
                    sources = write_dmmath_fixture(root / "raw")
                    argv = ["--task", "dmmath", "--modules", "arithmetic__add_or_sub", "numbers__gcd", "--rounds", 1,
                            "--per-round", 40, "--eval-id", 6, "--eval-ood", 4]
                run_import(argv + ["--raw", root / "raw", "--out", root / "data", "--dev", 4, "--calibration", 4,
                                   "--gradient-reference", 2], sources)
                tasks = read_jsonl(root / "data" / "train_001.jsonl")
                experiment, subsets, labels = self.matched_run(root, tasks, wrong)
                self.assertEqual((experiment["task"], experiment["target_signatures"]), (kind, [target]))
                for name in ("ladder.json", "matched_subsets.json"):  # the matching files say which target
                    record = read_json(root / "out" / "matching" / name)
                    self.assertEqual((record["task"], record["target_signatures"]), (kind, [target]))
                errors = [label for label in labels["S"] if label != "correct"]
                self.assertEqual(errors, [target] * (experiment["K"] // 4))
                self.assertEqual(subsets["audit"]["S_target_hits"], experiment["K"] // 4)

    def test_the_pilot_report_uses_the_configured_ratio_and_tolerance(self):
        """Target errors 1 token long, the other errors 4: no prompt qualifies at tolerance 0, every one at
        3; the K = 64 criteria need E = 64 x error_fraction prompts (8 at 0.125)."""
        repo = Path(import_data.__file__).parent
        spec = importlib.util.spec_from_file_location("pilot_report", repo / "scripts" / "multiround" / "pilot_report.py")
        pilot_report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot_report)
        reports = {}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run_import(["--task", "gsm8k", "--rounds", 1, "--per-round", 20, "--raw", root / "raw", "--out",
                        root / "data", "--dev", 6, "--calibration", 4, "--gradient-reference", 2],
                       write_gsm8k_fixture(root / "raw"))
            dev = read_jsonl(root / "data" / "dev.jsonl")
            other = {t["id"]: next(x for x in t["operands"] if x not in t["intermediates"]) for t in dev}
            targets = {t["intermediates"][0] for t in dev}
            pool = [{"id": "%s-%d" % (t["id"], s), "task_id": t["id"], "response": response, "completion_tokens": 1}
                    for t in dev for s, response in enumerate((t["answer"], t["intermediates"][0], other[t["id"]]))]
            write_jsonl(root / "pool.jsonl", pool)
            raw = json.loads((repo / "configs" / "matched_pool.json").read_text())
            config = load_config(repo / "configs" / "matched_pool.json")
            write_json(str(root / "pool.jsonl") + ".meta.json", {
                "hash": file_hash(root / "pool.jsonl"), "split": {"name": "dev"},
                "data": {"dataset_hash": digest(verify_dataset(root / "data"))},
                "model": config["model"], "revision": config["revision"], "generation": config["generation"]})
            stub = types.SimpleNamespace(AutoTokenizer=mock.Mock())
            for tolerance in (0, 3):
                raw["matching"] = {"error_fraction": 0.125, "token_tolerance": tolerance}
                write_json(root / "config.json", raw)
                args = argparse.Namespace(config=str(root / "config.json"), data=str(root / "data"),
                                          pool=str(root / "pool.jsonl"), split="dev", out=str(root / "report.json"))
                with mock.patch.dict(sys.modules, {"transformers": stub}), \
                        mock.patch.object(pilot_report, "supervised_token_count",
                                          lambda tokenizer, text: 1 if text in targets else 4), \
                        redirect_stdout(io.StringIO()):
                    pilot_report.main(args)
                reports[tolerance] = read_json(root / "report.json")
        for tolerance, (eligible, go) in ((0, (0, False)), (3, (6, True))):
            report = reports[tolerance]
            self.assertEqual((report["error_fraction"], report["token_tolerance"]), (0.125, tolerance))
            self.assertEqual(report["prompts_eligible_J_E"], eligible)
            # 6 of 6 prompts x 20 round-1 prompts = 20 >= 8 (and >= 10 with the margin) at tolerance 3.
            self.assertEqual((report["go"]["K64_expected_at_round1_pool"],
                              report["go"]["K64_with_25pct_margin_at_round1_pool"]), (go, go))

    def test_the_pilot_report_counts_the_task_target(self):
        repo = Path(import_data.__file__).parent
        spec = importlib.util.spec_from_file_location("pilot_report", repo / "scripts" / "multiround" / "pilot_report.py")
        pilot_report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot_report)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run_import(["--task", "gsm8k", "--rounds", 1, "--per-round", 20, "--raw", root / "raw", "--out",
                        root / "data", "--dev", 6, "--calibration", 4, "--gradient-reference", 2],
                       write_gsm8k_fixture(root / "raw"))
            dev = read_jsonl(root / "data" / "dev.jsonl")
            other = {t["id"]: next(x for x in t["operands"] if x not in t["intermediates"]) for t in dev}
            pool = [{"id": "%s-%d" % (t["id"], s), "task_id": t["id"], "response": response, "completion_tokens": 1}
                    for t in dev for s, response in enumerate((t["answer"], t["intermediates"][0], other[t["id"]]))]
            # One answer cut at max_new_tokens: judged and counted, but it takes no matching role.
            pool.append({"id": "cut", "task_id": dev[0]["id"], "response": dev[0]["answer"], "completion_tokens": 2048,
                         "truncated": True})
            write_jsonl(root / "pool.jsonl", pool)
            config = load_config(repo / "configs" / "matched_pool.json")
            meta = {"hash": file_hash(root / "pool.jsonl"), "split": {"name": "dev"},
                    "data": {"dataset_hash": digest(verify_dataset(root / "data"))},
                    "model": config["model"], "revision": config["revision"], "generation": config["generation"]}
            write_json(str(root / "pool.jsonl") + ".meta.json", meta)
            args = argparse.Namespace(config=str(repo / "configs" / "matched_pool.json"), data=str(root / "data"),
                                      pool=str(root / "pool.jsonl"), split="dev", out=str(root / "report.json"))
            # One token per answer, so every prompt has its target and another error at one length.
            # A stub transformers module: the report loads only a tokenizer, and this suite runs without one.
            stub = types.SimpleNamespace(AutoTokenizer=mock.Mock())
            with mock.patch.dict(sys.modules, {"transformers": stub}), \
                    mock.patch.object(pilot_report, "supervised_token_count", lambda tokenizer, text: 1), \
                    redirect_stdout(io.StringIO()):
                pilot_report.main(args)
            report = read_json(root / "report.json")
            # A pool drawn before the data changed (e.g. a new prompt: task ids stay the same) is refused.
            write_json(str(root / "pool.jsonl") + ".meta.json", dict(meta, data={"dataset_hash": "f" * 64}))
            with self.assertRaises(SystemExit) as stale:
                pilot_report.main(args)
            self.assertIn("data (a prompt or the questions changed since)", str(stale.exception))
        self.assertEqual((report["error_fraction"], report["token_tolerance"]), (0.25, 0))
        self.assertEqual((report["task"], report["target_signatures"]), ("gsm8k", ["intermediate"]))
        self.assertEqual(report["signatures"], {"correct": 7, "intermediate": 6, "operand": 6})
        self.assertEqual((report["prompts_with_target"], report["prompts_eligible_J_E"]), (6, 6))
        self.assertEqual((report["truncated_answers"], report["go"]["no_truncated_answers"]), (1, False))
        self.assertEqual(report["completion_tokens"]["max"], 2048)


class PipelineTests(unittest.TestCase):
    def test_mock_loop_and_evaluation_without_eval_ood(self):
        from evaluate import evaluate
        from rsi.experiment import run
        from rsi.paired_evaluation import freeze
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = write_gsm8k_fixture(root / "raw")
            run_import(["--task", "gsm8k", "--raw", root / "raw", "--out", root / "data", "--rounds", 2,
                        "--per-round", 10, "--dev", 4, "--calibration", 4, "--gradient-reference", 3], sources)
            cfg = copy.deepcopy(DEFAULTS)
            cfg.update(backend="mock", rounds=2)
            cfg["training"]["examples"] = 8
            cfg["verifier"]["kind"] = "persistent"
            with redirect_stdout(io.StringIO()):
                run(cfg, root / "data", root / "run")
                result = evaluate(root / "run", full_draws=2, full_rounds=[0, 2])
            self.assertEqual({r["split"] for r in result}, {"eval_id"})
            self.assertEqual({r["round"] for r in result}, {0, 1, 2})
            with self.assertRaises(ValueError):  # asked for explicitly, a missing split is refused up front
                evaluate(root / "run", splits=["eval_ood"])
            done = read_json(root / "run/round_001/complete.json")
            for row in read_jsonl(root / "run" / done["attempt"] / "training.jsonl"):
                self.assertEqual(set(row), {"prompt", "response", "weight"})
            write_json(root / "config.json", dict(copy.deepcopy(DEFAULTS), backend="mock"))
            protocol = freeze(root / "config.json", root / "data", root / "protocol.json", 4, 100, 1, 50, 42)
            self.assertEqual(protocol["splits"], ["eval_id"])

    def test_pilot_suites_for_the_imported_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with redirect_stdout(io.StringIO()):
                jobs = prepare_suite.prepare("pilot", root / "suite", root / "data", root / "runs", root / "cal",
                                             tasks=["gsm8k", "dmmath"])
            self.assertEqual(len(jobs), 12)
            self.assertEqual({Path(j["argv"][j["argv"].index("--data") + 1]).name for j in jobs}, {"gsm8k", "dmmath"})


if __name__ == "__main__":
    unittest.main()
