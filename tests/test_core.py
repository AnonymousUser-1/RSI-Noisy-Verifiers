import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_data import generate
from rsi.auditing import audit_pool
from rsi.common import DEFAULTS, digest, read_json, read_jsonl, rng_for, verify_dataset, write_json
from rsi.tasks import arithmetic_value, judge, make_instance, reference_answer, stratum
from rsi.verifiers import one_per_prompt, select_controlled
from simulate import trajectory


class CorrectnessTests(unittest.TestCase):
    def test_arithmetic_exact_and_no_execution(self):
        self.assertEqual(str(arithmetic_value("(1 / 3) + (2 / 3)")), "1")
        for malicious in ["__import__('os').getcwd()", "x", "2**999999", "[1]", "True"]:
            with self.assertRaises((ValueError, SyntaxError)):
                arithmetic_value(malicious)
        task = {"task": "arithmetic", "expression": "(2 + 3) * 4"}
        self.assertTrue(judge(task, "20")["correct"])
        self.assertEqual(judge(task, "14")["error"], "ignore_parentheses")
        self.assertFalse(judge(task, "20 and 14")["correct"])

    def test_graph_accepts_alternative_shortest_path(self):
        task = {"task": "graph", "nodes": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "d"], ["a", "c"], ["c", "d"]],
                "source": "a", "target": "d"}
        self.assertTrue(judge(task, '["a","b","d"]')["correct"])
        self.assertTrue(judge(task, '["a","c","d"]')["correct"])
        self.assertEqual(judge(task, '["a","b","a","c","d"]')["error"], "nonshortest")
        self.assertEqual(judge(task, '["a","d"]')["error"], "nonedge")

    def test_graph_prompt_is_neighbour_list_with_worked_example(self):
        from rsi.tasks import GRAPH_EXAMPLE, GRAPH_EXAMPLE_REPLY
        self.assertTrue(judge(GRAPH_EXAMPLE, GRAPH_EXAMPLE_REPLY)["correct"])
        rng = rng_for("graph", "train_001")
        task = make_instance("graph", rng, "train_001", 1)
        prompt = task["prompt"]
        self.assertIn("Reply: " + GRAPH_EXAMPLE_REPLY + "\n", prompt)
        problem = prompt.split("Now solve\n", 1)[1]
        lines = problem.splitlines()
        self.assertEqual(lines[0], "Neighbours of each node:")
        self.assertEqual(lines[-1], "From %s to %s." % (task["source"], task["target"]))
        listed = {}
        for line in lines[1:-1]:
            node, neighbours = line.split(": ")
            listed[node] = set(neighbours.split(", "))
        expected = {n: set() for n in task["nodes"]}
        for a, b in task["edges"]:
            expected[a].add(b)
            expected[b].add(a)
        self.assertEqual(listed, expected)

    def test_arithmetic_prompt_asks_for_steps_and_a_final_answer_line(self):
        from rsi.tasks import ARITHMETIC_EXAMPLE, ARITHMETIC_EXAMPLE_STEPS, arithmetic_value
        rng = rng_for("arithmetic", "train_001")
        task = make_instance("arithmetic", rng, "train_001", 2)
        prompt = task["prompt"]
        self.assertTrue(prompt.endswith("Expression: " + task["expression"]))
        example_reply = "\n".join(ARITHMETIC_EXAMPLE_STEPS) + "\nAnswer: %s" % arithmetic_value(ARITHMETIC_EXAMPLE)
        self.assertIn("Reply:\n" + example_reply + "\n\n", prompt)
        # The worked example is right step by step, and the judge accepts its reply.
        for step in ARITHMETIC_EXAMPLE_STEPS:
            left, right = step.split(" = ")
            self.assertEqual(arithmetic_value(left), arithmetic_value(right), step)
        self.assertEqual(arithmetic_value(ARITHMETIC_EXAMPLE_STEPS[-1].split(" = ")[1]), arithmetic_value(ARITHMETIC_EXAMPLE))
        self.assertTrue(judge({"task": "arithmetic", "expression": ARITHMETIC_EXAMPLE}, example_reply)["correct"])
        # The instruction (before the example) names no number for the model to copy.
        self.assertIsNone(re.search(r"\d", prompt.split("Example\n", 1)[0]))
        self.assertNotIn("show your work", prompt)
        # S's target error must stay possible: the prompt does not tell the model to mind parentheses.
        self.assertNotIn("parenthes", prompt.lower())

    def test_arithmetic_judge_reads_the_final_line(self):
        task = {"task": "arithmetic", "expression": "(-2 - (19 - 24))"}   # 3; without brackets -45
        verdicts = {
            "3": "correct",
            "Answer: 3": "correct",
            "19 - 24 = -5\n-2 - -5 = 3\nAnswer: 3": "correct",
            "19 - 24 = -5\n-2 - -5 = 3\n\nAnswer: 3\n": "correct",
            "**Answer:** 3": "correct",
            "**Answer: 3**": "correct",
            "answer: 3.": "correct",
            "steps\nAnswer: -45": "ignore_parentheses",
            "Answer: -3": "sign",
            "Answer: -50": "other",
            "Answer: 3\nI hope this helps.": "format",
            "-2 - (19 - 24) = 3": "format",
            "The answer is 3": "format",
            "Answer: 2*3": "format",
            "Answer:": "format",
            "": "format",
            "Answer: 6/2": "correct",
        }
        for response, error in verdicts.items():
            with self.subTest(response=response):
                self.assertEqual(judge(task, response)["error"], error)

    def test_all_generated_references_are_correct(self):
        for kind in ["graph", "arithmetic"]:
            for split in ["train_001", "eval_ood"]:
                rng = rng_for(kind, split)
                for i in range(90):
                    task = make_instance(kind, rng, split, i % 3)
                    self.assertTrue(judge(task, reference_answer(task))["correct"])

    def test_disjoint_reproducible_splits_and_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            for out in (a, b):
                generate(out, "arithmetic", 7, 2, 20, 10, 10, 10, 10)
            self.assertEqual(read_json(a/"manifest.json"), read_json(b/"manifest.json"))
            ids = [r["id"] for p in a.glob("*.jsonl") for r in read_jsonl(p)]
            self.assertEqual(len(ids), len(set(ids)))
            verify_dataset(a)
            (a/"dev.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                verify_dataset(a)


def fixture():
    tasks, candidates = {}, []
    for g in range(3):
        for i in range(30):
            tid = str((g, i))
            task = {"id": tid, "task": "arithmetic", "difficulty": str(g),
                    "expression": "(2+3)*4", "prompt": "Compute (2+3)*4"}
            tasks[tid] = task
            for sample, response in enumerate(["20", "14", "-20", "bad"]):
                candidates.append({"id": digest([tid, sample]), "task_id": tid, "sample": sample, "response": response})
    return tasks, candidates


class SelectionTests(unittest.TestCase):
    def test_quotas_and_identity(self):
        tasks, candidates = fixture()
        outputs = []
        for kind in ["iid", "persistent", "rotating"]:
            rows = select_controlled(candidates, tasks, {"kind": kind, "tpr": .9, "fpr": .2}, 5, 1)
            good_ids = set()
            for g in range(3):
                retained = [r for r in rows if r["accepted"] and tasks[r["task_id"]]["difficulty"] == str(g)]
                good = [r for r in retained if judge(tasks[r["task_id"]], r["response"])["correct"]]
                self.assertEqual(len(good), 27)
                self.assertEqual(len(retained)-len(good), 18)
                good_ids.update(r["id"] for r in good)
            outputs.append(good_ids)
            self.assertTrue(all("correct" not in r and "error" not in r for r in rows))
            if kind == "persistent":
                wrong = [r for r in rows if r["accepted"] and r["response"] != "20"]
                self.assertTrue(all(r["response"] == "14" for r in wrong))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_one_answer_per_prompt(self):
        tasks, candidates = fixture()
        pool = one_per_prompt([dict(r, accepted=True) for r in candidates], 0, 1)
        self.assertEqual(len(pool), len(tasks))
        self.assertEqual(len({r["task_id"] for r in pool}), len(pool))

    def test_audit_never_queries_more_than_budget(self):
        tasks, candidates = fixture()
        pool = one_per_prompt([dict(r, accepted=True) for r in candidates], 0, 1)
        for policy in ["uniform", "balanced", "adaptive"]:
            with patch("rsi.auditing.judge", wraps=judge) as oracle:
                retained, report = audit_pool(pool, tasks, {"policy": policy, "budget": 16, "weighting": True}, 0, 1)
                self.assertEqual(oracle.call_count, 16)
            self.assertEqual(report["queries"], 16)
            self.assertEqual(len({r["candidate_id"] for r in report["labels"]}), 16)
            self.assertTrue(all(0 <= r["weight"] <= 1 for r in retained))
            wrong = {r["candidate_id"] for r in report["labels"] if not r["correct"]}
            self.assertTrue(wrong.isdisjoint(r["id"] for r in retained))

    def test_zero_audit_uses_no_oracle(self):
        tasks, candidates = fixture()
        pool = one_per_prompt([dict(r, accepted=True) for r in candidates], 0, 1)
        with patch("rsi.auditing.judge", side_effect=AssertionError("Oracle leak")):
            retained, report = audit_pool(pool, tasks, {"policy": "none", "budget": 0, "weighting": True}, 0, 1)
        self.assertEqual(len(pool), len(retained))

    def test_simulator_boundaries(self):
        self.assertGreater(trajectory(.5, .9, .1, 1000, 1, .5, 1)[-1]["population_accuracy"], .5)
        self.assertLess(trajectory(.5, .1, .9, 1000, 1, .5, 1)[-1]["population_accuracy"], .5)
        self.assertEqual(trajectory(.5, 0, 0, 1000, 5, .5, 1)[-1]["accuracy"], .5)
        self.assertAlmostEqual(trajectory(.5, .9, .9, 1000, 5, .5, 1)[-1]["population_accuracy"], .5)


class LossTests(unittest.TestCase):
    def test_masking_and_microbatch_weighting(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed; run this test on the GPU environment too")
        from rsi.backends import response_losses
        logits = torch.randn(3, 5, 7, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 1, 4, -100, -100], [-100, -100, -100, 2, 5]])
        weights = torch.tensor([.1, 1.0, .4])
        loss = (response_losses(logits, labels)*weights).sum()/weights.sum()
        loss.backward()
        whole = logits.grad.clone()
        logits.grad.zero_()
        for i in range(3):
            (response_losses(logits[i:i+1], labels[i:i+1])*weights[i]/weights.sum()).sum().backward()
        self.assertTrue(torch.allclose(whole, logits.grad, atol=1e-6))
        # Last position cannot predict a supervised next token.
        self.assertEqual(float(logits.grad[:, -1].abs().sum()), 0)


class IntegrationTests(unittest.TestCase):
    def test_mock_resume_and_evaluation(self):
        from rsi.experiment import run
        from evaluate import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate(root/"data", "graph", 9, 2, 18, 6, 8, 6, 6)
            cfg = copy.deepcopy(DEFAULTS)
            cfg.update(backend="mock", rounds=2)
            cfg["training"]["examples"] = 8
            cfg["audit"].update(policy="adaptive", budget=4)
            run(cfg, root/"data", root/"run")
            before = read_json(root/"run/round_002/complete.json")
            run(cfg, root/"data", root/"run", resume=True)
            self.assertEqual(before, read_json(root/"run/round_002/complete.json"))
            result = evaluate(root/"run", full_draws=2, full_rounds=[0, 2])
            self.assertEqual(len(result), 6)
            self.assertTrue(all(r["DEMO_ONLY"] for r in result))
            attempt = root/"run"/before["attempt"]
            for row in read_jsonl(attempt/"training.jsonl"):
                self.assertEqual(set(row), {"prompt", "response", "weight"})
            different = copy.deepcopy(cfg)
            different["seed"] = 1
            with self.assertRaises(ValueError):
                run(different, root/"data", root/"run", resume=True)

    def test_learned_calibration_mock_pipeline(self):
        from calibrate_verifier import calibrate
        from rsi.experiment import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate(root/"data", "graph", 3, 1, 12, 6, 20, 6, 6)
            cfg = copy.deepcopy(DEFAULTS)
            cfg.update(backend="mock", rounds=1)
            cfg["verifier"]["kind"] = "llm_self"
            cfg["training"]["examples"] = 8
            cal = calibrate(cfg, root/"data", root/"cal.json")
            self.assertEqual(cal["trusted_queries"], 80)
            cfg["verifier"]["calibration"] = str(root/"cal.json")
            run(cfg, root/"data", root/"run")
            self.assertEqual(read_json(root/"run/finished.json")["calibration_queries"], 80)


if __name__ == "__main__":
    unittest.main()
