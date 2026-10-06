"""Tests for the gradient_reference split (HANDOFF A.1, PLAN section 5).

The split holds the prompts the reference gradient h is computed on: 64 by
default, drawn like the training prompts (same task, same generator,
difficulty i % 3, the in-distribution branch of `make_instance`), sharing no
prompt with train_*, dev, calibration, eval_id or eval_ood, and read only for
diagnostics.

It is generated last.  The splits share one `seen` set, so a split generated
before another takes instances the later one then has to redraw.  Generated
last, it changes no older file, and it changes manifest.json in exactly three
places: one more `files` entry, one more `counts` entry, and `unique_instances`
up by its size.  An earlier placement would only show when a draw collides,
which at these instance-space sizes is rare, so two tests force collisions:
they make the split's random stream repeat each other split's stream in turn.

What decides the expectations: `OLD` is what generate_data.py at 015cf5e,
before the split existed, writes with `ARGS`.  Its hashes are over the content
with "\\r\\n" made "\\n": write_jsonl writes in text mode, so the raw bytes, and
the manifest's own hashes, carry the platform's line ending.  The
in-distribution checks take their ranges from rsi/tasks.py: graph 6+d..10+d
nodes (eval_ood 14..18); arithmetic 3+2d leaves (eval_ood 4 more, inside a
"-(" wrapper).
"""
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from generate_data import generate
from rsi.common import file_hash, read_json, read_jsonl, rng_for, verify_dataset, write_jsonl

REPO = Path(__file__).resolve().parents[1]
KINDS = ("graph", "arithmetic")
ARGS = (7, 2, 20, 10, 10, 10, 10)  # seed, rounds, per_round, dev, calibration, eval_id, eval_ood
OTHER = ("train_001", "train_002", "dev", "calibration", "eval_id", "eval_ood")
REFERENCE = "gradient_reference.jsonl"

OLD = {
    "arithmetic": {
        "content": {
            "calibration.jsonl": "40d970f3311e866c1734f3a6828660533d7e0ad56acd64e7dec926666a6ddab4",
            "dev.jsonl": "5709b9a346ee417c1b87741b851151cd5f76395bcf17d4b9b7bac410d9d99663",
            "eval_id.jsonl": "36d29877f58c4552942bd23a47f85049e9c3624b02e5662291540b1fb5b61ef1",
            "eval_ood.jsonl": "7c76dec7a2efdeafd59da386ce745edbf1bccdf2b20147524e021906b89545c9",
            "train_001.jsonl": "1fe5ceec463c921f56e607c6d602ed367824a2a6b3af5fc4feeab45221bd8c29",
            "train_002.jsonl": "5385c556bf50551111ad69676c2a2eb1f33f479ffb70321aef02426bfe8c7b4b",
        },
        "manifest": {"counts": {"calibration": 10, "dev": 10, "eval_id": 10, "eval_ood": 10,
                                "train_001": 20, "train_002": 20},
                     "rounds": 2, "seed": 7, "task": "arithmetic", "unique_instances": 80, "version": 1},
    },
    "graph": {
        "content": {
            "calibration.jsonl": "cded3cb81640b1ddc5d442f5f0cdafa1f433f27c760ba467227859abdbe08713",
            "dev.jsonl": "7aa0ac23eea344015cb9320f29521431ad20714b6304b7917d4e33348077c863",
            "eval_id.jsonl": "9aff9f102ba9ff2e5aaa146eb4c645d121dc8e432e406b4c8c62a297b86f1d22",
            "eval_ood.jsonl": "e256b99e1c7e5203afd41794d14e860e2715a36dd508a6fcf09fd3223452a7cc",
            "train_001.jsonl": "5f311d08abe7f2e52c457d3ae78b3a9471131e9f73230b672541fb73d97dfd9c",
            "train_002.jsonl": "bac1b228b9cc700e27300bc0c978f3ff6184fa9ceaf91eb97f462d7f31c62423",
        },
        "manifest": {"counts": {"calibration": 10, "dev": 10, "eval_id": 10, "eval_ood": 10,
                                "train_001": 20, "train_002": 20},
                     "rounds": 2, "seed": 7, "task": "graph", "unique_instances": 80, "version": 1},
    },
}


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def old_prompt_sha256(path, kind):
    """`lf_sha256` of the file as 015cf5e would have written it.

    The graph and arithmetic prompts changed after 015cf5e (a strict reply
    instruction plus a worked example, rsi/tasks.py); nothing else in a row did.
    Putting the 015cf5e prompt back and serialising as write_jsonl does must
    reproduce the 015cf5e bytes exactly, so these hashes still pin every other
    field of every row.
    """
    def old_prompt(row):
        if kind == "graph":
            return ("Find a shortest path in this undirected, unweighted graph. "
                    "Return ONLY a JSON list of node names, including both endpoints.\n"
                    + json.dumps({k: row[k] for k in ("nodes", "edges", "source", "target")}))
        return "Evaluate this expression exactly. Return ONLY the integer answer.\n" + row["expression"]

    lines = []
    for row in read_jsonl(path):
        row = dict(row, prompt=old_prompt(row))
        lines.append(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def ids(path):
    return {row["id"] for row in read_jsonl(path)}


def colliding(split):
    """An rng_for that gives the reference split `split`'s random stream."""
    def rng(seed, kind, name):
        return rng_for(seed, kind, split if name == "gradient_reference" else name)
    return rng


class GradientReferenceSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = {}
        for kind in KINDS:
            cls.data[kind] = cls.root / kind
            generate(cls.data[kind], kind, *ARGS)  # the split left at its default size

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def reference_rows(self, root):
        self.assertTrue((root / REFERENCE).exists(), "no %s in %s" % (REFERENCE, root))
        return read_jsonl(root / REFERENCE)

    def collided(self, kind, split):
        """Data generated with the reference split drawing from `split`'s stream."""
        root = self.root / ("collided_%s_%s" % (kind, split))
        if not root.exists():
            with mock.patch("generate_data.rng_for", colliding(split)):
                generate(root, kind, *ARGS)
        return root

    def run_cli(self, out, *extra):
        command = [sys.executable, "generate_data.py", "--out", str(out), "--task", "arithmetic", "--seed", "7",
                   "--rounds", "1", "--per-round", "5", "--dev", "2", "--calibration", "2", "--eval-id", "2",
                   "--eval-ood", "2", *extra]
        return subprocess.run(command, cwd=REPO, capture_output=True, text=True)

    def test_the_split_holds_64_prompts_by_default(self):
        for kind in KINDS:
            rows = self.reference_rows(self.data[kind])
            self.assertEqual(len(rows), 64, kind)
            self.assertEqual(len({row["id"] for row in rows}), 64, kind)
            self.assertEqual({row["split"] for row in rows}, {"gradient_reference"}, kind)
            self.assertEqual(read_json(self.data[kind] / "manifest.json")["counts"].get("gradient_reference"), 64)

    def test_the_split_shares_no_prompt_with_any_other_split(self):
        for kind in KINDS:
            self.assertEqual(sorted(p.stem for p in self.data[kind].glob("*.jsonl")),
                             sorted(OTHER + ("gradient_reference",)), kind)
            reference = {row["id"] for row in self.reference_rows(self.data[kind])}
            for split in OTHER:
                self.assertFalse(reference & ids(self.data[kind] / (split + ".jsonl")), (kind, split))

    def test_a_reference_draw_that_repeats_another_split_is_redrawn(self):
        for kind in KINDS:
            for split in OTHER:
                root = self.collided(kind, split)
                reference = {row["id"] for row in self.reference_rows(root)}
                self.assertEqual(len(reference), 64, (kind, split))
                for other in OTHER:
                    self.assertFalse(reference & ids(root / (other + ".jsonl")), (kind, split, other))

    def test_the_other_splits_are_the_files_015cf5e_wrote(self):
        for kind in KINDS:
            for name, expected in OLD[kind]["content"].items():
                self.assertEqual(old_prompt_sha256(self.data[kind] / name, kind), expected, (kind, name))

    def test_the_other_splits_do_not_move_when_the_reference_split_collides_with_them(self):
        for kind in KINDS:
            for split in OTHER:
                root = self.collided(kind, split)
                for name, expected in OLD[kind]["content"].items():
                    self.assertEqual(old_prompt_sha256(root / name, kind), expected, (kind, split, name))

    def test_the_manifest_changes_in_exactly_three_places(self):
        for kind in KINDS:
            root = self.data[kind]
            self.assertTrue((root / REFERENCE).exists(), kind)
            expected = copy.deepcopy(OLD[kind]["manifest"])
            # The recorded hashes are of the raw bytes, which carry the platform's
            # line ending, so they come from the files themselves; the test above
            # pins the older files' content.
            expected["files"] = {name: file_hash(root / name) for name in OLD[kind]["content"]}
            expected["files"][REFERENCE] = file_hash(root / REFERENCE)
            expected["counts"]["gradient_reference"] = 64
            expected["unique_instances"] += 64
            self.assertEqual(read_json(root / "manifest.json"), expected, kind)

    def test_a_changed_reference_file_fails_the_dataset_check(self):
        root = self.root / "changed"
        generate(root, "arithmetic", *ARGS)
        verify_dataset(root)
        write_jsonl(root / REFERENCE, self.reference_rows(root)[1:])
        with self.assertRaises(ValueError):
            verify_dataset(root)

    def test_the_split_is_drawn_like_the_training_prompts(self):
        for kind in KINDS:
            rows = self.reference_rows(self.data[kind])
            self.assertEqual({row["task"] for row in rows}, {kind})
            self.assertEqual([row["difficulty"] for row in rows], [str(i % 3) for i in range(64)], kind)
            for row in rows:
                d = int(row["difficulty"])
                if kind == "graph":
                    self.assertTrue(6 + d <= len(row["nodes"]) <= 10 + d, row["id"])
                else:
                    self.assertFalse(row["expression"].startswith("-("), row["id"])
                    self.assertEqual(len(re.findall(r"-?\d+", row["expression"])), 3 + 2 * d, row["id"])

    def test_the_command_line_writes_64_by_default_and_the_size_it_is_given(self):
        default, sized = self.root / "cli_default", self.root / "cli_sized"
        self.assertEqual(self.run_cli(default).returncode, 0)
        self.assertEqual(self.run_cli(sized, "--gradient-reference", "3").returncode, 0)
        for root, size in ((default, 64), (sized, 3)):
            self.assertEqual(len(self.reference_rows(root)), size, root.name)
            manifest = read_json(root / "manifest.json")
            self.assertEqual(manifest["counts"].get("gradient_reference"), size, root.name)
            self.assertEqual(manifest["unique_instances"], 5 + 2 * 4 + size, root.name)

    def test_the_command_line_refuses_an_empty_split(self):
        out = self.root / "cli_zero"
        result = self.run_cli(out, "--gradient-reference", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("All counts must be positive", result.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
