"""a-2 tests: the shared encode and the reference gradient `h = grad R_reference`.

`h` is only meaningful if these hold, and each one is a silent-wrong-number
failure rather than a crash, so each gets a test that fails when it is removed:

  1. the encode is the *training* encode (same functional as the arms), and its
     final label is the EOS token;
  2. the aggregation is each example's own response-token mean, then the mean of
     those with weight 1/K -- not a token-count-weighted mean, which differs as
     soon as response lengths differ;
  3. `normalize` is off, so what is saved is `grad R_reference` and not its
     direction;
  4. h's parameter names, shapes and order are the recorder's `Delta theta` keys.

The encode is pinned against `rsi/backends.py` **as frozen** (blob `e1e05c3a`; PR #14 left
the encode lines unchanged),
by reading its encode lines out of the file rather than copying them, so the day
that file's encode changes this fails instead of two drifted copies agreeing.

The aggregation has two independent expected values, neither computed with
`rsi.backends.response_losses`: the reference derivation's analytic fixture (exact
fractions, re-derived here), and a hand log-softmax oracle on a toy LM whose two
examples have 1 and 3 supervised tokens.

CPU-only by construction: no transformers, no peft, no download.  The real-model
run of `compute_reference_gradient.py` is a GPU run, not this file's.
"""
import ast
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from rsi.common import file_hash, load_config
from rsi.shared_adapter import lora_protocol
from rsi.reference_encode import (
    assert_supervised_last_token,
    check_parameter_binding,
    encode_example,
    supervised_token_count,
)

# An exported RSI_BASE_PIN must not leak into these tests (tests/pin_isolation.py).
from pin_isolation import isolate as setUpModule, restore as tearDownModule  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "compute_reference_gradient.py"


class StubTokenizer:
    """Character ids, no transformers.  `encode` mirrors the real call shape."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False, "the training encode asks for no special tokens"
        return [ord(c) % 100 + 1 for c in text]


class StubBackend:
    """`.prompt_ids(prompt)` and `.tokenizer`: the two attributes encode_example uses."""

    device = "cpu"

    def __init__(self, model=None):
        self.tokenizer = StubTokenizer()
        self.model = model

    def prompt_ids(self, prompt):
        return [200 + i for i in range(1 + len(prompt) % 3)]


class FrozenEncodeTests(unittest.TestCase):
    """The encode here is the frozen training encode, not a lookalike."""

    def frozen_training_encode(self):
        """The encode lines, read out of `rsi/backends.py` as it stands.

        Extracted from the source text, not imported: importing the method would
        need transformers, and the point is to compare against the *file* rather
        than against a copy of it.
        """
        source = (REPO / "rsi" / "backends.py").read_text()
        body = source[source.index("    def train(self, rows, output, seed):"):]
        lines = [line.strip() for line in body.splitlines()]
        wanted = [
            "prefix = self.prompt_ids(r[\"prompt\"])",
            "suffix = self.tokenizer.encode(r[\"response\"], add_special_tokens=False) + [self.tokenizer.eos_token_id]",
            "ids = prefix + suffix",
            "encoded.append((ids, [-100]*len(prefix) + suffix, float(r[\"weight\"])))",
        ]
        for line in wanted:
            self.assertIn(line, lines, "rsi/backends.py no longer encodes this way: %s" % line)
        return wanted

    def test_the_encode_matches_the_frozen_training_encode_line_for_line(self):
        """`encode_example` and `HFBackend.train` build the same ids and labels."""
        self.frozen_training_encode()
        for response, prompt in (("1234", "what is 1+1"), ("", "x"), ("42", "a longer prompt here")):
            stub = StubBackend()
            row = {"prompt": prompt, "response": response, "weight": 1.0}
            # The frozen lines, executed against the same stubs in the order they appear.
            prefix = stub.prompt_ids(row["prompt"])
            suffix = stub.tokenizer.encode(row["response"], add_special_tokens=False) + [stub.tokenizer.eos_token_id]
            ids = prefix + suffix
            encoded = (ids, [-100] * len(prefix) + suffix, float(row["weight"]))
            got_ids, got_labels, got_prefix = encode_example(StubBackend(), prompt, response)
            self.assertEqual(got_ids, encoded[0], (prompt, response))
            self.assertEqual(got_labels, encoded[1], (prompt, response))
            self.assertEqual(got_prefix, len(prefix), (prompt, response))

    def test_the_prompt_is_masked_and_the_response_through_eos_is_supervised(self):
        backend = StubBackend()
        for response in ("7", "1234", "0"):
            ids, labels, prefix = encode_example(backend, "compute", response)
            self.assertEqual(len(ids), len(labels), response)
            self.assertEqual(labels[:prefix], [-100] * prefix, response)
            self.assertEqual(labels[prefix:], ids[prefix:], response)
            self.assertEqual(labels[-1], backend.tokenizer.eos_token_id, response)
            self.assertEqual(labels.count(-100), prefix, response)

    def test_a_masked_final_label_is_refused(self):
        """The EOS supervision is asserted, not assumed."""
        backend = StubBackend()
        ids, labels, prefix = encode_example(backend, "compute", "7")
        assert_supervised_last_token(labels, backend.tokenizer.eos_token_id, "ok")
        labels[-1] = -100
        with self.assertRaises(ValueError):
            assert_supervised_last_token(labels, backend.tokenizer.eos_token_id, "masked")

    def test_a_response_that_tokenizes_to_nothing_still_gets_eos(self):
        backend = StubBackend()
        ids, labels, prefix = encode_example(backend, "compute", "")
        self.assertEqual(labels[prefix:], [backend.tokenizer.eos_token_id])
        assert_supervised_last_token(labels, backend.tokenizer.eos_token_id, "empty response")

    def test_the_supervised_token_count_is_the_encode_s_supervised_labels(self):
        """Matching's length variable is the count the arms train on, from the tokenizer alone."""
        backend = StubBackend()
        for prompt, response in (("compute", ""), ("compute", "7"), ("a longer prompt", '["v1", "v3", "v4"]')):
            ids, labels, prefix = encode_example(backend, prompt, response)
            supervised = sum(label != -100 for label in labels)
            self.assertEqual(supervised_token_count(backend.tokenizer, response), supervised, response)
            self.assertEqual(supervised, len(ids) - prefix, response)


# reference derivation: theta0 = (0, 0), token loss BCE(sigmoid(theta . x), y).
# Responses A, B, C have 1, 3 and 2 tokens.
CODEX_FIXTURE = {
    "A": [((2, 0), 1)],
    "B": [((0, 2), 0), ((0, 4), 0), ((2, 2), 0)],
    "C": [((-2, 0), 1), ((0, -2), 1)],
}
CODEX_STATED = {"h": (Fraction(-1, 18), Fraction(11, 18)),
                "token_mean": (Fraction(1, 6), Fraction(5, 6)),
                "split_2_1_example_means_batches_equal": (Fraction(1, 12), Fraction(7, 12))}
# Labels whose stub encoding gives exactly the fixture's token counts (+ EOS).
CODEX_LABELS = {"A": "", "B": "ab", "C": "c"}
MARKER, SEP, PREFIX = 300, 299, 2


def exact_token_gradients(tokens):
    """d/dtheta BCE(sigmoid(theta . x), y) at theta = 0 is (1/2 - y) x, exactly."""
    return [tuple((Fraction(1, 2) - y) * xi for xi in x) for x, y in tokens]


def mean(vectors):
    return tuple(sum(c) / len(vectors) for c in zip(*vectors))


class ReferenceFixtureModel(nn.Module):
    """the reference derivation's per-token loss, put through the entry's own pipeline.

    A two-way softmax with logits (z, 0) on (label, other) gives -log sigmoid(z),
    and (0, z) gives -log(1 - sigmoid(z)).  So placing z = theta . x on the label
    for y = 1, and on a reserved class for y = 0, makes the cross-entropy that
    `response_losses` takes at that position exactly the reference's BCE.  Every other
    vocabulary entry is -1e4 and contributes exp(-1e4) = 0.

    The example is read from the first prompt token and the response position
    from the fixed two-token prefix.  A device for feeding a given per-token loss
    through encode / shift / per-example mean / accumulate -- not a language model.
    """

    VOCAB, OTHER, FLOOR = 304, 303, -1e4

    def __init__(self):
        super().__init__()
        self.lora_theta = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def forward(self, input_ids, attention_mask=None):
        ids = input_ids[0].tolist()
        tokens = CODEX_FIXTURE["ABC"[ids[0] - MARKER]]
        base = torch.full((len(ids), self.VOCAB), self.FLOOR, dtype=torch.float64)
        coefficients = torch.zeros(len(ids), self.VOCAB, 2, dtype=torch.float64)
        for p in range(len(ids)):
            following = ids[p + 1] if p + 1 < len(ids) else 0
            base[p, following] = 0.0
            base[p, self.OTHER] = 0.0
            j = p - (PREFIX - 1)  # position p predicts response token j
            if 0 <= j < len(tokens):
                x, y = tokens[j]
                live = following if y == 1 else self.OTHER
                coefficients[p, live] = torch.tensor(x, dtype=torch.float64)
        return SimpleNamespace(logits=(base + coefficients @ self.lora_theta).unsqueeze(0))


class ReferenceBackend(StubBackend):
    def prompt_ids(self, prompt):
        return [MARKER + "ABC".index(prompt), SEP]


def reference_pairs():
    return [({"id": name, "prompt": name}, CODEX_LABELS[name]) for name in "ABC"]


class ReferenceFixtureTests(unittest.TestCase):
    """The raw h against the reference derivation's exact values, not against response_losses."""

    def test_the_fixture_reproduces_the_stated_values(self):
        """Re-derived in exact fractions; also pins the two wrong 2+1 readings.

        A 2+1 split ({A, B}, {C}) averaged with equal batch weight is wrong two
        ways, depending on what a batch's own mean is: the mean of its examples'
        means gives (1/12, 7/12), as the reference derivation states; the mean of its
        tokens gives (1/4, 3/4).  Both are reproducible, and both flip the sign of
        h's first coordinate.
        """
        per_example = {name: mean(exact_token_gradients(t)) for name, t in CODEX_FIXTURE.items()}
        self.assertEqual(per_example["A"], (-1, 0))
        self.assertEqual(per_example["B"], (Fraction(1, 3), Fraction(4, 3)))
        self.assertEqual(per_example["C"], (Fraction(1, 2), Fraction(1, 2)))
        h = mean(list(per_example.values()))
        all_tokens = [g for t in CODEX_FIXTURE.values() for g in exact_token_gradients(t)]
        split_example_means = mean([mean([per_example["A"], per_example["B"]]), per_example["C"]])
        ab_tokens = exact_token_gradients(CODEX_FIXTURE["A"]) + exact_token_gradients(CODEX_FIXTURE["B"])
        split_token_means = mean([mean(ab_tokens), mean(exact_token_gradients(CODEX_FIXTURE["C"]))])
        self.assertEqual(h, CODEX_STATED["h"])
        self.assertEqual(mean(all_tokens), CODEX_STATED["token_mean"])
        self.assertEqual(split_example_means, CODEX_STATED["split_2_1_example_means_batches_equal"])
        self.assertEqual(split_token_means, (Fraction(1, 4), Fraction(3, 4)))
        for wrong in (mean(all_tokens), split_example_means, split_token_means):
            self.assertLess(h[0], 0)
            self.assertGreater(wrong[0], 0)

    def compute(self):
        from compute_reference_gradient import compute_h

        model = ReferenceFixtureModel()
        h, per_sample = compute_h(ReferenceBackend(model), reference_pairs(), seed=0)
        return model, h, per_sample

    def test_compute_h_returns_the_raw_mean_of_per_response_means(self):
        _, h, per_sample = self.compute()
        self.assertEqual(list(h), ["lora_theta"])
        expected = torch.tensor([float(c) for c in CODEX_STATED["h"]])
        self.assertTrue(torch.allclose(h["lora_theta"], expected, rtol=0, atol=1e-6),
                        "%s vs %s" % (h["lora_theta"].tolist(), expected.tolist()))
        self.assertEqual([s["supervised_tokens"] for s in per_sample], [1, 3, 2])
        for s in per_sample:
            self.assertAlmostEqual(s["loss"], math.log(2), places=6)  # every token sits at z = 0

    def test_h_is_the_gradient_not_its_direction(self):
        """|h| is the reference's sqrt(122)/18, not 1: nothing normalises it."""
        _, h, _ = self.compute()
        norm = float(h["lora_theta"].double().norm())
        self.assertAlmostEqual(norm, math.sqrt(122) / 18, places=6)
        self.assertGreater(abs(norm - 1.0), 0.3)


class TinyLM(nn.Module):
    """A toy LM: fixed random embeddings, then two LoRA-named factors to logits."""

    VOCAB = 304

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.register_buffer("embed", torch.randn(self.VOCAB, 6, generator=g))
        self.lora_A = nn.Parameter(0.5 * torch.randn(6, 4, generator=g))
        self.lora_B = nn.Parameter(0.5 * torch.randn(4, self.VOCAB, generator=g))

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.embed[input_ids] @ self.lora_A @ self.lora_B)


class UnequalLengthTests(unittest.TestCase):
    """1 and 3 supervised tokens: token weights (1/4, 3/4) against example weights (1/2, 1/2)."""

    PAIRS = [({"id": "short", "prompt": "a"}, ""), ({"id": "long", "prompt": "bb"}, "42")]

    def oracle_token_losses(self, model, prompt, label):
        """Per-token NLLs by hand: explicit shift, mask and log-sum-exp; no response_losses."""
        ids, labels, _ = encode_example(StubBackend(), prompt, label)
        logits = model(torch.tensor([ids])).logits[0]
        return [torch.logsumexp(logits[p], 0) - logits[p, labels[p + 1]]
                for p in range(len(ids) - 1) if labels[p + 1] != -100]

    def test_compute_h_is_the_example_mean_and_not_the_token_mean(self):
        from compute_reference_gradient import compute_h

        model = TinyLM()
        params = [model.lora_A, model.lora_B]
        token_losses = [self.oracle_token_losses(model, row["prompt"], label) for row, label in self.PAIRS]
        counts = [len(t) for t in token_losses]
        self.assertEqual(counts, [1, 3])
        self.assertEqual([Fraction(n, sum(counts)) for n in counts], [Fraction(1, 4), Fraction(3, 4)])

        example_mean = sum(torch.stack(t).mean() for t in token_losses) / len(token_losses)
        token_mean = torch.stack([l for t in token_losses for l in t]).mean()
        want = torch.autograd.grad(example_mean, params, retain_graph=True)
        weighted = torch.autograd.grad(token_mean, params)
        gap = max(float((a - b).abs().max()) for a, b in zip(want, weighted))
        self.assertGreater(gap, 1e-2, "the fixture does not separate the two aggregations")

        h, per_sample = compute_h(StubBackend(model), self.PAIRS, seed=0)
        self.assertEqual(list(h), ["lora_A", "lora_B"])
        self.assertEqual([s["supervised_tokens"] for s in per_sample], [1, 3])
        for name, w, t in zip(h, want, weighted):
            self.assertTrue(torch.allclose(h[name], w, rtol=1e-5, atol=1e-6), name)
            self.assertFalse(torch.allclose(h[name], t, rtol=1e-3, atol=1e-4), name)
        for s, t in zip(per_sample, token_losses):
            self.assertAlmostEqual(s["loss"], float(torch.stack(t).mean().detach()), places=5)


class ParameterBindingTests(unittest.TestCase):
    """The projection's coordinates, stated as a check."""

    EXPECTED = {"base.lora_A.weight": (2, 3), "base.lora_B.weight": (4, 2)}

    def test_matching_names_shapes_and_order_pass(self):
        h = {"base.lora_A.weight": torch.zeros(2, 3), "base.lora_B.weight": torch.zeros(4, 2)}
        self.assertTrue(check_parameter_binding(h, self.EXPECTED))

    def test_a_missing_parameter_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            check_parameter_binding({"base.lora_A.weight": torch.zeros(2, 3)}, self.EXPECTED)

    def test_an_extra_parameter_is_refused(self):
        h = {"base.lora_A.weight": torch.zeros(2, 3), "base.lora_B.weight": torch.zeros(4, 2),
             "other.lora_A.weight": torch.zeros(1, 1)}
        with self.assertRaisesRegex(ValueError, "unexpected"):
            check_parameter_binding(h, self.EXPECTED)

    def test_a_right_name_with_the_wrong_shape_is_refused(self):
        """`_dot_product` checks names, not shapes, so this has to be checked here."""
        h = {"base.lora_A.weight": torch.zeros(2, 3), "base.lora_B.weight": torch.zeros(9, 9)}
        with self.assertRaisesRegex(ValueError, "wrong shapes"):
            check_parameter_binding(h, self.EXPECTED)

    def test_the_right_parameters_in_another_order_are_refused(self):
        h = {"base.lora_B.weight": torch.zeros(4, 2), "base.lora_A.weight": torch.zeros(2, 3)}
        with self.assertRaisesRegex(ValueError, "order"):
            check_parameter_binding(h, self.EXPECTED)


class WriteReferenceGradientTests(unittest.TestCase):
    """The core of the entry end to end on CPU: h.pt, per_sample.jsonl, h.json."""

    BINDINGS = {"base_model": "fixture", "base_revision": "0" * 40,
                "shared_adapter_parameter_hash": "fixture"}

    def write(self, out, backend=None, pairs=None):
        from compute_reference_gradient import write_reference_gradient

        backend = backend or ReferenceBackend(ReferenceFixtureModel())
        return write_reference_gradient(backend, pairs or reference_pairs(), out, 0, dict(self.BINDINGS),
                                        "reference_answer")

    def test_the_record_binds_the_files_and_states_the_definitions(self):
        from rsi.common import source_hash
        from rsi.gradient_recorder import GradientRecorder

        git = {"git_commit": "c" * 40, "git_dirty": True, "changes": ["x.py"], "reason": None}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("compute_reference_gradient.git_state", return_value=git):
            out = Path(tmp) / "h"
            model = ReferenceFixtureModel()
            record = self.write(out, ReferenceBackend(model))
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["h.json", "h.pt", "per_sample.jsonl"])
            on_disk = json.loads((out / "h.json").read_text())
            self.assertEqual(on_disk, json.loads(json.dumps(record)))
            self.assertEqual(on_disk["h"]["sha256"], file_hash(out / "h.pt"))
            self.assertEqual(on_disk["per_sample"]["sha256"], file_hash(out / "per_sample.jsonl"))
            self.assertNotIn(file_hash(out / "h.json"), (out / "h.json").read_text())
            self.assertIs(on_disk["normalize"], False)
            self.assertFalse(on_disk["loss"]["token_weighted"])
            self.assertEqual(on_disk["label_source"], "reference_answer")
            self.assertIn("not derived as the only reading", on_disk["label_source_note"])
            self.assertEqual(on_disk["h"]["parameter_names"], ["lora_theta"])
            self.assertAlmostEqual(on_disk["h"]["l2_norm"], math.sqrt(122) / 18, places=6)
            self.assertEqual(on_disk["supervised_tokens"], {"min": 1, "max": 3, "equal": False})
            self.assertEqual(on_disk["bindings"], self.BINDINGS)
            # The code that ran: git_state's fields as the runner binds them, and source_hash.
            self.assertEqual(on_disk["code"], {"git_commit": "c" * 40, "git_dirty": True, "git_changes": ["x.py"],
                                               "git_reason": None, "source_hash": source_hash()})
            lines = (out / "per_sample.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(l)["task_id"] for l in lines], ["A", "B", "C"])
            # What an arm loads is what was computed.
            recorder = GradientRecorder(model, str(out / "h.pt"))
            self.assertTrue(torch.allclose(recorder.h["lora_theta"],
                                           torch.tensor([-1 / 18, 11 / 18]), atol=1e-6))

    def test_a_selection_the_recorder_does_not_share_is_refused(self):
        """The round trip compares against the recorder's own Delta theta, so it can fail."""
        import compute_reference_gradient as entry

        original = entry.lora_parameters
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(entry, "lora_parameters", lambda m: list(reversed(original(m)))):
            with self.assertRaisesRegex(ValueError, "order"):
                self.write(Path(tmp) / "h", StubBackend(TinyLM()), UnequalLengthTests.PAIRS)
            self.assertFalse((Path(tmp) / "h" / "h.json").exists())

    def test_a_written_reference_gradient_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "h"
            out.mkdir()
            (out / "h.pt").write_bytes(b"earlier")
            with self.assertRaises(SystemExit):
                self.write(out)
            self.assertEqual((out / "h.pt").read_bytes(), b"earlier")


def code_only(path):
    """The file's executable code, without docstrings.

    A text scan sees the docstring that explains a forbidden pattern and flags it;
    dropping the docstring nodes and unparsing the rest leaves only code.
    """
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


class EntryPointTests(unittest.TestCase):
    """The entry's own choices, on the source and through the real CLI."""

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO,
                              capture_output=True, text=True)

    def test_the_label_source_defaults_to_reference_answer(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reference_answer", result.stdout)
        match = re.search(r'add_argument\("--label-source",\s*default="([^"]+)"', SCRIPT.read_text())
        self.assertIsNotNone(match, "no --label-source default found in compute_reference_gradient.py")
        self.assertEqual(match.group(1), "reference_answer")

    def test_the_entry_never_normalises_and_never_token_weights(self):
        """The two implementation risks as *code* properties, docstrings excluded."""
        code = code_only(SCRIPT)
        self.assertNotIn("size(0)", code, "the entry multiplies by a token count")
        self.assertIn("REQUIRED_NORMALIZE = False", code)
        self.assertIn("len(pairs)", code)

    def test_the_entry_does_not_use_the_default_normalising_helper(self):
        """It must not delegate to `ReferenceGradientComputer`, default `normalize=True`."""
        tree = ast.parse(SCRIPT.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        self.assertNotIn("rsi.reference_gradient", imported)
        self.assertNotIn("ReferenceGradientComputer", code_only(SCRIPT))
        self.assertNotIn("create_reference_gradient", code_only(SCRIPT))

    def test_normalize_is_refused_by_the_real_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "h"
            result = self.run_cli("--config", "configs/pilot.json", "--data", "data",
                                  "--shared-adapter", "shared_adapter", "--out", str(out), "--normalize")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("normalize must be False", result.stderr)
            self.assertFalse(out.exists())

    def test_the_mock_backend_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "h"
            result = self.run_cli("--config", "configs/pilot.json", "--data", "data",
                                  "--shared-adapter", "shared_adapter", "--out", str(out),
                                  "--backend", "mock")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mock backend", result.stderr)
            self.assertFalse(out.exists())

    def test_the_config_is_pinned_before_the_model_is_loaded(self):
        """`verify_shared_adapter` needs a commit id; the default revision is "main"."""
        import compute_reference_gradient as entry

        seen = []

        def pin(config):
            seen.append(config["revision"])
            raise SystemExit("pin_config reached")

        args = Namespace(config="configs/pilot.json", data="data", shared_adapter="shared_adapter",
                         out="unused", label_source="reference_answer", normalize=False, seed=0,
                         backend=None)
        raised = None
        with mock.patch("rsi.experiment.pin_config", pin), \
                mock.patch("rsi.backends.backend") as load:
            try:
                entry.main(args)
            except (Exception, SystemExit) as exc:
                raised = exc
        self.assertEqual(seen, ["main"])
        self.assertEqual(str(raised), "pin_config reached")
        load.assert_not_called()

    def test_a_config_the_arms_would_refuse_is_refused_before_pinning(self):
        """Dropout on: `train()` mode would make h a random draw; the arms refuse it too."""
        import compute_reference_gradient as entry

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "dropout.json"
            config.write_text(json.dumps({"training": {"lora_dropout": 0.1}}))
            args = Namespace(config=str(config), data="data", shared_adapter="shared_adapter",
                             out=str(Path(tmp) / "h"), label_source="reference_answer",
                             normalize=False, seed=0, backend=None)
            raised = None
            with mock.patch("rsi.experiment.pin_config", side_effect=SystemExit("pinned")) as pin:
                try:
                    entry.main(args)
                except (ValueError, SystemExit) as exc:
                    raised = exc
            self.assertIsInstance(raised, ValueError)
            self.assertIn("isolation violated", str(raised))
            pin.assert_not_called()


class EntryBindingTests(unittest.TestCase):
    """`main` end to end on CPU: a generated dataset, the repo's config, a toy model.

    Stubbed: the revision lookup (`pin_config`, network), the model load
    (`rsi.backends.backend`, transformers/peft) and `verify_shared_adapter`
    (a saved PEFT adapter).  Everything between them is the real entry.
    """

    CONFIG = REPO / "configs" / "matched_pool.json"  # complete, as every hf matched stage requires
    PINNED = "0" * 40

    def dataset(self, root):
        from generate_data import generate

        return generate(root, "arithmetic", 7, rounds=1, per_round=2, dev=1, calibration=1, eval_id=1,
                        eval_ood=1, gradient_reference=3)

    def run_main(self, tmp, data):
        import compute_reference_gradient as entry

        # The base pin the entry checks against, in the same directory the run
        # writes into.  It is set to the model this config names at the forty
        # zeros the stubbed pin_config resolves to, so a-2's subjects are
        # unchanged; the pin itself is exercised in tests/test_base_pin.py.
        raw = json.loads(self.CONFIG.read_text())
        (Path(tmp) / "base_pin.json").write_text(json.dumps(
            {"model": raw.get("model", "Qwen/Qwen3-1.7B"), "revision": self.PINNED,
             "source": "a-2 fixture", "prepared": "test"}), encoding="utf-8")

        stub = StubBackend(TinyLM())
        stub.model_name, stub.revision, stub.close = "fixture/base", self.PINNED, mock.Mock()
        pinned = []

        def pin(config):
            pinned.append(dict(config, revision=self.PINNED))
            return pinned[-1]

        args = Namespace(config=str(self.CONFIG), data=str(data), shared_adapter=str(Path(tmp) / "adapter"),
                         out=str(Path(tmp) / "h"), label_source="reference_answer", normalize=False, seed=0,
                         backend=None)
        raised = None
        with mock.patch("rsi.experiment.pin_config", pin), \
                mock.patch("rsi.base_pin.BASE_PIN", Path(tmp) / "base_pin.json"), \
                mock.patch("rsi.backends.backend", return_value=stub) as load, \
                mock.patch("rsi.shared_adapter.verify_shared_adapter",
                           return_value={"parameter_hash": "fixture-hash",
                                         "protocol": lora_protocol(load_config(self.CONFIG)["training"])}) as verify:
            try:
                entry.main(args)
            except (Exception, SystemExit) as exc:
                raised = exc
        return SimpleNamespace(raised=raised, stub=stub, pinned=pinned, load=load, verify=verify, args=args)

    def test_main_binds_the_config_the_data_and_the_initialisation(self):
        import hashlib

        from rsi.common import digest
        from rsi.tasks import reference_answer

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            manifest = self.dataset(data)
            run = self.run_main(tmp, data)
            self.assertIsNone(run.raised, repr(run.raised))
            self.assertEqual(len(run.pinned), 1, "pin_config was not called exactly once")
            bindings = json.loads((Path(run.args.out) / "h.json").read_text())["bindings"]
            self.assertEqual(bindings["config_hash"], digest(run.pinned[0]))
            self.assertEqual(bindings["config_file"]["sha256"], hashlib.sha256(self.CONFIG.read_bytes()).hexdigest())
            self.assertEqual(bindings["data"]["manifest_sha256"],
                             hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest())
            self.assertEqual(bindings["data"]["dataset_hash"], digest(manifest))
            self.assertEqual(bindings["data"]["gradient_reference_sha256"],
                             manifest["files"]["gradient_reference.jsonl"])
            self.assertEqual(bindings["shared_adapter_parameter_hash"], "fixture-hash")
            self.assertEqual((bindings["base_model"], bindings["base_revision"], bindings["device"]),
                             ("fixture/base", self.PINNED, "cpu"))
            run.load.assert_called_once_with(run.pinned[0], adapter=run.args.shared_adapter, trainable=True)
            run.verify.assert_called_once_with(run.stub.model, run.args.shared_adapter, "fixture/base", self.PINNED)
            run.stub.close.assert_called_once_with()
            rows = [json.loads(line) for line in (data / "gradient_reference.jsonl").read_text().splitlines()]
            per_sample = [json.loads(line) for line in (Path(run.args.out) / "per_sample.jsonl").read_text().splitlines()]
            self.assertEqual([(s["task_id"], s["label"]) for s in per_sample],
                             [(r["id"], reference_answer(r)) for r in rows])

    def test_a_reference_split_the_manifest_does_not_bind_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            self.dataset(data)
            manifest = json.loads((data / "manifest.json").read_text())
            del manifest["files"]["gradient_reference.jsonl"]
            (data / "manifest.json").write_text(json.dumps(manifest))
            run = self.run_main(tmp, data)
            self.assertIsInstance(run.raised, SystemExit)
            self.assertIn("does not list gradient_reference.jsonl", str(run.raised))
            run.load.assert_not_called()

    def test_a_changed_reference_split_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            self.dataset(data)
            split = data / "gradient_reference.jsonl"
            split.write_text("".join(split.read_text().splitlines(keepends=True)[1:]))
            run = self.run_main(tmp, data)
            self.assertIsInstance(run.raised, ValueError)
            self.assertIn("Dataset changed: gradient_reference.jsonl", str(run.raised))
            run.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
