"""Regression tests for the one-step accumulation defect in `one_step.py`.

At a6ad238 `_run_hf_arm` called `optimizer.zero_grad()` before *every* example,
so the single `optimizer.step()` saw only the last example's gradient instead of
the K-example gradient the comparison is defined on.  It also recorded each
example's loss already divided by K, and then divided the mean loss by K again.
With K > 1 the recorder's own composition check (G_C + G_E = G_preclip) raised at
`recorder.save` -- after the step had already been applied.

These tests drive `run_arm` -> `_run_hf_arm` itself, not a copy of its loop, on
CPU: a toy model with LoRA-named parameters stands in for the PEFT-wrapped base
and a fake backend stands in for `HFBackend`.  The reference gradient path is
passed explicitly, so the separate missing-path defect in `main` is not what
these tests see.  Expected values come from a recomputation (per-example
autograd on a fresh copy of the toy) that is independent of the code under
test except for three shared components: rsi.backends.response_losses, the
encoding convention (prompt_ids + encode(response, add_special_tokens=False)
+ [eos]), and one_step.CLIP_THRESHOLD.

The last test is about the start, not the step: the shared adapter's hash
covers the adapter alone, so `_run_hf_arm` must refuse an adapter recorded on
another base before any forward pass.
"""
import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import one_step
import run_iterative_experiment
from rsi.backends import response_losses
from rsi.common import load_config
from rsi.shared_adapter import adapter_parameter_hash

LEARNING_RATE = 1e-2  # large enough that the update sits far above float32 noise
EOS = 1
BASE_MODEL, BASE_REVISION = "toy/base", "0123456789abcdef0123456789abcdef01234567"
# K = 4: with K = 1 the defect is invisible.  C:E = 3:1 as in the matched subsets.
ROWS = [
    {"id": "c0", "prompt": "3+4", "response": "7", "correct": True, "weight": 1.0},
    {"id": "c1", "prompt": "10+5", "response": "15", "correct": True, "weight": 1.0},
    {"id": "c2", "prompt": "2+2", "response": "4", "correct": True, "weight": 1.0},
    {"id": "e0", "prompt": "9+9", "response": "81", "correct": False, "weight": 1.0},
]


def encode(text):
    return [2 + ord(ch) % 10 for ch in text]


class FakePeftModel(torch.nn.Module):
    """Stands in for `peft.PeftModel`, which `verify_shared_adapter` checks with isinstance."""


@contextlib.contextmanager
def fake_peft():
    """Make `from peft import PeftModel` give FakePeftModel, touching only that entry.

    `mock.patch.dict(sys.modules, ...)` would be the obvious tool, but on exit it
    also drops every module first imported inside the block -- torch._dynamo, on
    the first optimizer.step() -- and torch cannot import those twice in one process.
    """
    module = types.ModuleType("peft")
    module.PeftModel = FakePeftModel
    previous = sys.modules.get("peft")
    sys.modules["peft"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["peft"]
        else:
            sys.modules["peft"] = previous


class ToyLoraModel(FakePeftModel):
    """Frozen embedding, projection and head around one trainable LoRA pair."""

    def __init__(self, vocab=12, width=5, rank=2):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Parameter(torch.randn(vocab, width, generator=g), requires_grad=False)
        self.q_proj = torch.nn.Parameter(torch.randn(width, width, generator=g), requires_grad=False)
        self.head = torch.nn.Parameter(torch.randn(vocab, width, generator=g), requires_grad=False)
        # PEFT starts lora_B at zero; it is non-zero here so that lora_A gets a
        # gradient as well and every coordinate of the step is exercised.
        self.lora_A = torch.nn.Parameter(0.5 * torch.randn(rank, width, generator=g))
        self.lora_B = torch.nn.Parameter(0.5 * torch.randn(width, rank, generator=g))
        self.config = types.SimpleNamespace(use_cache=True)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed[input_ids]
        x = x @ self.q_proj.T + 2.0 * (x @ self.lora_A.T @ self.lora_B.T)
        return types.SimpleNamespace(logits=torch.tanh(x) @ self.head.T)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class ToyBackend:
    """The surface of `HFBackend` that `_run_hf_arm` uses."""

    def __init__(self, model):
        self.model, self.device = model, "cpu"
        self.model_name, self.revision = BASE_MODEL, BASE_REVISION
        self.tokenizer = types.SimpleNamespace(
            eos_token_id=EOS, encode=lambda text, add_special_tokens: encode(text))

    def prompt_ids(self, prompt):
        return encode(prompt)

    def close(self):
        pass


def trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def expected_step(rows):
    """Recomputation: per-example gradients, their mean, the clip, one AdamW step.

    Independent of the code under test except rsi.backends.response_losses, the
    encoding convention (prompt_ids + encode(response, add_special_tokens=False)
    + [eos]), and one_step.CLIP_THRESHOLD.
    """
    model = ToyLoraModel()
    params = trainable(model)
    losses, total = [], {n: torch.zeros_like(p) for n, p in params.items()}
    active = [row for row in rows if row["weight"] > 0]
    denominator = sum(row["weight"] for row in active)
    for row in active:
        prefix, suffix = encode(row["prompt"]), encode(row["response"]) + [EOS]
        ids = torch.tensor([prefix + suffix])
        labels = torch.full_like(ids, -100)
        labels[0, len(prefix):] = torch.tensor(suffix)
        loss = response_losses(model(ids).logits, labels).sum()
        for name, g in zip(params, torch.autograd.grad(loss, list(params.values()))):
            total[name] += g * row["weight"]
        losses.append(loss.item())
    mean = {n: g / denominator for n, g in total.items()}
    norm = torch.sqrt(sum((g ** 2).sum() for g in mean.values())).item()
    scale = min(1.0, one_step.CLIP_THRESHOLD / (norm + 1e-6))  # clip_grad_norm_'s rule
    clipped = {n: g * scale for n, g in mean.items()}
    optimizer = torch.optim.AdamW(list(params.values()), lr=LEARNING_RATE, weight_decay=0.0)
    for name, p in params.items():
        p.grad = clipped[name].clone()
    optimizer.step()
    return {"losses": losses, "preclip_norm": norm, "clipped": clipped,
            "theta": {n: p.detach().clone() for n, p in params.items()}}


class OneStepAccumulationTests(unittest.TestCase):
    def run_arm(self, branch, shared=None, rows=None):
        """Run one arm through `one_step.run_arm` on the toy; capture every optimizer step.

        `shared` overrides fields of the shared adapter record, which otherwise
        matches the toy.  An exception from the arm is returned, not raised, so a
        test can still inspect the step that was taken before it.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp = Path(tmp.name)
        config_path = tmp / "config.json"
        config_path.write_text(json.dumps({"training": {"learning_rate": LEARNING_RATE}}))
        config = load_config(str(config_path), 0, "hf")
        model = ToyLoraModel()
        theta0 = {n: p.detach().clone() for n, p in trainable(model).items()}
        adapter = tmp / "shared_adapter"
        adapter.mkdir()
        fields = {"parameter_hash": adapter_parameter_hash(model), "base_model": BASE_MODEL,
                  "base_revision": BASE_REVISION}
        fields.update(shared or {})
        (adapter / "shared_adapter.json").write_text(json.dumps(fields))
        reference = tmp / "reference_gradient.pt"
        torch.save({n: torch.ones_like(v) for n, v in theta0.items()}, reference)

        names = {id(p): n for n, p in model.named_parameters()}
        steps, forwards = [], []

        def capture(optimizer, args, kwargs):
            steps.append({names[id(p)]: p.grad.detach().clone()
                          for group in optimizer.param_groups for p in group["params"]})

        record, error = None, None
        hook = register_optimizer_step_pre_hook(capture)
        model.register_forward_pre_hook(lambda module, args: forwards.append(1))
        try:
            with fake_peft(), \
                    mock.patch("rsi.backends.backend", lambda config, adapter=None, **kw: ToyBackend(model)):
                record = one_step.run_arm(branch, ROWS if rows is None else rows, config, str(adapter), tmp / branch, 0,
                                          {"reference_gradient_path": str(reference)})
        except Exception as exc:  # returned to the test, see docstring
            error = exc
        finally:
            hook.remove()
        diagnostics_path = tmp / branch / "diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else None
        return {"record": record, "error": error, "diagnostics": diagnostics, "steps": steps,
                "forwards": len(forwards), "theta0": theta0,
                "theta": {n: p.detach().clone() for n, p in trainable(model).items()}, "out": tmp / branch}

    def assertArmCompleted(self, run):
        self.assertIsNone(run["error"], "the arm raised: %r" % (run["error"],))

    def test_the_step_uses_the_gradient_of_all_k_examples(self):
        """The one optimizer.step() sees clip(mean_i grad loss_i), not the last example's gradient."""
        run, want = self.run_arm("R"), expected_step(ROWS)
        self.assertEqual(len(run["steps"]), 1)
        for name, grad in want["clipped"].items():
            torch.testing.assert_close(run["steps"][0][name], grad, rtol=1e-5, atol=1e-7)
        self.assertArmCompleted(run)
        diagnostics = run["diagnostics"]
        self.assertTrue(diagnostics["validation"]["composition_valid"])
        # When the clip fires (it does for this toy) the step sees a vector of norm
        # CLIP_THRESHOLD whatever G's scale, so a pure scale error in G shows only in
        # G_preclip_norm; that is why it is checked here as well.
        self.assertEqual(diagnostics["clipping"]["fired"], want["preclip_norm"] > one_step.CLIP_THRESHOLD)
        self.assertAlmostEqual(diagnostics["gradient_norms"]["G_preclip_norm"], want["preclip_norm"],
                               delta=1e-5 * want["preclip_norm"])

    def test_per_sample_losses_are_recorded_unscaled(self):
        """The recorder is given loss_i itself; at a6ad238 it was given loss_i / K."""
        run, want = self.run_arm("R"), expected_step(ROWS)
        self.assertArmCompleted(run)
        recorded = run["diagnostics"]["per_sample_losses"]
        self.assertEqual(len(recorded), len(ROWS))
        for got, expected in zip(recorded, want["losses"]):
            self.assertAlmostEqual(got, expected, delta=1e-5 * expected)

    def test_mean_loss_is_the_mean_per_example_loss(self):
        """At a6ad238 the mean was divided by K a second time: sum(loss_i) / K**2."""
        run, want = self.run_arm("R"), expected_step(ROWS)
        self.assertArmCompleted(run)
        expected = sum(want["losses"]) / len(ROWS)
        self.assertAlmostEqual(run["record"]["mean_loss"], expected, delta=1e-5 * expected)

    def test_the_update_is_one_adamw_step_from_that_gradient(self):
        """End to end: the parameters land where one AdamW step from that gradient puts them.

        AdamW's first step is close to lr * sign(g), so this cannot see a pure
        scale error in the gradient; the G_preclip_norm check above does.
        """
        run, want = self.run_arm("R"), expected_step(ROWS)
        self.assertArmCompleted(run)
        for name, theta in want["theta"].items():
            torch.testing.assert_close(run["theta"][name], theta, rtol=1e-5, atol=1e-6)
            self.assertFalse(torch.equal(run["theta"][name], run["theta0"][name]))
        self.assertTrue((run["out"] / "adapter").is_dir())

    def test_null_arm_records_the_same_gradient_and_takes_no_step(self):
        """Same forward passes and recorder as R, no optimizer.step(), no adapter, Delta theta = 0."""
        run, want = self.run_arm("null"), expected_step(ROWS)
        self.assertEqual(run["steps"], [])
        self.assertArmCompleted(run)
        for name, theta in run["theta0"].items():
            self.assertTrue(torch.equal(run["theta"][name], theta))
        diagnostics = run["diagnostics"]
        self.assertEqual(diagnostics["delta_theta"]["norm"], 0.0)
        self.assertEqual(diagnostics["main_diagnostic"]["h_T_delta_theta"], 0.0)
        self.assertAlmostEqual(diagnostics["gradient_norms"]["G_preclip_norm"], want["preclip_norm"],
                               delta=1e-5 * want["preclip_norm"])
        self.assertFalse((run["out"] / "adapter").exists())

    def test_an_adapter_recorded_on_another_base_is_refused_before_any_forward_pass(self):
        """The adapter hash leaves the base out, so the record's base model and revision bind it.

        Here the adapter tensors match the record exactly; only the base it was
        recorded on differs from the base the backend loaded.
        """
        for key, other in (("base_model", "toy/other"), ("base_revision", "f" * 40)):
            run = self.run_arm("R", {key: other})
            self.assertIsInstance(run["error"], ValueError, "%s %s was accepted" % (key, other))
            self.assertIn(key, str(run["error"]))
            self.assertEqual((run["forwards"], run["steps"], run["diagnostics"]), (0, [], None), key)
            self.assertFalse((run["out"] / "arm.json").exists())

    def test_audited_weights_normalize_the_actual_gradient_and_loss(self):
        rows = [dict(row, weight=w) for row, w in zip(ROWS, (1, 0.5, 0, 2))]
        run, want = self.run_arm("R", rows=rows), expected_step(rows)
        self.assertArmCompleted(run)
        self.assertEqual(run["forwards"], 3)
        self.assertEqual(run["record"]["positive_weight_examples"], 3)
        self.assertTrue(run["diagnostics"]["validation"]["composition_valid"])
        for name, grad in want["clipped"].items():
            torch.testing.assert_close(run["steps"][0][name], grad, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(run["theta"][name], want["theta"][name], rtol=1e-5, atol=1e-6)
        weights = [r["weight"] for r in rows if r["weight"] > 0]
        loss = sum(w * value for w, value in zip(weights, want["losses"])) / sum(weights)
        self.assertAlmostEqual(run["record"]["mean_loss"], loss, delta=1e-5 * loss)

    def test_empty_or_zero_weight_audited_subset_is_a_saved_noop(self):
        for rows in ([], [dict(r, weight=0) for r in ROWS]):
            with self.subTest(examples=len(rows)):
                run = self.run_arm("R", rows=rows)
                self.assertArmCompleted(run)
                self.assertEqual((run["forwards"], run["steps"]), (0, []))
                self.assertEqual(run["record"]["updates"], 0)
                self.assertFalse(run["record"]["trained"])
                self.assertIsNone(run["record"]["mean_loss"])
                self.assertEqual(run["diagnostics"]["delta_theta"]["norm"], 0)
                self.assertEqual(run["diagnostics"]["main_diagnostic"]["h_T_delta_theta"], 0)
                self.assertTrue((run["out"] / "adapter").is_dir())
                for name in run["theta"]:
                    self.assertTrue(torch.equal(run["theta"][name], run["theta0"][name]))

    def test_invalid_audited_weights_are_refused_before_output_or_forward_pass(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                run = self.run_arm("R", rows=[dict(ROWS[0], weight=weight)])
                self.assertIsInstance(run["error"], ValueError)
                self.assertEqual(run["forwards"], 0)
                self.assertFalse(run["out"].exists())

    def test_iterative_noop_training_saves_unchanged_adapter_and_zero_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ToyLoraModel()
            h = root / "h.pt"
            torch.save({n: torch.ones_like(p) for n, p in trainable(model).items()}, h)
            wrapped = ToyBackend(model)
            wrapped.train = mock.Mock(return_value={"trained": False, "steps": 0, "mean_loss": None})
            wrapped.tokenizer.save_pretrained = mock.Mock()
            with mock.patch("rsi.backends.backend", return_value=wrapped):
                stats = run_iterative_experiment.train_arm({}, root / "prior", [], root / "adapter", 0, h)
            self.assertEqual(stats["steps"], 0)
            self.assertEqual(stats["delta_theta_norm"], 0)
            self.assertEqual(stats["h_T_delta_theta"], 0)
            self.assertTrue((root / "adapter").is_dir())
            wrapped.tokenizer.save_pretrained.assert_called_once_with(root / "adapter")


if __name__ == "__main__":
    unittest.main()
