"""Tests for the shared adapter's parameter hash and the check that binds it to a base.

`h.json` will bind `adapter_parameter_hash` and the ordered adapter parameter
names, and the gradient recorder keys Delta theta by those names, so the hash
has to mean the adapter tensors and nothing else.  At e7eea0e it hashed
`model.state_dict()`, which for a PeftModel is the whole model: every base
weight was copied to the CPU on each call, and the value moved whenever the
base did.  With the base out of the hash, nothing in it ties the adapter to its
base, so `verify_shared_adapter` takes the base model ID and revision the
caller loaded and refuses a record made on any other.

The toy is laid out the way PEFT lays out a LoRA-wrapped causal LM:
`base_model.model.model.layers.N.self_attn.q_proj.base_layer.weight`, with
`lora_A.<adapter>.weight` and `lora_B.<adapter>.weight` beside it and a
`lora_dropout` that holds no parameters.  That layout is taken from reading
PEFT's LoraLayer; PEFT is not installed here, so it is not checked against a
real PeftModel.  Twelve layers make sorted names differ from model order
("layers.10" sorts before "layers.2").

What decides the expectations: the hash's domain and the name order come from
the pinned GradientRecorder (`theta_before`), which applies the same "lora" name
rule as the code under test, so those tests show agreement with the recorder,
which is what `h.json` needs, and not that the rule is right in general.  The
toy marks its adapter tensors by module structure, not by name, and a
precondition checks that the recorder finds exactly those.  Every other
expectation is a property of the digest (equal or not) and needs no second
implementation of it.
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

from rsi.gradient_recorder import GradientRecorder
from rsi.shared_adapter import (adapter_parameter_hash, create_shared_adapter, init_shared_adapter,
                                lora_parameter_names, require_commit_id, verify_shared_adapter)

WIDTH, RANK, LAYERS = 4, 2, 12
REVISION = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d"
BASE = {"base_model": "toy/base", "base_revision": REVISION}


class FakePeftModel(torch.nn.Module):
    """Stands in for `peft.PeftModel`, which `verify_shared_adapter` checks with isinstance."""


def peft_module(**members):
    """A stand-in for `peft` carrying the names the code under test imports.

    `rsi.backends` and `rsi.shared_adapter` both import from peft inside the
    function that needs it, so this only has to be in `sys.modules` for the
    call.  `PeftModel` is the toy class the isinstance check needs and
    `LoraConfig` is a mock, so a test that gets past the revision check reaches
    its own guard rather than a module that is not installed here.
    """
    module = types.ModuleType("peft")
    module.PeftModel = FakePeftModel
    module.LoraConfig = mock.MagicMock()
    for name, value in members.items():
        setattr(module, name, value)
    return module


@contextlib.contextmanager
def fake_peft(**members):
    """Make `from peft import ...` see `peft_module(**members)`, touching only that entry.

    As in test_one_step_accumulation: `mock.patch.dict(sys.modules, ...)` would
    also drop every module first imported inside the block on exit.
    """
    module = peft_module(**members)
    previous = sys.modules.get("peft")
    sys.modules["peft"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["peft"]
        else:
            sys.modules["peft"] = previous


class LoraLinear(torch.nn.Module):
    """A linear layer wrapped as PEFT's LoraLayer wraps it, as far as parameter names go.

    The wrapped layer is `base_layer`; each adapter name has one A/B pair in the
    `lora_A` / `lora_B` ModuleDicts.  Nothing here runs a forward pass.
    """

    def __init__(self, adapter, device):
        super().__init__()
        self.base_layer = torch.nn.Linear(WIDTH, WIDTH, bias=False, device=device)
        self.lora_dropout = torch.nn.ModuleDict({adapter: torch.nn.Identity()})
        self.lora_A = torch.nn.ModuleDict({adapter: torch.nn.Linear(WIDTH, RANK, bias=False)})
        self.lora_B = torch.nn.ModuleDict({adapter: torch.nn.Linear(RANK, WIDTH, bias=False)})

    def adapter_parameters(self):
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())


class ToyPeftModel(FakePeftModel):
    """A causal LM with LoRA on q_proj and v_proj, named the way PEFT names it.

    Adapter values are multiples of 1/4 in [-2, 2], exact in bfloat16, drawn in
    sorted-name order from `adapter_seed`, so a parameter's value follows its
    name, not where it was registered.  Base values are drawn from `base_seed`,
    or there are none: on the meta device the base has shapes but no data.  The
    base is frozen and the adapter trainable, as in a training load.
    """

    def __init__(self, adapter="default", adapter_seed=0, base_seed=0, base_device="cpu", vocab=11,
                 layer_order=None, with_adapter=True):
        super().__init__()

        def projection():
            if with_adapter:
                return LoraLinear(adapter, base_device)
            return torch.nn.Linear(WIDTH, WIDTH, bias=False, device=base_device)

        layers = torch.nn.ModuleDict()
        for index in range(LAYERS) if layer_order is None else layer_order:
            attention = torch.nn.Module()
            attention.q_proj = projection()
            attention.k_proj = torch.nn.Linear(WIDTH, WIDTH, bias=False, device=base_device)
            attention.v_proj = projection()
            layer = torch.nn.Module()
            layer.self_attn = attention
            layer.input_layernorm = torch.nn.RMSNorm(WIDTH, device=base_device)
            layers[str(index)] = layer
        decoder = torch.nn.Module()
        decoder.embed_tokens = torch.nn.Embedding(vocab, WIDTH, device=base_device)
        decoder.layers = layers
        decoder.norm = torch.nn.RMSNorm(WIDTH, device=base_device)
        causal_lm = torch.nn.Module()
        causal_lm.model = decoder
        causal_lm.lm_head = torch.nn.Linear(WIDTH, vocab, bias=False, device=base_device)
        self.base_model = torch.nn.Module()
        self.base_model.model = causal_lm

        adapter_ids = {id(p) for m in self.modules() if isinstance(m, LoraLinear) for p in m.adapter_parameters()}
        self.adapter_names = [name for name, p in self.named_parameters() if id(p) in adapter_ids]
        adapter_draw = torch.Generator().manual_seed(adapter_seed)
        base_draw = torch.Generator().manual_seed(base_seed)
        with torch.no_grad():
            for name, param in sorted(self.named_parameters(), key=lambda item: item[0]):
                if id(param) in adapter_ids:
                    param.copy_(torch.randint(-8, 9, param.shape, generator=adapter_draw) / 4)
                elif base_device != "meta":
                    param.copy_(torch.randn(param.shape, generator=base_draw))
                param.requires_grad_(id(param) in adapter_ids)


def recorder_keys(model):
    """The names the pinned GradientRecorder snapshots as theta, in its order: the keys of Delta theta."""
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "reference_gradient.pt"
        torch.save({}, reference)
        recorder = GradientRecorder(model, str(reference))
    recorder.before_update()
    return list(recorder.theta_before)


def adapter_tensors(model):
    """(name, tensor) for the toy's adapter, in sorted-name order."""
    values = dict(model.named_parameters())
    return [(name, values[name]) for name in sorted(model.adapter_names)]


class AdapterHashTests(unittest.TestCase):
    def test_the_hash_covers_exactly_the_tensors_the_gradient_recorder_tracks(self):
        """Changing one value changes the digest exactly when the recorder tracks that parameter.

        The recorder's theta, and so Delta theta and the names `h.json` binds, is
        the trainable LoRA-named parameters.  At e7eea0e every base weight was in
        the hash as well.
        """
        model = ToyPeftModel()
        tracked = recorder_keys(model)
        self.assertEqual(tracked, model.adapter_names)  # the recorder's rule finds the toy's LoRA tensors
        reference = adapter_parameter_hash(model)
        changed = []
        for name, param in model.named_parameters():
            saved = param.detach().clone()
            with torch.no_grad():
                param.view(-1)[0] += 1
            if adapter_parameter_hash(model) != reference:
                changed.append(name)
            with torch.no_grad():
                param.copy_(saved)
        self.assertEqual(adapter_parameter_hash(model), reference)  # every change was undone
        self.assertEqual(changed, tracked)

    def test_the_recorded_names_are_the_delta_theta_keys_in_order(self):
        """`lora_parameter_names` is the recorder's Delta theta keys in the recorder's order.

        `h.json` binds this list, so it has to be that order, which is model
        order; with twelve layers sorted order differs, so a sort shows here.
        """
        model = ToyPeftModel()
        tracked = recorder_keys(model)
        self.assertNotEqual(tracked, sorted(tracked))
        self.assertEqual(lora_parameter_names(model), tracked)

    def test_the_same_adapter_on_another_base_hashes_the_same(self):
        """Other base weights, even another vocabulary size, leave the digest alone."""
        self.assertEqual(adapter_parameter_hash(ToyPeftModel(base_seed=1, vocab=13)),
                         adapter_parameter_hash(ToyPeftModel()))

    def test_the_base_is_never_read(self):
        """A base with no data (meta device) hashes as a real one does: no base tensor is copied.

        At e7eea0e every base weight was copied to the CPU on each call.
        """
        meta = ToyPeftModel(base_device="meta")
        base = [p for name, p in meta.named_parameters() if name not in meta.adapter_names]
        self.assertTrue(base and all(p.is_meta for p in base))
        self.assertEqual(adapter_parameter_hash(meta), adapter_parameter_hash(ToyPeftModel()))

    def test_a_frozen_adapter_hashes_the_same(self):
        """Trainability is not part of the digest, so an inference load matches a training load."""
        model = ToyPeftModel()
        trainable = adapter_parameter_hash(model)
        for param in model.parameters():
            param.requires_grad_(False)
        self.assertEqual(adapter_parameter_hash(model), trainable)

    def test_precision_is_part_of_the_hash(self):
        """The same values in bfloat16 are another adapter: a step from them is another step."""
        model, half = ToyPeftModel(), ToyPeftModel().to(torch.bfloat16)
        for (name, value), (_, other) in zip(adapter_tensors(model), adapter_tensors(half)):
            self.assertEqual(other.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(other.double(), value.double()), name)  # only the dtype differs
        self.assertNotEqual(adapter_parameter_hash(half), adapter_parameter_hash(model))

    def test_the_names_are_part_of_the_hash(self):
        """The same tensors under other names, here another adapter name, are another adapter."""
        model, renamed = ToyPeftModel(), ToyPeftModel(adapter="other")
        pairs = list(zip(adapter_tensors(model), adapter_tensors(renamed)))
        self.assertEqual(len(pairs), len(model.adapter_names))
        for (name, value), (other_name, other) in pairs:
            self.assertNotEqual(name, other_name)
            self.assertTrue(torch.equal(value, other), name)  # only the names differ
        self.assertNotEqual(adapter_parameter_hash(renamed), adapter_parameter_hash(model))

    def test_registration_order_does_not_change_the_hash(self):
        """The same names and values registered in another order hash the same."""
        model, reordered = ToyPeftModel(), ToyPeftModel(layer_order=list(reversed(range(LAYERS))))
        names, other = [n for n, _ in model.named_parameters()], [n for n, _ in reordered.named_parameters()]
        self.assertEqual(sorted(names), sorted(other))
        self.assertNotEqual(names, other)
        for (name, value), (_, same) in zip(adapter_tensors(model), adapter_tensors(reordered)):
            self.assertTrue(torch.equal(value, same), name)
        self.assertEqual(adapter_parameter_hash(reordered), adapter_parameter_hash(model))

    def test_a_model_without_adapter_parameters_cannot_be_hashed(self):
        with self.assertRaisesRegex(ValueError, "no adapter"):
            adapter_parameter_hash(ToyPeftModel(with_adapter=False))


class RequireCommitIdTests(unittest.TestCase):
    """The revision written into the record, and compared at load time, must name one commit.

    The adapter hash leaves the base out, so `base_model` + `base_revision` are
    the only binding to the base.  An alias compares equal to itself while
    pointing at whatever commit the hub serves that day, which is a match that
    proves nothing, so the shape is checked instead.
    """

    def test_a_commit_id_is_accepted_as_it_is(self):
        self.assertEqual(require_commit_id(REVISION), REVISION)
        self.assertEqual(require_commit_id("0" * 40), "0" * 40)

    def test_an_alias_is_refused_even_when_the_other_side_says_the_same(self):
        """`"main" == "main"` is exactly the comparison this check exists to stop."""
        for alias in ("main", "MAIN", "refs/heads/main", "v1.0", "latest"):
            with self.assertRaisesRegex(ValueError, "40-character lowercase hex commit id"):
                require_commit_id(alias)

    def test_a_commit_id_in_another_case_or_with_whitespace_is_refused(self):
        """`resolve_revision` lowercases before its own check, so uppercase reaches the record as typed."""
        for value in (REVISION.upper(), "A" + REVISION[1:], REVISION[:8] + "A" + REVISION[9:],
                      REVISION + "\n", " " + REVISION, REVISION[:-1], REVISION + "0", ""):
            with self.assertRaisesRegex(ValueError, "40-character lowercase hex commit id"):
                require_commit_id(value)

    def test_a_non_hex_character_of_the_right_length_is_refused(self):
        for index in range(40):
            value = REVISION[:index] + "g" + REVISION[index + 1:]
            with self.assertRaisesRegex(ValueError, "40-character lowercase hex commit id"):
                require_commit_id(value)

    def test_a_missing_revision_raises_value_error_not_type_error(self):
        """A TypeError would escape the `except ValueError` that names the branch in one_step."""
        for value in (None, 40, b"a" * 40, ["a"] * 40):
            with self.assertRaises(ValueError) as caught:
                require_commit_id(value)
            self.assertNotIsInstance(caught.exception, TypeError)
            self.assertIn("not %r" % (value,), str(caught.exception))

    def test_the_message_names_what_was_checked(self):
        with self.assertRaisesRegex(ValueError, "the loaded base_revision"):
            require_commit_id("main", "the loaded base_revision")


class CreateSharedAdapterTests(unittest.TestCase):
    """The check runs before the backend loads the base, so a bad config downloads nothing."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name) / "shared_adapter"
        self.config = {"model": "toy/base", "revision": REVISION, "backend": "hf",
                       "training": {"lora_rank": RANK, "lora_alpha": 2 * RANK, "lora_dropout": 0.0,
                                    "target_modules": ["q_proj", "v_proj"]}}

    def create(self, backend=None, init=None):
        """Call create_shared_adapter with the backend and the adapter builder both watched."""
        backend = mock.MagicMock(side_effect=AssertionError("the backend was called")) if backend is None else backend
        init = mock.MagicMock(side_effect=AssertionError("the adapter was built")) if init is None else init
        with mock.patch("rsi.backends.backend", backend), \
                mock.patch("rsi.shared_adapter.init_shared_adapter", init):
            try:
                record = create_shared_adapter(dict(self.config), self.output, 0)
            except Exception as exc:  # noqa: BLE001 - the caller decides; this reports what happened
                return exc, backend, init
            return record, backend, init

    def test_an_unpinned_revision_is_refused_before_the_backend_is_called(self):
        self.config["revision"] = "main"
        exc, backend, init = self.create()
        self.assertIsInstance(exc, ValueError)
        self.assertRegex(str(exc), "40-character lowercase hex commit id")
        backend.assert_not_called()
        init.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_a_missing_revision_is_refused_before_the_backend_is_called(self):
        del self.config["revision"]
        exc, backend, init = self.create()
        self.assertIsInstance(exc, KeyError)  # config shape, not revision shape
        backend.assert_not_called()

    def test_a_pinned_revision_reaches_the_backend_and_is_recorded(self):
        """The check does not block the real path: a commit id goes straight through."""
        model = mock.MagicMock()
        wrapped = mock.MagicMock()
        record = dict(BASE, parameter_hash="ab" * 32, param_names=["a"], num_params=1, seed=0)
        with mock.patch("rsi.backends.backend", return_value=model) as backend, \
                mock.patch("rsi.shared_adapter.init_shared_adapter", return_value=(record, wrapped)) as init:
            created = create_shared_adapter(dict(self.config), self.output, 0)
        self.assertEqual(created["base_revision"], REVISION)
        self.assertEqual(backend.call_args.args[0]["revision"], REVISION)
        self.assertEqual(init.call_args.args[1]["revision"], REVISION)

    def test_init_shared_adapter_checks_the_revision_it_would_record(self):
        """`init_shared_adapter` is callable on its own, so it guards the value it writes.

        peft is not installed here, so this test supplies a stand-in for it: with
        the check in place the ValueError comes first, and without it the same
        config reaches the stand-in's adapter builder, which is an assertion.
        That keeps the red this test gives a property of the check and not of
        what happens to be installed on the machine.
        """
        config = dict(self.config, revision="main")
        built = mock.MagicMock(side_effect=AssertionError("the adapter was built"))
        with fake_peft(get_peft_model=built):
            with self.assertRaisesRegex(ValueError, "40-character lowercase hex commit id"):
                init_shared_adapter(torch.nn.Module(), config, 0)
        built.assert_not_called()


class VerifySharedAdapterTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.adapter_dir = Path(tmp.name)
        self.model = ToyPeftModel()

    def record(self, **fields):
        """A shared adapter record for the toy on BASE, with `fields` overriding it."""
        record = dict(BASE, parameter_hash=adapter_parameter_hash(self.model), seed=0)
        record.update(fields)
        return record

    def verify(self, record, loaded=BASE, model=None):
        """Write `record` as the shared adapter record, then verify a model loaded on `loaded`."""
        (self.adapter_dir / "shared_adapter.json").write_text(json.dumps(record))
        with fake_peft():
            return verify_shared_adapter(self.model if model is None else model, self.adapter_dir,
                                         loaded["base_model"], loaded["base_revision"])

    def test_the_record_is_returned_when_adapter_and_base_both_match(self):
        record = self.record()
        self.assertEqual(self.verify(record), record)

    def test_another_base_is_refused(self):
        """The adapter tensors match the record; only the base the caller loaded differs."""
        for key, other in (("base_model", "toy/other"), ("base_revision", "f" * 40)):
            with self.assertRaisesRegex(ValueError, key):
                self.verify(self.record(), dict(BASE, **{key: other}))

    def test_a_record_bound_to_no_base_is_refused(self):
        for key in BASE:
            record = self.record()
            del record[key]
            with self.assertRaisesRegex(ValueError, "no %s" % key):
                self.verify(record)

    def test_another_adapter_is_refused(self):
        other = adapter_parameter_hash(ToyPeftModel(adapter_seed=1))
        self.assertNotEqual(other, adapter_parameter_hash(self.model))
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.verify(self.record(parameter_hash=other))

    def test_a_model_that_is_not_a_peft_model_is_refused(self):
        """Even when its adapter tensors hash to the record's value."""
        bare = torch.nn.Module()
        bare.inner = self.model
        with self.assertRaisesRegex(ValueError, "PeftModel"):
            self.verify(self.record(parameter_hash=adapter_parameter_hash(bare)), model=bare)

    def test_two_aliases_that_spell_the_same_are_refused(self):
        """The case the commit-id rule exists for: `"main"` recorded and `"main"` loaded.

        Without the shape check these two compare equal and the verification
        returns the record, which is a pass that proves nothing about the base.
        """
        with self.assertRaisesRegex(ValueError, "40-character lowercase hex commit id"):
            self.verify(self.record(base_revision="main"),
                        loaded=dict(BASE, base_revision="main"))

    def test_a_loaded_revision_that_is_not_a_commit_id_is_refused(self):
        """`one_step` passes the backend's revision, which is `"main"` on an unpinned load."""
        for loaded in ("main", REVISION.upper(), REVISION + "\n", None):
            with self.assertRaisesRegex(ValueError, "40-character lowercase hex commit id"):
                self.verify(self.record(), loaded=dict(BASE, base_revision=loaded))

    def test_a_record_holding_an_alias_is_refused_even_when_it_matches(self):
        """A record written before this rule, or edited by hand, is caught at load time."""
        with self.assertRaisesRegex(ValueError, "the recorded base_revision"):
            self.verify(self.record(base_revision="main"))

    def test_a_record_with_no_revision_is_still_refused_for_that_reason(self):
        """The missing-key message stays the one the older test expects."""
        record = self.record()
        del record["base_revision"]
        with self.assertRaisesRegex(ValueError, "no base_revision"):
            self.verify(record)


if __name__ == "__main__":
    unittest.main()
