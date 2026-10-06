"""a-3 tests: the frozen base pin, checked against the resolved configuration.

The gap these close is a silent one.  Every entry reached the hub through
`pin_config`, which turns whatever the config says into a resolved revision, and
nothing compared the result with the value `HANDOFF.md:13` freezes.  A config
carrying a legal 40-hex id for a *different* commit resolved without complaint,
and every record still looked pinned: the revision is a commit id, the adapter
record agrees with the adapter, and the parameter hash says nothing about the
base.  A whole run could train on the wrong base and no artefact would say so.

What each group here does:

  1. `ReadTests` -- the pin file is the anchor, so a malformed one must stop the
     run rather than let it continue unchecked.  An alias, a truncated id or a
     non-string cannot serve as an anchor.
  2. `CheckTests` -- the comparison is exact and on the **resolved** revision;
     there is no `strip()`, no `lower()`, no alias handling.
  3. `EntryTests` -- each of the three entries that can reach a real base runs
     the check *before* it loads anything, on the hf path only, and mock keeps
     its current behaviour (the pin file is not even read).
  4. `RefusalTests` -- the controlled-resolve rejection case the review asks
     for: a full legal 40-hex revision that **resolves successfully** and is
     still not the anchor.  No download failure, no parse failure, no ellipsis.
  5. `CliTests` -- the same refusal through the real command line, in a real
     process, with the real `pin_config` and the repository's own pin file, so
     it is not an artefact of the test harness patching `main`.

CPU-only by construction: no transformers, no peft, no download, no real HF
revision.  The synthetic id below is a fixture, not a model revision.

Where a check needs a dataset, the fixture generates a real one with
`generate_data.generate` (0.04 s): a fake directory would let the pin check pass
because `verify_dataset` failed first, which proves nothing about the pin.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from rsi.shared_adapter import lora_protocol

# An exported RSI_BASE_PIN must not leak into these tests (tests/pin_isolation.py).
from pin_isolation import isolate as setUpModule, restore as tearDownModule  # noqa: F401

REPO = Path(__file__).resolve().parents[1]

# A full, legal, lowercase 40-hex string.  It is **not** a real Hugging Face
# revision and is never presented as one: it exists only so a test can hand the
# controlled resolver a value that passes every syntactic gate and still is not
# the anchor.
SYNTHETIC = "0123456789abcdef0123456789abcdef01234567"

ANCHOR = json.loads((REPO / "matched-dynamics" / "base_pin.json").read_text())["revision"]

# What an hf run of sample_candidates.py must also name (science-aux 5d124fd6):
# the split, and the study and phase that fix it.  A main run on train_001 is
# the pairing these pin tests use; the split rule itself is tested in
# tests/test_sample_candidates.py.
P_HF_FLAGS = ["--split", "train_001", "--study-id", "pin-test", "--phase", "main"]

CONFIG = {
    "model": "Qwen/Qwen3-1.7B",
    "revision": SYNTHETIC,
    "backend": "hf",
    "dtype": "bfloat16",
    "device": "cuda",
    "rounds": 2,
    # Every field the hf matched stages read is stated (rsi.common.require_explicit).
    "generation": {"candidates": 4, "prompts_per_pool": None, "batch_size": 8, "max_new_tokens": 2048,
                   "max_sequence_length": 4096, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_stop": None},
    "training": {"lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                 "target_modules": ["q_proj", "v_proj"], "examples": 64},
    "verifier": {"kind": "persistent"},
    "audit": {"policy": "none", "budget": 0, "weighting": True},
}

SITECUSTOMIZE = (
    "import rsi.base_pin as _b\n"
    "from pathlib import Path as _P\n"
    "_b.BASE_PIN = _P(r'%s')\n"
)


def write_config(path, **overrides):
    config = dict(CONFIG, **overrides)
    Path(path).write_text(json.dumps(config), encoding="utf-8")
    return Path(path)


class PinFixture(unittest.TestCase):
    """A temp directory with a config, a dataset, and usually a pin file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = write_config(self.tmp / "config.json")

    def dataset(self, root=None):
        """A real dataset: `verify_dataset` must pass for the entry to reach the step under test."""
        from generate_data import generate

        root = Path(root or self.tmp / "data")
        generate(root, "arithmetic", 7, rounds=1, per_round=2, dev=1, calibration=1, eval_id=1,
                 eval_ood=1, gradient_reference=3)
        return root

    def write_pin(self, name="base_pin.json", **pin):
        values = {"model": "Qwen/Qwen3-1.7B", "revision": ANCHOR,
                  "source": "test", "prepared": "test"}
        values.update(pin)
        path = self.tmp / name
        path.write_text(json.dumps(values), encoding="utf-8")
        return path

    def pinned(self, revision=SYNTHETIC):
        """A `pin_config` seam returning a resolved revision, without the network."""
        return mock.patch("rsi.experiment.pin_config", lambda c: dict(c, revision=revision))

    def h_args(self, out=None, data=None):
        return Namespace(config=str(self.config), data=str(data or self.tmp / "data"),
                         shared_adapter=str(self.tmp / "adapter"), out=str(out or self.tmp / "h"),
                         label_source="reference_answer", normalize=False, seed=0, backend=None)

    def run_entry(self, script, argv, pin_path=None):
        """Run a module-level entry for real, in a subprocess.

        `make_shared_adapter.py` has no `main(args)`; its body is under
        `if __name__ == "__main__"`.  `sample_candidates.py` now has
        `main(argv)`, and is still run this way here.  Running them as
        subprocesses is both the simplest way to exercise that body and the
        honest one -- it is the same path the run package's commands take, and
        it catches what a helper calling `main()` structurally cannot see
        (tests/test_reference_encode.py documents that lesson in
        `test_only_main_calls_main`).

        A `sitecustomize.py` on `PYTHONPATH` redirects the pin file before the
        entry imports anything, so the repository's own pin is never at risk and
        nothing in the repo is touched by a test.
        """
        (self.tmp / "sitecustomize.py").write_text(
            SITECUSTOMIZE % (pin_path if pin_path is not None else self.tmp / "base_pin.json"),
            encoding="utf-8")
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8", PYTHONPATH=os.pathsep.join([str(self.tmp), str(REPO)]))
        command = [sys.executable, str(REPO / script)] + [str(a) for a in argv]
        return subprocess.run(command, cwd=str(REPO), capture_output=True, text=True,
                              env=env, timeout=300)


class ReadTests(PinFixture):
    """The pin file is the anchor: malformed means stop, never skip."""

    def test_the_committed_pin_is_the_handoff_value_verbatim(self):
        """The committed file, not a copy: HANDOFF.md:13's model and revision."""
        handoff = (REPO / "matched-dynamics" / "HANDOFF.md").read_text(encoding="utf-8")
        pin = json.loads((REPO / "matched-dynamics" / "base_pin.json").read_text(encoding="utf-8"))
        lines = [text for text in handoff.splitlines() if "model pin:" in text.lower()]
        self.assertEqual(len(lines), 1, "HANDOFF.md no longer has exactly one model pin line")
        self.assertIn("`%s`" % pin["model"], lines[0])
        self.assertIn("`%s`" % pin["revision"], lines[0])
        self.assertEqual(pin["revision"], ANCHOR)

    def test_reading_the_committed_pin_returns_it(self):
        from rsi.base_pin import read_base_pin

        pin = read_base_pin(REPO / "matched-dynamics" / "base_pin.json")
        self.assertEqual(pin["model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(pin["revision"], ANCHOR)

    def test_a_missing_pin_file_is_an_error_not_a_skip(self):
        from rsi.base_pin import read_base_pin

        with self.assertRaises(ValueError) as caught:
            read_base_pin(self.tmp / "absent.json")
        self.assertIn("does not exist", str(caught.exception))

    def test_a_malformed_pin_file_is_an_error(self):
        from rsi.base_pin import read_base_pin

        for body in ("{not json", "[1, 2]", '"a string"'):
            path = self.tmp / "bad.json"
            path.write_text(body, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_base_pin(path)

    def test_a_pin_missing_a_required_key_is_an_error(self):
        from rsi.base_pin import read_base_pin

        for missing in ("model", "revision"):
            pin = {"model": "Qwen/Qwen3-1.7B", "revision": ANCHOR}
            del pin[missing]
            path = self.tmp / ("missing_%s.json" % missing)
            path.write_text(json.dumps(pin), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                read_base_pin(path)
            self.assertIn(missing, str(caught.exception))

    def test_a_non_string_revision_cannot_serve_as_an_anchor(self):
        from rsi.base_pin import read_base_pin

        with self.assertRaises(ValueError):
            read_base_pin(self.write_pin(revision=42))

    def test_an_alias_cannot_serve_as_an_anchor(self):
        """'main' is exactly the value the pin exists to replace."""
        from rsi.base_pin import read_base_pin

        with self.assertRaises(ValueError) as caught:
            read_base_pin(self.write_pin(revision="main"))
        self.assertIn("commit id", str(caught.exception))

    def test_a_truncated_id_cannot_serve_as_an_anchor(self):
        from rsi.base_pin import read_base_pin

        with self.assertRaises(ValueError):
            read_base_pin(self.write_pin(revision=ANCHOR[:39]))

    def test_a_non_string_model_cannot_serve_as_an_anchor(self):
        from rsi.base_pin import read_base_pin

        with self.assertRaises(ValueError):
            read_base_pin(self.write_pin(model=None))

    def test_extra_keys_are_tolerated(self):
        """`source` and `prepared` are provenance; a later note must not break it."""
        from rsi.base_pin import read_base_pin

        pin = read_base_pin(self.write_pin(extra="value", note="anything"))
        self.assertEqual(pin["extra"], "value")


class CheckTests(PinFixture):
    """The comparison: exact, on the resolved value, naming both sides."""

    def setUp(self):
        super().setUp()
        self.pin_path = self.write_pin()

    def check(self, **overrides):
        from rsi.base_pin import check_base_pin

        config = {"model": "Qwen/Qwen3-1.7B", "revision": ANCHOR, "backend": "hf"}
        config.update(overrides)
        return check_base_pin(config, path=self.pin_path)

    def test_an_equal_resolved_revision_returns_the_binding(self):
        from rsi.common import file_hash

        binding = self.check()
        self.assertEqual(binding["model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(binding["pin_revision"], ANCHOR)
        self.assertEqual(binding["resolved_revision"], ANCHOR)
        self.assertEqual(binding["pin_path"], str(self.pin_path))
        self.assertEqual(binding["pin_sha256"], file_hash(self.pin_path))

    def test_the_binding_records_both_sides_of_the_comparison(self):
        binding = self.check()
        self.assertEqual(binding["pin_revision"], binding["resolved_revision"])
        self.assertNotIn("sha256", binding, "the binding must not carry a self-hash")

    def test_a_different_legal_commit_is_refused_and_names_both(self):
        with self.assertRaises(ValueError) as caught:
            self.check(revision=SYNTHETIC)
        message = str(caught.exception)
        self.assertIn(ANCHOR, message)
        self.assertIn(SYNTHETIC, message)

    def test_an_uppercase_resolved_revision_is_refused_not_normalized(self):
        """`require_commit_id` is authoritative; a-3 adds no case folding."""
        with self.assertRaises(ValueError) as caught:
            self.check(revision=ANCHOR.upper())
        self.assertIn("commit id", str(caught.exception))

    def test_an_alias_as_the_resolved_revision_is_refused(self):
        with self.assertRaises(ValueError):
            self.check(revision="main")

    def test_whitespace_around_the_resolved_revision_is_refused(self):
        with self.assertRaises(ValueError):
            self.check(revision=" %s " % ANCHOR)

    def test_a_different_model_with_the_same_revision_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.check(model="Qwen/Qwen3-1.7B-Other")
        self.assertIn("base model", str(caught.exception))

    def test_a_missing_resolution_is_refused(self):
        """`config['revision']` absent is a refusal, not a pass by default."""
        with self.assertRaises(ValueError):
            self.check(revision=None)


class EntryTests(PinFixture):
    """Each entry checks before it loads, on hf only, and mock is untouched."""

    # -- compute_reference_gradient (H) -------------------------------------
    def test_h_refuses_a_resolved_base_that_is_not_the_anchor(self):
        import compute_reference_gradient as entry

        with self.pinned(), mock.patch("rsi.base_pin.BASE_PIN", self.write_pin()), \
                mock.patch("rsi.backends.backend") as load:
            with self.assertRaises(SystemExit) as caught:
                entry.main(self.h_args())
        self.assertIn("base pin mismatch", str(caught.exception))
        load.assert_not_called()

    def test_h_refuses_before_it_verifies_the_dataset(self):
        """The check sits before the data check, so the failure is 'wrong base'."""
        import compute_reference_gradient as entry

        with self.pinned(), mock.patch("rsi.base_pin.BASE_PIN", self.write_pin()), \
                mock.patch("compute_reference_gradient.verify_dataset") as verify, \
                mock.patch("rsi.backends.backend") as load:
            with self.assertRaises(SystemExit) as caught:
                entry.main(self.h_args())
        self.assertIn("base pin mismatch", str(caught.exception))
        verify.assert_not_called()
        load.assert_not_called()

    def test_h_passes_the_anchor_and_continues(self):
        import compute_reference_gradient as entry

        landed = self.tmp / "h"
        with self.pinned(revision=ANCHOR), mock.patch("rsi.base_pin.BASE_PIN", self.write_pin()), \
                mock.patch("compute_reference_gradient.verify_dataset", side_effect=RuntimeError("reached the dataset")), \
                mock.patch("rsi.backends.backend") as load:
            with self.assertRaises(RuntimeError) as caught:
                entry.main(self.h_args(out=landed))
        self.assertEqual(str(caught.exception), "reached the dataset",
                         "the anchor should pass and the run continue to the dataset")
        load.assert_not_called()
        self.assertFalse(landed.exists(), "nothing is written before the base is known")

    def test_h_reads_the_pin_exactly_once_per_run(self):
        from rsi import base_pin

        real_read = base_pin.read_base_pin
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return real_read(*args, **kwargs)

        import compute_reference_gradient as entry

        with self.pinned(revision=ANCHOR), mock.patch("rsi.base_pin.BASE_PIN", self.write_pin()), \
                mock.patch("rsi.base_pin.read_base_pin", counted), \
                mock.patch("compute_reference_gradient.verify_dataset", side_effect=RuntimeError("stop")):
            with self.assertRaises(RuntimeError):
                entry.main(self.h_args())
        self.assertEqual(len(calls), 1)

    def test_h_records_the_pin_binding_in_the_run_record(self):
        """The binding reaches `write_reference_gradient`'s bindings dict.

        a-2's `EntryBindingTests.test_main_binds_the_config_the_data_and_the_initialisation`
        proves this dict is the one written into `h.json`; here the pin half of
        it is pinned down, so a dropped `base_pin` cannot pass unnoticed.
        """
        import compute_reference_gradient as entry

        data = self.dataset()
        pin_path = self.write_pin()
        stub = mock.Mock(model_name="fixture/base", revision=ANCHOR, device="cpu")
        stub.close = mock.Mock()
        captured = {}

        def capture(backend, pairs, out, seed, bindings, label_source):
            captured.update(bindings)
            # Only the shape `main` prints on the way out; the real writer's own
            # record is a-2's subject, not this test's.
            return {"h": {"parameter_names": [], "l2_norm": 0.0}, "examples": len(pairs)}

        with self.pinned(revision=ANCHOR), mock.patch("rsi.base_pin.BASE_PIN", pin_path), \
                mock.patch("rsi.backends.backend", return_value=stub), \
                mock.patch("rsi.shared_adapter.verify_shared_adapter",
                           return_value={"parameter_hash": "fixture-hash",
                                         "protocol": lora_protocol(CONFIG["training"])}), \
                mock.patch("compute_reference_gradient.write_reference_gradient", capture):
            entry.main(self.h_args(out=self.tmp / "h", data=data))

        from rsi.common import file_hash

        binding = captured["base_pin"]
        self.assertEqual(binding["pin_path"], str(pin_path))
        self.assertEqual(binding["pin_sha256"], file_hash(pin_path))
        self.assertEqual(binding["model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(binding["pin_revision"], ANCHOR)
        self.assertEqual(binding["resolved_revision"], ANCHOR)

    # -- sample_candidates (P) and make_shared_adapter (A) -------------------
    def test_sample_candidates_refuses_a_resolved_base_that_is_not_the_anchor(self):
        data = self.dataset()
        self.write_pin()
        write_config(self.config, revision=SYNTHETIC)
        result = self.run_entry("sample_candidates.py",
                                ["--config", self.config, "--data", data,
                                 "--out", self.tmp / "pool.jsonl"] + P_HF_FLAGS)
        output = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("base pin mismatch", output)
        self.assertIn(ANCHOR, output)
        self.assertIn(SYNTHETIC, output)
        self.assertFalse((self.tmp / "pool.jsonl").exists(), "a pool is not written for another base")

    def test_sample_candidates_on_mock_does_not_read_the_pin(self):
        """The pin file is not even visited on a mock run, so mock is unchanged.

        Discriminating, not self-confirming: the dataset is real, so
        `verify_dataset` passes and the entry reaches the pin check.  The pin
        path points at a file that does not exist, so *any* visit would raise
        "base pin ... does not exist" instead of writing the pool.
        """
        data = self.dataset()
        absent = self.tmp / "never_written.json"
        write_config(self.config, revision="main")
        result = self.run_entry("sample_candidates.py",
                                ["--config", self.config, "--data", data,
                                 "--out", self.tmp / "pool.jsonl", "--backend", "mock"],
                                pin_path=absent)
        output = result.stderr + result.stdout
        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("base pin", output, "a mock run must not consult the pin")
        self.assertFalse(absent.exists(), "the pin file was read on a mock run")
        self.assertTrue((self.tmp / "pool.jsonl").exists(), "the mock pool is written as before")

    def test_sample_candidates_refuses_a_wrong_base_before_it_verifies_the_files(self):
        """The refusal is "wrong base", not "bad manifest".

        Discriminating against the order the entry happens to have now
        (`verify_dataset` first, then the pin): the dataset directory here has no
        manifest at all, so a run that verified first would say "manifest.json"
        and this reads "base pin mismatch" instead.
        """
        self.write_pin()
        write_config(self.config, revision=SYNTHETIC)
        absent = self.tmp / "no_such_dataset"
        result = self.run_entry("sample_candidates.py",
                                ["--config", self.config, "--data", absent,
                                 "--out", self.tmp / "pool.jsonl"] + P_HF_FLAGS)
        output = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("base pin mismatch", output)
        self.assertNotIn("manifest", output, "the base is checked before the files are verified")

    def test_sample_candidates_refuses_a_missing_pin_before_it_verifies_the_files(self):
        """Same shape for the reading side: an absent pin stops the run at once.

        The pin path does not exist, so a run that read the files first would
        report the manifest; this one reports the pin.
        """
        absent_data = self.tmp / "no_such_dataset"
        result = self.run_entry("sample_candidates.py",
                                ["--config", self.config, "--data", absent_data,
                                 "--out", self.tmp / "pool.jsonl"] + P_HF_FLAGS,
                                pin_path=self.tmp / "never_written.json")
        output = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not exist", output)
        self.assertNotIn("manifest", output)

    def test_make_shared_adapter_refuses_a_resolved_base_that_is_not_the_anchor(self):
        self.write_pin()
        write_config(self.config, revision=SYNTHETIC)
        result = self.run_entry("make_shared_adapter.py",
                                ["--config", self.config, "--out", self.tmp / "adapter"])
        output = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("base pin mismatch", output)
        self.assertIn(ANCHOR, output)
        self.assertFalse((self.tmp / "adapter").exists())

    def test_make_shared_adapter_on_mock_keeps_its_own_refusal_and_ignores_the_pin(self):
        """Mock is refused by the pre-existing gate, and the pin is never read.

        The pin check sits after that gate on purpose: on mock no base is used,
        so mock's message and behaviour stay exactly what they were.
        """
        absent = self.tmp / "never_written.json"
        write_config(self.config, revision=SYNTHETIC)
        result = self.run_entry("make_shared_adapter.py",
                                ["--config", self.config, "--out", self.tmp / "adapter",
                                 "--backend", "mock"], pin_path=absent)
        output = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be initialised from a real backend", output, output)
        self.assertNotIn("base pin", output, "a mock run must not consult the pin")
        self.assertFalse(absent.exists(), "the pin file was read on a mock run")
        self.assertFalse((self.tmp / "adapter").exists())


class RefusalTests(PinFixture):
    """The review's case: resolves successfully, is legal, and is still not the anchor."""

    def test_a_legal_revision_that_resolves_but_is_not_the_anchor_is_refused(self):
        """Controlled resolve: `resolve_revision` returns a full legal 40-hex id.

        `pin_config` is the entry's seam; here the resolver under it is
        controlled, so the value it produces is syntactically perfect.  The
        refusal therefore cannot be blamed on a download, a parse failure, an
        alias or an ellipsis -- the only thing wrong with the value is that it is
        not the anchor.
        """
        import compute_reference_gradient as entry

        asked, returned = [], []

        def resolve(model, revision):
            asked.append((model, revision))
            returned.append(SYNTHETIC)
            return SYNTHETIC

        pin_path = self.write_pin()
        with mock.patch("rsi.experiment.resolve_revision", resolve), \
                mock.patch("rsi.base_pin.BASE_PIN", pin_path), \
                mock.patch("rsi.backends.backend") as load, \
                mock.patch("rsi.shared_adapter.verify_shared_adapter") as verify:
            with self.assertRaises(SystemExit) as caught:
                entry.main(self.h_args())

        # (i) the controlled resolve ran, was asked for the configured revision,
        #     and *returned* a full legal id: the refusal is not a resolution
        #     failure, and the value compared is the resolved one.
        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0][1], SYNTHETIC)
        self.assertEqual(returned, [SYNTHETIC], "the resolver's return value is what must be compared")
        self.assertNotEqual(SYNTHETIC, ANCHOR)
        # (ii) the refusal is the mismatch, and both values are named.
        message = str(caught.exception)
        self.assertIn("base pin mismatch", message)
        self.assertIn(ANCHOR, message)
        self.assertIn(SYNTHETIC, message)
        # (iii) nothing was loaded: no forward, no backward, no adapter read.
        load.assert_not_called()
        verify.assert_not_called()

    def test_the_anchor_is_not_taken_from_the_resolved_config(self):
        """Change the anchor, and the same legal resolve is still refused.

        If the pin were backfilled from (or compared against) the resolved value,
        this would pass.  It is refused, which is what "prepared before preflight,
        never from the resolved config" means in behaviour.
        """
        import compute_reference_gradient as entry

        other = "f" * 40
        with mock.patch("rsi.experiment.pin_config", lambda c: dict(c, revision=SYNTHETIC)), \
                mock.patch("rsi.base_pin.BASE_PIN", self.write_pin(revision=other)), \
                mock.patch("rsi.backends.backend") as load:
            with self.assertRaises(SystemExit) as caught:
                entry.main(self.h_args())
        message = str(caught.exception)
        self.assertIn("base pin mismatch", message)
        self.assertIn(other, message)
        self.assertIn(SYNTHETIC, message)
        load.assert_not_called()

    def test_the_same_refusal_for_the_adapter_entry(self):
        """The adapter entry refuses for the same reason, not a different one."""
        self.write_pin()
        write_config(self.config, revision=SYNTHETIC)
        result = self.run_entry("make_shared_adapter.py",
                                ["--config", self.config, "--out", self.tmp / "adapter"])
        output = result.stderr + result.stdout
        self.assertIn("base pin mismatch", output)
        self.assertIn(ANCHOR, output)
        self.assertIn(SYNTHETIC, output)


class CliTests(PinFixture):
    """The same refusal through the real command line, with the real `pin_config`.

    No `main` stub and no resolve stub.  `resolve_revision` returns a 40-hex
    revision without touching the network (rsi/backends.py:12-18), so a full id
    in the config reaches the pin check on this offline box by the same path a
    real run takes.
    """

    def run_cli(self, script, revision, data, extra=()):
        write_config(self.config, revision=revision)
        command = [sys.executable, str(REPO / script), "--config", str(self.config),
                   "--data", str(data), "--out", str(self.tmp / "out")] + [str(a) for a in extra]
        # CUDA hidden: on a GPU node whose cache holds the anchor checkpoint the positive
        # control would otherwise load the model and succeed instead of stopping in the backend.
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                   CUDA_VISIBLE_DEVICES="")
        return subprocess.run(command, cwd=str(REPO), capture_output=True, text=True,
                              env=env, timeout=300)

    def test_sample_candidates_cli_refuses_a_legal_revision_that_is_not_the_anchor(self):
        """The repository's own pin, and a real 40-hex config value."""
        data = self.dataset()
        result = self.run_cli("sample_candidates.py", SYNTHETIC, data, extra=P_HF_FLAGS)
        output = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("base pin mismatch", output)
        self.assertIn(ANCHOR, output)
        self.assertIn(SYNTHETIC, output)
        self.assertFalse((self.tmp / "out").exists())

    def test_sample_candidates_cli_passes_the_anchor_and_stops_later(self):
        """Positive control: the pin does not refuse a run that is on the anchor.

        This box has no transformers, so the run stops at the model load; a GPU
        box, with CUDA hidden from the run, stops at the backend's CUDA check
        whether or not the checkpoint is cached.  Either way it is past the pin, which is what
        this asserts -- without it, the refusal test above could be passing on a
        check that refuses everything.
        """
        data = self.dataset()
        result = self.run_cli("sample_candidates.py", ANCHOR, data, extra=P_HF_FLAGS)
        output = result.stderr + result.stdout
        self.assertNotIn("base pin", output, "the anchor must not be refused by the pin")
        self.assertNotEqual(result.returncode, 0, "the run still cannot load a real model here")
        # Past the pin, the split check and the data: it stopped in the model
        # load (rsi/backends.py), not at a refusal that writes nothing.
        self.assertIn("backends.py", output, "the run did not reach the model load")
        self.assertNotIn("Nothing was written", output)
        self.assertFalse((self.tmp / "out").exists())

    def test_the_scripts_reject_an_unknown_option_without_reaching_the_pin(self):
        """Control: a bad command line fails in argparse, not in the pin check."""
        result = self.run_cli("sample_candidates.py", ANCHOR, self.tmp, extra=["--nonsense"])
        output = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("base pin mismatch", output)


if __name__ == "__main__":
    unittest.main()
