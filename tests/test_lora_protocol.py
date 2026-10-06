"""LoRA settings come from the config, are stated in every shipped config, and are validated.

The LoRA values in an earlier write-up disagreed with code defaults in `rsi/common.py` and with a
seven-module list hardcoded in `rsi/backends.py`.  The guard is that every
shipped config states its LoRA fields and that `rsi/backends.py` takes them from
the config; the values themselves are each config's own and are not pinned here
(configs/README.md).

These tests do not require torch/peft/transformers: they exercise config
resolution and the adapter-construction argument path with a stub.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi.backends import lora_config_from_training
from rsi.common import load_config

# The study's current values, used as examples of a valid setting.
FROZEN_RANK = 8
FROZEN_ALPHA = 16
FROZEN_DROPOUT = 0.0
FROZEN_TARGETS = ["q_proj", "v_proj"]


class LoraConfigTests(unittest.TestCase):
    def test_shipped_configs_state_their_lora_values_explicitly(self):
        """Every shipped config writes out the four LoRA fields, so no run takes them silently from
        DEFAULTS.  Their values are each config's own (configs/README.md): nothing pins them."""
        configs_dir = Path(__file__).resolve().parent.parent / "configs"
        # matched_evaluation.json holds only the evaluation splits and batch size; it loads no model.
        found = sorted(p for p in configs_dir.glob("*.json") if p.name != "matched_evaluation.json")
        self.assertTrue(found, f"no configs in {configs_dir}")
        for path in found:
            training = json.loads(path.read_text(encoding="utf-8")).get("training", {})
            for key in ("lora_rank", "lora_alpha", "lora_dropout", "target_modules"):
                self.assertIn(key, training, f"{path.name} does not state training.{key} explicitly")

    def test_legacy_field_name_is_normalised(self):
        """Older configs used `lora_target_modules`; it must map onto the frozen name."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps({"training": {"lora_target_modules": FROZEN_TARGETS}}),
                            encoding="utf-8")
            training = load_config(path)["training"]
        self.assertEqual(training["target_modules"], FROZEN_TARGETS)
        self.assertNotIn("lora_target_modules", training)

    def test_conflicting_field_names_are_rejected(self):
        """Both names present is a protocol fork and must fail loudly, not pick a winner."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conflict.json"
            path.write_text(json.dumps({"training": {"target_modules": ["q_proj"],
                                                     "lora_target_modules": FROZEN_TARGETS}}),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


class ConfigValidationTests(unittest.TestCase):
    def _load(self, training_overrides):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({"training": training_overrides}), encoding="utf-8")
            return load_config(path)

    def test_rejects_empty_target_modules(self):
        with self.assertRaises(ValueError):
            self._load({"target_modules": []})

    def test_rejects_non_positive_rank_and_alpha(self):
        for key in ("lora_rank", "lora_alpha"):
            with self.assertRaises(ValueError):
                self._load({key: 0})

    def test_rejects_out_of_range_dropout(self):
        for value in (-0.1, 1.0, 2.0):
            with self.assertRaises(ValueError):
                self._load({"lora_dropout": value})

    def test_rejects_non_string_or_empty_module_names(self):
        for value in ([1], [""], ["q_proj", ""], ""):
            with self.assertRaises(ValueError):
                self._load({"target_modules": value})

    def test_accepts_explicit_frozen_protocol(self):
        config = self._load({
            "lora_rank": FROZEN_RANK,
            "lora_alpha": FROZEN_ALPHA,
            "lora_dropout": FROZEN_DROPOUT,
            "target_modules": FROZEN_TARGETS,
        })
        self.assertEqual(config["training"]["target_modules"], FROZEN_TARGETS)


class AdapterConstructionTests(unittest.TestCase):
    """backends.py must pass config-derived values to LoraConfig, not literals."""

    def _captured_lora_kwargs(self, training_overrides):
        """Run the protocol's LoRA-config builder with a stubbed PEFT and capture its args."""
        captured = {}

        class StubLoraConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({"training": training_overrides}), encoding="utf-8")
            config = load_config(path)

        import types
        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig = StubLoraConfig
        with patch.dict("sys.modules", {"peft": fake_peft}):
            lora_config_from_training(config["training"])
        return captured

    def test_backends_passes_the_config_values_to_lora_config(self):
        """Values other than the defaults reach LoraConfig unchanged: they come from the config."""
        captured = self._captured_lora_kwargs({"lora_rank": 4, "lora_alpha": 12, "lora_dropout": 0.1,
                                               "target_modules": ["k_proj", "o_proj"]})
        self.assertEqual((captured.get("r"), captured.get("lora_alpha"), captured.get("lora_dropout")), (4, 12, 0.1))
        self.assertEqual(list(captured.get("target_modules", [])), ["k_proj", "o_proj"])

    def test_no_seven_module_hardcode_remains(self):
        captured = self._captured_lora_kwargs({})
        targets = set(captured.get("target_modules", []))
        self.assertNotEqual(
            targets,
            {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
            "backends.py still hardcodes the seven-module list")

    def test_use_rslora_is_disabled(self):
        """The frozen convention is plain alpha/r scaling, not the rsqrt variant."""
        captured = self._captured_lora_kwargs({})
        self.assertIs(captured.get("use_rslora"), False)

    def test_train_builds_adapter_via_the_shared_helper(self):
        """Guards against train() re-inlining a hardcoded LoraConfig again."""
        import inspect
        from rsi import backends
        source = inspect.getsource(backends.HFBackend.train)
        self.assertIn("lora_config_from_training", source,
                      "train() no longer routes through the shared protocol helper")
        self.assertNotIn("target_modules=", source,
                         "train() hardcodes target_modules again")
        self.assertNotIn("lora_alpha=", source,
                         "train() hardcodes lora_alpha again")


if __name__ == "__main__":
    unittest.main()
