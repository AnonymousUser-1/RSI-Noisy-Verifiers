"""The matched-line configs: complete, and each on its own model's pin.

Every value a matched-line run uses comes from its config file: on hf each stage
refuses a file that leaves a field it reads to DEFAULTS (rsi.common.STAGE_FIELDS,
require_explicit).  The values themselves are each config's own -- sampling,
max tokens, learning rates, later-round pool size -- and nothing here pins them
(configs/README.md lists every field, the stage that reads it and its values).
"""
import json
import unittest
from pathlib import Path

from rsi.base_pin import BASE_PIN, read_base_pin
from rsi.common import load_config, require_explicit

# An exported RSI_BASE_PIN must not leak into these tests (tests/pin_isolation.py).
from pin_isolation import isolate as setUpModule, restore as tearDownModule  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
# Which stages read which config (scripts/multiround: 03/04/07 POOL_CONFIG, 05/06 ITER_CONFIG,
# 09 AUDIT_CONFIG).
POOL_STAGES = ("pool", "adapter", "reference", "one_step")
ITERATIVE_STAGES = ("pool", "adapter", "reference", "iterative")
KINDS = ("matched_pool", "matched_iterative_audit", "matched_iterative")


def model_tag(path):
    """'' for Qwen3-1.7B, else e.g. '_llama3.2-3b'; the audit variants belong to their model."""
    kind = next(k for k in KINDS if path.stem.startswith(k))
    tag = path.stem[len(kind):]
    return tag[len("_audit"):] if tag.startswith("_audit") else tag


def matched_configs():
    found = sorted(p for p in CONFIGS.glob("matched_*.json") if p.name != "matched_evaluation.json")
    assert found, "no matched configs"
    return found


def pin_for(path):
    tag = model_tag(path)
    return BASE_PIN if not tag else REPO / "configs" / "pins" / ("base_pin%s.json" % tag)


class MatchedConfigTests(unittest.TestCase):
    def test_every_matched_config_states_every_field_its_stages_read(self):
        for path in matched_configs():
            stages = ITERATIVE_STAGES if path.stem.startswith("matched_iterative") else POOL_STAGES
            for stage in stages:
                with self.subTest(config=path.name, stage=stage):
                    require_explicit(path, stage)

    def test_every_matched_config_resolves_and_names_its_pin(self):
        for path in matched_configs():
            with self.subTest(config=path.name):
                config = load_config(path)
                pin = read_base_pin(pin_for(path))
                self.assertEqual((config["model"], config["revision"]), (pin["model"], pin["revision"]))

    def test_a_config_that_leaves_a_read_field_to_defaults_is_refused(self):
        import tempfile
        raw = json.loads((CONFIGS / "matched_pool.json").read_text())
        del raw["generation"]["max_new_tokens"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError) as refused:
                require_explicit(path, "pool")
        self.assertIn("generation.max_new_tokens", str(refused.exception))

    def test_a_model_s_configs_share_its_base_and_lora(self):
        """The shared adapter is made once, from the iterative config, and every run starts from it.
        The R/S selection settings are shared too: the pilot reads the pool config's, the runs the
        iterative and audit configs'."""
        lora = ("lora_rank", "lora_alpha", "lora_dropout", "target_modules")
        for path in matched_configs():
            iterative = CONFIGS / ("matched_iterative%s.json" % model_tag(path))
            with self.subTest(config=path.name):
                a, b = load_config(path), load_config(iterative)
                self.assertEqual((a["model"], a["revision"]), (b["model"], b["revision"]))
                self.assertEqual({k: a["training"][k] for k in lora}, {k: b["training"][k] for k in lora})
                self.assertEqual(a["matching"], b["matching"])

    def test_every_model_has_its_three_configs(self):
        tags = {model_tag(p) for p in matched_configs()}
        for tag in tags:
            for kind in KINDS:
                with self.subTest(tag=tag, kind=kind):
                    self.assertTrue((CONFIGS / ("%s%s.json" % (kind, tag))).exists())

    def test_the_readme_values_tables_are_the_configs(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_table", REPO / "scripts" / "multiround" / "config_table.py")
        config_table = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_table)
        self.assertEqual(config_table.readme_section(), config_table.section(),
                         "configs/README.md is stale: run scripts/multiround/config_table.py --write")


if __name__ == "__main__":
    unittest.main()
