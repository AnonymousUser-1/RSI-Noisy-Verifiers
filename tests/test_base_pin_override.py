"""RSI_BASE_PIN: running the matched study on another base model.

The study pins one base (configs/pins/base_pin.json, Qwen3-1.7B).  A
teammate running the same protocol on another model (e.g. Qwen3-4B) names that
model's pin file with RSI_BASE_PIN instead of editing the frozen one.  The pin
file and its sha256 are already in every binding check_base_pin returns, so a
run records which pin it was checked against.
"""
import os
import unittest
from pathlib import Path
from unittest import mock

from rsi.base_pin import BASE_PIN, check_base_pin, read_base_pin
from rsi.common import file_hash

REPO = Path(__file__).resolve().parents[1]
QWEN3_4B = REPO / "configs" / "pins" / "base_pin_qwen3-4b.json"


class BasePinOverrideTests(unittest.TestCase):
    def test_without_the_variable_the_frozen_pin_applies(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RSI_BASE_PIN", None)
            self.assertEqual(read_base_pin(), read_base_pin(BASE_PIN))
            self.assertEqual(read_base_pin()["model"], "Qwen/Qwen3-1.7B")

    def test_the_variable_names_another_pin_and_the_binding_records_it(self):
        pin = read_base_pin(QWEN3_4B)
        self.assertEqual(pin["model"], "Qwen/Qwen3-4B")
        config = {"backend": "hf", "model": pin["model"], "revision": pin["revision"]}
        with mock.patch.dict(os.environ, {"RSI_BASE_PIN": str(QWEN3_4B)}):
            binding = check_base_pin(config)
            with self.assertRaises(ValueError):
                check_base_pin({"backend": "hf", "model": "Qwen/Qwen3-1.7B",
                                "revision": read_base_pin(BASE_PIN)["revision"]})
        self.assertEqual(binding["pin_path"], str(QWEN3_4B))
        self.assertEqual(binding["pin_sha256"], file_hash(QWEN3_4B))

    def test_a_named_pin_that_does_not_exist_is_an_error(self):
        with mock.patch.dict(os.environ, {"RSI_BASE_PIN": str(REPO / "no-such-pin.json")}):
            with self.assertRaises(ValueError):
                read_base_pin()


if __name__ == "__main__":
    unittest.main()
