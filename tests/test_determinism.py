"""Deterministic kernels for the real-model gradient diagnostics.

GPU pilot 2026-10-01 (Qwen3-1.7B + the shared LoRA adapter, RTX 4070 Laptop): the
same 16 examples' summed gradient, computed twice in one process, differed by up
to 2.0e-4 per element without deterministic kernels and was bitwise equal with
them, at about 4% more time.  An R-versus-S contrast in h^T Delta theta should
not carry that run-to-run noise, so every real-model gradient path enables them
before it touches the GPU.
"""
import os
import unittest
from unittest import mock

import torch

import one_step
from rsi.determinism import enable_determinism


class EnableDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.was_enabled = torch.are_deterministic_algorithms_enabled()
        self.env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    def tearDown(self):
        torch.use_deterministic_algorithms(self.was_enabled)
        if self.env is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.env

    def test_it_turns_on_deterministic_kernels_and_records_them(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        torch.use_deterministic_algorithms(False)
        record = enable_determinism()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(record, {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"})

    def test_a_workspace_setting_already_in_place_is_kept(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        self.assertEqual(enable_determinism()["cublas_workspace_config"], ":16:8")

    def test_the_hf_arm_enables_it_before_the_model_is_loaded(self):
        order = []

        class Stop(Exception):
            pass

        def load(*args, **kwargs):
            order.append("backend")
            raise Stop

        with mock.patch("one_step.enable_determinism", side_effect=lambda: order.append("determinism")), \
                mock.patch("rsi.backends.backend", load):
            with self.assertRaises(Stop):
                one_step._run_hf_arm("R", [], {}, None, None, 0, {})
        self.assertEqual(order, ["determinism", "backend"])


if __name__ == "__main__":
    unittest.main()
