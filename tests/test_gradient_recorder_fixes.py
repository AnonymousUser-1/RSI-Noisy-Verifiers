"""Regression tests for gradient_recorder.py fixes (Issue B defect 3).

Tests the vector composition validation from review task T003:
D. Exact cancellation: g_C=(1,0), g_E=(-1,0), batch=(0,0)
E. Orthogonal: g_C=(1,0), g_E=(0,1), batch=(1,1)
F. Same norm wrong direction: g1=g2=(1,0), fake_batch=(0,2)
G. Scaled contributions: mean-scaled (0.5,0)+(0,0.5), batch=(0.5,0.5)
Plus: near-zero boundary case for relative error reporting
"""
import pytest
from pathlib import Path
import tempfile
import json

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class MockLoRAModel(torch.nn.Module):
    """Minimal model with LoRA-named parameters for testing."""
    def __init__(self, param_shape=(2,)):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(param_shape))
        self.lora_B = torch.nn.Parameter(torch.zeros(param_shape))


@pytest.fixture
def temp_reference_gradient():
    """Create a temporary reference gradient file."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        # Simple reference gradient: h = {"lora_A": [1.0, 0.0], "lora_B": [0.0, 1.0]}
        h = {
            "lora_A": torch.tensor([1.0, 0.0]),
            "lora_B": torch.tensor([0.0, 1.0])
        }
        torch.save(h, f.name)
        yield f.name
    Path(f.name).unlink()


def test_fix_D_exact_cancellation(temp_reference_gradient):
    """Fix D: g_C=(1,0), g_E=(-1,0), G_preclip=(0,0) should pass.

    Exact cancellation: residual=0, preclip_l2=0 -> relative_error=0.0
    """
    from rsi.gradient_recorder import GradientRecorder

    model = MockLoRAModel()
    recorder = GradientRecorder(model, temp_reference_gradient)

    recorder.before_update()

    # Record g_C = (1,0) scaled contribution
    g_C = {
        "lora_A": torch.tensor([1.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_C", is_correct=True, gradient=g_C, loss=0.5)

    # Record g_E = (-1,0) scaled contribution
    g_E = {
        "lora_A": torch.tensor([-1.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_E", is_correct=False, gradient=g_E, loss=0.5)

    # G_preclip = (0,0) after cancellation
    G_preclip = {
        "lora_A": torch.tensor([0.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    G_after = G_preclip  # No clipping
    recorder.record_batch_gradient(G_preclip, G_after, clipping_fired=False)

    # Simulate optimizer step (no actual update)
    recorder.after_update()

    # Should pass validation
    diagnostics = recorder.compute_diagnostics()

    assert diagnostics["validation"]["composition_valid"], "Exact cancellation should pass"
    assert diagnostics["validation"]["residual_l2"] < 1e-6, "Residual should be ~0"
    assert diagnostics["validation"]["relative_error"] == 0.0, "Relative error should be 0.0 for exact cancellation"
    assert not diagnostics["validation"]["undefined_zero_reference"], "Should not be undefined"


def test_fix_E_orthogonal_vectors(temp_reference_gradient):
    """Fix E: g_C=(1,0), g_E=(0,1), G_preclip=(1,1) should pass.

    Old check: sum(||g_i||) = 2 vs ||G|| = sqrt(2) -> FAIL
    New check: vector equality -> PASS
    """
    from rsi.gradient_recorder import GradientRecorder

    model = MockLoRAModel()
    recorder = GradientRecorder(model, temp_reference_gradient)

    recorder.before_update()

    g_C = {
        "lora_A": torch.tensor([1.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_C", is_correct=True, gradient=g_C, loss=0.5)

    g_E = {
        "lora_A": torch.tensor([0.0, 1.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_E", is_correct=False, gradient=g_E, loss=0.5)

    G_preclip = {
        "lora_A": torch.tensor([1.0, 1.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    G_after = G_preclip
    recorder.record_batch_gradient(G_preclip, G_after, clipping_fired=False)

    recorder.after_update()

    diagnostics = recorder.compute_diagnostics()

    assert diagnostics["validation"]["composition_valid"], "Orthogonal vectors should pass"
    assert diagnostics["validation"]["residual_l2"] < 1e-6
    assert [(r["id"], r["correct"], r["loss"]) for r in diagnostics["per_sample"]] == [
        ("sample_C", True, 0.5), ("sample_E", False, 0.5)]
    assert [round(r["norm"], 6) for r in diagnostics["per_sample"]] == [1.0, 1.0]
    assert diagnostics["gradient_norms"]["sum_correct_sample_norms"] == diagnostics["per_sample"][0]["norm"]
    assert diagnostics["validation"]["relative_error"] < 1e-6


def test_fix_F_same_norm_wrong_direction(temp_reference_gradient):
    """Fix F: g1=g2=(1,0), fake_batch=(0,2) should FAIL.

    Old check: sum(||g_i||) = 2 vs ||fake|| = 2 -> PASS (wrong!)
    New check: ||(1,0)+(1,0) - (0,2)|| = sqrt(8) -> FAIL (correct)
    """
    from rsi.gradient_recorder import GradientRecorder

    model = MockLoRAModel()
    recorder = GradientRecorder(model, temp_reference_gradient)

    recorder.before_update()

    g1 = {
        "lora_A": torch.tensor([1.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_1", is_correct=True, gradient=g1, loss=0.5)

    g2 = {
        "lora_A": torch.tensor([1.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_2", is_correct=True, gradient=g2, loss=0.5)

    # Fake batch gradient with same norm but wrong direction
    G_fake = {
        "lora_A": torch.tensor([0.0, 2.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    G_after = G_fake
    recorder.record_batch_gradient(G_fake, G_after, clipping_fired=False)

    recorder.after_update()

    # Should FAIL validation
    with pytest.raises(ValueError, match="Gradient composition validation failed"):
        recorder.compute_diagnostics()


def test_fix_G_mean_scaled_contributions(temp_reference_gradient):
    """Fix G: mean-scaled contributions should pass.

    g1_scaled = (0.5, 0), g2_scaled = (0, 0.5), batch = (0.5, 0.5)
    """
    from rsi.gradient_recorder import GradientRecorder

    model = MockLoRAModel()
    recorder = GradientRecorder(model, temp_reference_gradient)

    recorder.before_update()

    # Already scaled by (w_i / denominator)
    g1_scaled = {
        "lora_A": torch.tensor([0.5, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_1", is_correct=True, gradient=g1_scaled, loss=0.5)

    g2_scaled = {
        "lora_A": torch.tensor([0.0, 0.5]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_2", is_correct=False, gradient=g2_scaled, loss=0.5)

    G_batch = {
        "lora_A": torch.tensor([0.5, 0.5]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    G_after = G_batch
    recorder.record_batch_gradient(G_batch, G_after, clipping_fired=False)

    recorder.after_update()

    diagnostics = recorder.compute_diagnostics()

    assert diagnostics["validation"]["composition_valid"], "Mean-scaled contributions should pass"
    assert diagnostics["validation"]["residual_l2"] < 1e-6


def test_near_zero_relative_error_reporting(temp_reference_gradient):
    """Boundary case: preclip_l2=5e-10, residual_l2=2.5e-10.

    Expected:
    - relative_error = 0.5 (not undefined)
    - undefined_zero_reference = False
    - CPU may pass due to atol, but relative_error is still reported as 0.5
    """
    from rsi.gradient_recorder import GradientRecorder

    model = MockLoRAModel()
    recorder = GradientRecorder(model, temp_reference_gradient)

    recorder.before_update()

    # g_C and g_E chosen so their sum has very small norm
    # Let's use g_C = (5e-10, 0), g_E = (0, 0), so G_C + G_E = (5e-10, 0)
    # And set G_preclip = (2.5e-10, 0) to create a small residual
    g_C = {
        "lora_A": torch.tensor([5e-10, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_C", is_correct=True, gradient=g_C, loss=0.5)

    g_E = {
        "lora_A": torch.tensor([0.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_E", is_correct=False, gradient=g_E, loss=0.5)

    # Set G_preclip = (2.5e-10, 0)
    # residual = (5e-10, 0) - (2.5e-10, 0) = (2.5e-10, 0)
    # residual_l2 = 2.5e-10, preclip_l2 = 2.5e-10
    # relative_error = 1.0 (not 0.5 as intended, let me recalculate)

    # Let me fix: preclip = (5e-10, 0), residual should be (2.5e-10, 0)
    # So G_C + G_E = (7.5e-10, 0), G_preclip = (5e-10, 0)
    g_C = {
        "lora_A": torch.tensor([7.5e-10, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.per_sample_records[-1]['norm'] = 7.5e-10  # Update after changing g_C

    # Actually, let me construct this more carefully
    # Want: preclip_l2 = 5e-10, residual_l2 = 2.5e-10
    # So: ||G_preclip|| = 5e-10, ||residual|| = 2.5e-10
    # residual = G_C + G_E - G_preclip
    # If G_C = (7.5e-10, 0), G_E = (0, 0), G_preclip = (5e-10, 0)
    # residual = (2.5e-10, 0), residual_l2 = 2.5e-10 ✓
    # relative_error = 2.5e-10 / 5e-10 = 0.5 ✓

    # Re-record with correct values
    recorder.per_sample_records.clear()
    recorder.G_C_accumulated = {
        "lora_A": torch.tensor([7.5e-10, 0.0], dtype=torch.float32),
        "lora_B": torch.tensor([0.0, 0.0], dtype=torch.float32)
    }
    recorder.G_E_accumulated = {
        "lora_A": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "lora_B": torch.tensor([0.0, 0.0], dtype=torch.float32)
    }
    recorder.per_sample_records.append({'sample_id': 'C', 'is_correct': True, 'norm': 7.5e-10, 'loss': 0.5})
    recorder.per_sample_records.append({'sample_id': 'E', 'is_correct': False, 'norm': 0.0, 'loss': 0.5})

    G_preclip = {
        "lora_A": torch.tensor([5e-10, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    G_after = G_preclip
    recorder.record_batch_gradient(G_preclip, G_after, clipping_fired=False)

    recorder.after_update()

    diagnostics = recorder.compute_diagnostics()

    # May pass CPU validation due to atol=1e-6
    # But relative_error should still be reported accurately
    val = diagnostics["validation"]
    assert abs(val["relative_error"] - 0.5) < 0.1, f"Expected relative_error ~0.5, got {val['relative_error']}"
    assert not val["undefined_zero_reference"], "Should not be undefined for positive preclip_l2"


def test_zero_residual_positive_preclip(temp_reference_gradient):
    """Boundary case: residual=0, preclip_l2 > 0 -> relative_error=0 (not undefined)."""
    from rsi.gradient_recorder import GradientRecorder

    model = MockLoRAModel()
    recorder = GradientRecorder(model, temp_reference_gradient)

    recorder.before_update()

    # g_C = (1,0), g_E = (0,0), G_preclip = (1,0) -> residual = 0
    g_C = {
        "lora_A": torch.tensor([1.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_C", is_correct=True, gradient=g_C, loss=0.5)

    g_E = {
        "lora_A": torch.tensor([0.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_per_sample_gradient("sample_E", is_correct=False, gradient=g_E, loss=0.5)

    G_preclip = {
        "lora_A": torch.tensor([1.0, 0.0]),
        "lora_B": torch.tensor([0.0, 0.0])
    }
    recorder.record_batch_gradient(G_preclip, G_preclip, clipping_fired=False)

    recorder.after_update()

    diagnostics = recorder.compute_diagnostics()

    assert diagnostics["validation"]["composition_valid"]
    assert diagnostics["validation"]["residual_l2"] < 1e-6
    assert diagnostics["validation"]["relative_error"] == 0.0, "Should be 0, not undefined"
    assert not diagnostics["validation"]["undefined_zero_reference"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_h_on_the_cpu_projects_a_gpu_update(temp_reference_gradient):
    """compute_reference_gradient.py saves h on the CPU; the arms' Delta theta is on the GPU.

    The first real GPU run (2026-10-01) failed here: "Expected all tensors to be
    on the same device, cuda:0 and cpu".  h = ((1, 0), (0, 1)) and the update
    Delta theta = ((2, 3), (5, 7)) give h^T Delta theta = 2 + 7 = 9.
    """
    from rsi.gradient_recorder import GradientRecorder

    model = MockLoRAModel().cuda()
    recorder = GradientRecorder(model, temp_reference_gradient)
    recorder.before_update()
    zero = {"lora_A": torch.zeros(2, device="cuda"), "lora_B": torch.zeros(2, device="cuda")}
    recorder.record_per_sample_gradient("sample_C", is_correct=True, gradient=zero, loss=0.5)
    recorder.record_batch_gradient(zero, zero, clipping_fired=False)
    with torch.no_grad():
        model.lora_A.copy_(torch.tensor([2.0, 3.0]))
        model.lora_B.copy_(torch.tensor([5.0, 7.0]))
    recorder.after_update()

    diagnostics = recorder.compute_diagnostics()

    assert diagnostics["main_diagnostic"]["h_T_delta_theta"] == 9.0
