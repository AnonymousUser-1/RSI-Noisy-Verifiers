from __future__ import annotations

"""Gradient recording and diagnostic computation for matched-subset experiments.

Records all diagnostics required by matched-dynamics/PLAN.md Section 4:
- per-sample supervised response-token loss
- streaming per-sample gradient vectors (G_C, G_E accumulation)
- gradient before/after clipping
- parameter change delta_theta
- reference gradient projection h^T delta_theta (main diagnostic)

Numerical precision requirements:
- CPU: rtol=1e-4, atol=1e-6 for vector composition identity G_C + G_E = G_preclip
- GPU: relative L2 error < 1% for gradient composition (separate verification gate)

Gradient normalization convention (from backends.py):
  g_i = grad_theta(loss_i)  -- gradient of loss_i w.r.t. parameters
  loss_i is the per-sample response-token-averaged loss (scalar)

  The training backward() is called on:
    batch_loss = sum(w_i * loss_i) / denominator
  where denominator = sum(w_i) over the effective batch.

  Therefore G_preclip = sum((w_i / denominator) * g_i).

  This recorder accumulates the scaled contributions (w_i / denominator) * g_i
  directly. For the main path with all weights = 1.0 and K samples per update,
  this is g_i / K.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class GradientRecorder:
    """Records all diagnostics for one training update.

    Usage:
        recorder = GradientRecorder(model, reference_gradient_path)
        recorder.before_update()

        # Training loop
        for sample in batch:
            loss = compute_loss(sample)  # per-sample loss
            loss.backward()

            # Extract gradient for this sample (before accumulation)
            grad_dict = {name: param.grad.clone()
                        for name, param in model.named_parameters()
                        if 'lora' in name.lower() and param.grad is not None}

            # Record the scaled contribution: (w_i / denominator) * g_i
            recorder.record_per_sample_gradient(
                sample_id=sample['id'],
                is_correct=sample['correct'],
                gradient=grad_dict,  # already scaled by (w_i / denominator)
                loss=loss.item()
            )

        # After all backward() calls, before clipping
        grad_before = recorder._get_current_gradient()

        # Apply clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_after = recorder._get_current_gradient()
        clipping_fired = (norm_before > 1.0)

        recorder.record_batch_gradient(grad_before, grad_after, clipping_fired)

        optimizer.step()
        recorder.after_update()

        diagnostics = recorder.compute_diagnostics()
        recorder.save(output_path, step=0, branch='R', seed=0)
    """

    def __init__(self, model: nn.Module, reference_gradient_path: str,
                 rtol: float = 1e-4, atol: float = 1e-6,
                 max_relative_error: float = 0.01):
        """Initialize gradient recorder.

        Args:
            model: The model being trained (must have LoRA parameters)
            reference_gradient_path: Path to reference gradient h (torch.save format)
            rtol: Relative tolerance for vector composition validation (CPU)
            atol: Absolute tolerance for vector composition validation (CPU)
            max_relative_error: GPU verification threshold (not enforced here)
        """
        self.model = model
        self.rtol = rtol
        self.atol = atol
        self.max_relative_error = max_relative_error

        # Load reference gradient
        self.h = torch.load(reference_gradient_path, weights_only=True)
        if not isinstance(self.h, dict):
            raise ValueError(f"Reference gradient must be a dict, got {type(self.h)}")

        # State storage
        self.theta_before: Optional[Dict[str, torch.Tensor]] = None
        self.theta_after: Optional[Dict[str, torch.Tensor]] = None
        self.per_sample_records: List[Dict] = []

        # Streaming vector accumulators (O(P) space, not O(N*P))
        self.G_C_accumulated: Optional[Dict[str, torch.Tensor]] = None
        self.G_E_accumulated: Optional[Dict[str, torch.Tensor]] = None

        self.batch_gradient_before: Optional[Dict[str, torch.Tensor]] = None
        self.batch_gradient_after: Optional[Dict[str, torch.Tensor]] = None
        self.clipping_fired: bool = False

    def _get_lora_params(self) -> Dict[str, torch.Tensor]:
        """Extract LoRA parameters as a flat dict.

        Returns:
            Dict mapping parameter names to cloned tensors (detached from computation graph)
        """
        params = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                params[name] = param.data.clone().detach()
        if not params:
            raise ValueError("No LoRA parameters found in model")
        return params

    def _get_current_gradient(self) -> Dict[str, torch.Tensor]:
        """Extract current accumulated gradient from model.

        Returns:
            Dict mapping parameter names to gradient tensors
        """
        grads = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.grad is not None:
                grads[name] = param.grad.clone().detach()
        return grads

    def _l2_norm(self, tensor_dict: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of a dict of tensors.

        Args:
            tensor_dict: Dict mapping names to tensors

        Returns:
            L2 norm (scalar)
        """
        return math.sqrt(sum((t.float() ** 2).sum().item() for t in tensor_dict.values()))

    def _dot_product(self, dict_a: Dict[str, torch.Tensor],
                     dict_b: Dict[str, torch.Tensor]) -> float:
        """Compute dot product between two tensor dicts.

        Args:
            dict_a: First dict (e.g., reference gradient h)
            dict_b: Second dict (e.g., parameter change delta_theta)

        Returns:
            Scalar dot product
        """
        if set(dict_a.keys()) != set(dict_b.keys()):
            raise ValueError(f"Key mismatch: {set(dict_a.keys())} vs {set(dict_b.keys())}")

        total = 0.0
        for key in dict_a:
            # Element-wise multiply and sum, use float() for numerical stability.
            # On the CPU: h is saved there (compute_reference_gradient.py) while
            # the arms' Delta theta lives on the GPU.
            total += (dict_a[key].float().cpu() * dict_b[key].float().cpu()).sum().item()
        return total

    def before_update(self):
        """Snapshot parameters and reset accumulators before the update.

        Call this before any backward() calls in the training step.
        """
        self.theta_before = self._get_lora_params()
        self.per_sample_records = []

        # Initialize vector accumulators (O(P) space)
        self.G_C_accumulated = {k: torch.zeros_like(v, dtype=torch.float32)
                                for k, v in self.theta_before.items()}
        self.G_E_accumulated = {k: torch.zeros_like(v, dtype=torch.float32)
                                for k, v in self.theta_before.items()}

        self.batch_gradient_before = None
        self.batch_gradient_after = None
        self.clipping_fired = False

    def record_per_sample_gradient(self, sample_id: str, is_correct: bool,
                                   gradient: Dict[str, torch.Tensor], loss: float):
        """Record gradient and loss for one sample.

        Args:
            sample_id: Unique identifier for this sample
            is_correct: Whether this is a correct (True) or error (False) sample
            gradient: Dict mapping parameter names to gradient tensors.
                     These should be the scaled contributions: (w_i / denominator) * g_i
            loss: Scalar loss value for this sample (loss_i, the response-token average)
        """
        # Verify keys match
        if set(gradient.keys()) != set(self.G_C_accumulated.keys()):
            raise ValueError(f"Gradient keys {set(gradient.keys())} do not match "
                           f"expected keys {set(self.G_C_accumulated.keys())}")

        # Verify shapes match
        for key in gradient:
            if gradient[key].shape != self.G_C_accumulated[key].shape:
                raise ValueError(f"Shape mismatch for {key}: {gradient[key].shape} vs "
                               f"{self.G_C_accumulated[key].shape}")

        # Compute norm for descriptive statistics
        norm = self._l2_norm(gradient)

        # Accumulate into correct or error vector (streaming, O(P) space)
        if is_correct:
            for key in gradient:
                self.G_C_accumulated[key] += gradient[key].float()
        else:
            for key in gradient:
                self.G_E_accumulated[key] += gradient[key].float()

        # Store per-sample descriptive statistics
        self.per_sample_records.append({
            'sample_id': sample_id,
            'is_correct': is_correct,
            'norm': norm,  # ||gradient|| for this sample
            'loss': loss
        })

    def record_batch_gradient(self, gradient_before_clip: Dict[str, torch.Tensor],
                             gradient_after_clip: Dict[str, torch.Tensor],
                             clipping_fired: bool):
        """Record full batch gradient before and after clipping.

        Args:
            gradient_before_clip: Gradient dict before clipping (G_preclip)
            gradient_after_clip: Gradient dict after clipping (may be same as before)
            clipping_fired: Whether gradient clipping was triggered
        """
        self.batch_gradient_before = {k: v.clone().detach().float()
                                     for k, v in gradient_before_clip.items()}
        self.batch_gradient_after = {k: v.clone().detach().float()
                                    for k, v in gradient_after_clip.items()}
        self.clipping_fired = clipping_fired

    def after_update(self):
        """Snapshot parameters after the update and compute delta_theta.

        Call this after optimizer.step().
        """
        if self.theta_before is None:
            raise RuntimeError("Must call before_update() first")

        self.theta_after = self._get_lora_params()

    def compute_diagnostics(self) -> Dict:
        """Compute all diagnostics and validate vector composition.

        Returns:
            Dict containing all diagnostic information

        Raises:
            ValueError: If vector composition validation fails
        """
        if self.theta_before is None or self.theta_after is None:
            raise RuntimeError("Must call before_update() and after_update() first")
        if self.batch_gradient_before is None:
            raise RuntimeError("Must call record_batch_gradient() first")

        # Vector composition validation: G_C + G_E = G_preclip
        # This is the primary criterion for CPU validation
        residual = {k: self.G_C_accumulated[k] + self.G_E_accumulated[k] - self.batch_gradient_before[k]
                   for k in self.batch_gradient_before.keys()}

        # Check vector equality using torch.allclose (main criterion)
        composition_valid = all(
            torch.allclose(
                self.G_C_accumulated[k] + self.G_E_accumulated[k],
                self.batch_gradient_before[k],
                rtol=self.rtol,
                atol=self.atol
            )
            for k in self.batch_gradient_before.keys()
        )

        # Compute L2 norms for reporting (descriptive statistics)
        residual_l2 = self._l2_norm(residual)
        preclip_l2 = self._l2_norm(self.batch_gradient_before)
        G_C_l2 = self._l2_norm(self.G_C_accumulated)
        G_E_l2 = self._l2_norm(self.G_E_accumulated)

        # Relative error (reporting value, not a gate)
        # Handle zero-denominator case carefully
        if preclip_l2 > 0.0:
            relative_error = residual_l2 / preclip_l2
            undefined_zero_reference = False
        elif residual_l2 == 0.0 and preclip_l2 == 0.0:
            # Both zero: exact cancellation or zero gradient
            relative_error = 0.0
            undefined_zero_reference = False
        else:
            # preclip_l2 = 0 but residual_l2 > 0: undefined relative error
            relative_error = None
            undefined_zero_reference = True

        if not composition_valid:
            error_msg = (
                f"Gradient composition validation failed (vector equality check):\n"
                f"  ||G_C||: {G_C_l2:.6e}\n"
                f"  ||G_E||: {G_E_l2:.6e}\n"
                f"  ||G_C + G_E||: {self._l2_norm({k: self.G_C_accumulated[k] + self.G_E_accumulated[k] for k in self.G_C_accumulated}):.6e}\n"
                f"  ||G_preclip||: {preclip_l2:.6e}\n"
                f"  ||residual||: {residual_l2:.6e}\n"
            )
            if relative_error is not None:
                error_msg += f"  Relative error: {relative_error:.6e}\n"
            else:
                error_msg += f"  Relative error: undefined (zero reference)\n"
            error_msg += f"  Tolerance: rtol={self.rtol:.6e}, atol={self.atol:.6e}"
            raise ValueError(error_msg)

        # Compute delta_theta
        delta_theta = {k: self.theta_after[k] - self.theta_before[k]
                      for k in self.theta_before}
        delta_theta_norm = self._l2_norm(delta_theta)

        # Main diagnostic: h^T delta_theta
        h_T_delta_theta = self._dot_product(self.h, delta_theta)

        # Gradient norms before/after clipping
        grad_before_norm = self._l2_norm(self.batch_gradient_before)
        grad_after_norm = self._l2_norm(self.batch_gradient_after)

        # Per-sample norm sums (for backward compatibility / descriptive stats)
        sum_correct_norms = sum(r['norm'] for r in self.per_sample_records if r['is_correct'])
        sum_error_norms = sum(r['norm'] for r in self.per_sample_records if not r['is_correct'])

        validation_result = {
            'composition_valid': composition_valid,
            'residual_l2': residual_l2,
            'relative_error': relative_error,
            'undefined_zero_reference': undefined_zero_reference,
            'rtol': self.rtol,
            'atol': self.atol
        }

        return {
            'gradient_norms': {
                'G_C_vector_norm': G_C_l2,
                'G_E_vector_norm': G_E_l2,
                'G_preclip_norm': preclip_l2,
                'sum_correct_sample_norms': sum_correct_norms,
                'sum_error_sample_norms': sum_error_norms,
            },
            'per_sample_losses': [r['loss'] for r in self.per_sample_records],
            # One entry per example, in training order: its id, label, ||(w_i/sum w) g_i|| and loss_i
            # (the paper's per-response norms; the sums above are these norms added up).
            'per_sample': [{'id': r['sample_id'], 'correct': r['is_correct'], 'norm': r['norm'],
                            'loss': r['loss']} for r in self.per_sample_records],
            'clipping': {
                'fired': self.clipping_fired,
                'grad_before_norm': grad_before_norm,
                'grad_after_norm': grad_after_norm
            },
            'delta_theta': {
                'norm': delta_theta_norm
            },
            'main_diagnostic': {
                'h_T_delta_theta': h_T_delta_theta
            },
            'validation': validation_result
        }

    def save(self, path: str, step: int, branch: str, seed: int):
        """Save diagnostics to JSON file.

        Args:
            path: Output file path
            step: Training step number
            branch: Branch name ('R' or 'S')
            seed: Random seed used
        """
        diagnostics = self.compute_diagnostics()

        output = {
            'step': step,
            'branch': branch,
            'seed': seed,
            'model_params': {
                'num_lora_params': len(self.theta_before),
                'param_names': list(self.theta_before.keys())
            },
            **diagnostics
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
