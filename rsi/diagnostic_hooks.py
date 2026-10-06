from __future__ import annotations

"""PyTorch backward hooks for per-sample gradient capture.

Provides utilities to capture per-sample gradients during backward passes,
which is needed for the gradient recorder to compute G_C, G_E statistics.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class DiagnosticHooks:
    """PyTorch backward hooks for per-sample gradient capture.

    Usage:
        hooks = DiagnosticHooks(model)
        hooks.register()

        # Training loop
        for sample in batch:
            model.zero_grad()
            loss = model(sample['input_ids'], labels=sample['labels']).loss
            loss.backward()

            # Get per-sample gradient
            gradient = hooks.get_current_gradient()
            recorder.record_per_sample_gradient(...)

        hooks.remove()
    """

    def __init__(self, model: nn.Module):
        """Initialize diagnostic hooks.

        Args:
            model: The model to attach hooks to (must have LoRA parameters)
        """
        self.model = model
        self.hooks: List = []
        self.current_gradients: Dict[str, torch.Tensor] = {}

    def _make_hook(self, param_name: str):
        """Create a hook function for a specific parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Hook function that captures gradient
        """
        def hook(grad: torch.Tensor):
            # Clone and detach the gradient
            self.current_gradients[param_name] = grad.clone().detach()
        return hook

    def register(self):
        """Attach hooks to all LoRA parameters.

        Call this once before training begins.
        """
        if self.hooks:
            raise RuntimeError("Hooks already registered. Call remove() first.")

        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                hook = param.register_hook(self._make_hook(name))
                self.hooks.append((name, hook))

        if not self.hooks:
            raise ValueError("No LoRA parameters found in model")

    def remove(self):
        """Remove all hooks.

        Call this after training to clean up and prevent memory leaks.
        """
        for _, hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.current_gradients = {}

    def get_current_gradient(self) -> Dict[str, torch.Tensor]:
        """Get the gradient captured from the last backward() call.

        Returns:
            Dict mapping parameter names to gradient tensors (on CPU)

        The gradients are cloned and detached, so they won't interfere with
        subsequent backward passes.
        """
        if not self.current_gradients:
            raise RuntimeError("No gradients captured. Did you call backward()?")

        # Return a copy to avoid mutation
        return {k: v.clone().cpu() for k, v in self.current_gradients.items()}

    def clear_current_gradient(self):
        """Clear the current gradient buffer.

        Call this between samples if you want to verify that each backward()
        is actually updating the gradients.
        """
        self.current_gradients = {}

    def __enter__(self):
        """Context manager entry: register hooks."""
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: remove hooks."""
        self.remove()


def get_full_gradient(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get the current gradient from all LoRA parameters.

    This is a utility function for cases where you don't need per-sample
    gradients, just the accumulated batch gradient.

    Args:
        model: Model with LoRA parameters

    Returns:
        Dict mapping parameter names to gradient tensors (on CPU)
    """
    gradients = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            if param.grad is None:
                raise ValueError(f"Parameter {name} has no gradient")
            gradients[name] = param.grad.clone().detach().cpu()

    if not gradients:
        raise ValueError("No LoRA parameters found in model")

    return gradients


def check_gradient_clipping(grad_before: Dict[str, torch.Tensor],
                           grad_after: Dict[str, torch.Tensor],
                           tolerance: float = 1e-6) -> bool:
    """Check if gradient clipping was triggered.

    Args:
        grad_before: Gradient dict before clipping
        grad_after: Gradient dict after clipping
        tolerance: Tolerance for considering gradients equal

    Returns:
        True if clipping was triggered (gradients differ)
    """
    if set(grad_before.keys()) != set(grad_after.keys()):
        raise ValueError("Gradient dicts have different keys")

    for key in grad_before:
        diff = (grad_before[key] - grad_after[key]).abs().max().item()
        if diff > tolerance:
            return True

    return False
