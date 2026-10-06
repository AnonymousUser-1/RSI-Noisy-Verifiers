from __future__ import annotations

"""Reference gradient computation for matched-subset experiments.

Computes h = grad R_reference on an independent held-out dataset, separate from:
- training data (train_*)
- development data (dev)
- calibration data (calibration)
- evaluation data (eval_id, eval_ood)

The reference gradient h is used to compute the main diagnostic h^T Δθ.
"""

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ReferenceGradientComputer:
    """Compute reference gradient h = grad R_reference.

    The reference gradient is computed once per seed on a held-out dataset and
    saved to disk. It is then loaded by GradientRecorder for projection computation.

    Usage:
        computer = ReferenceGradientComputer(model, reference_dataloader)
        h = computer.compute()
        computer.save('runs/seed0/reference_gradient.pt')

        # Later, in GradientRecorder
        recorder = GradientRecorder(model, 'runs/seed0/reference_gradient.pt')
    """

    def __init__(self, model: nn.Module, reference_dataloader: DataLoader,
                 device: str = 'cuda', normalize: bool = True):
        """Initialize reference gradient computer.

        Args:
            model: The model to compute gradients on (must have LoRA parameters)
            reference_dataloader: DataLoader for the gradient_reference split
            device: Device to run computation on ('cuda' or 'cpu')
            normalize: Whether to normalize the reference gradient to unit norm
        """
        self.model = model
        self.reference_dataloader = reference_dataloader
        self.device = device
        self.normalize = normalize
        self.h: Optional[Dict[str, torch.Tensor]] = None

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute reference gradient on the reference dataset.

        Returns:
            h: Dict mapping parameter names to gradient tensors

        The gradient is computed by:
        1. Accumulating loss over the entire reference dataset
        2. Computing backward() once on the total loss
        3. Extracting gradients from LoRA parameters
        4. Optionally normalizing to unit L2 norm
        """
        self.model.eval()
        self.model.to(self.device)

        # Zero out any existing gradients
        self.model.zero_grad()

        total_loss = 0.0
        num_samples = 0

        # Accumulate loss over reference set
        with torch.enable_grad():
            for batch in self.reference_dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                # Accumulate (don't backward yet - we want one gradient for the entire set)
                total_loss += loss * input_ids.size(0)  # Un-average the loss
                num_samples += input_ids.size(0)

            # Average loss over all samples
            avg_loss = total_loss / num_samples

            # Single backward pass for entire reference set
            avg_loss.backward()

        # Extract gradient from LoRA parameters
        h = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                if param.grad is None:
                    raise ValueError(f"Parameter {name} has no gradient")
                h[name] = param.grad.clone().detach().cpu()

        if not h:
            raise ValueError("No LoRA parameters found in model")

        # Normalize (optional)
        if self.normalize:
            h_norm = math.sqrt(sum((g ** 2).sum().item() for g in h.values()))
            if h_norm == 0:
                raise ValueError("Reference gradient has zero norm")
            h = {k: v / h_norm for k, v in h.items()}

        self.h = h
        return h

    def save(self, path: str):
        """Save reference gradient to disk.

        Args:
            path: Output path (will be created with parent directories)
        """
        if self.h is None:
            raise RuntimeError("Must call compute() before save()")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.h, path)

    @staticmethod
    def load(path: str) -> Dict[str, torch.Tensor]:
        """Load reference gradient from disk.

        Args:
            path: Path to saved reference gradient

        Returns:
            h: Dict mapping parameter names to gradient tensors
        """
        return torch.load(path, weights_only=True)

    def compute_and_save(self, path: str) -> Dict[str, torch.Tensor]:
        """Convenience method to compute and save in one call.

        Args:
            path: Output path

        Returns:
            h: The computed reference gradient
        """
        h = self.compute()
        self.save(path)
        return h


def create_reference_gradient(model: nn.Module, reference_dataset,
                             batch_size: int = 32,
                             output_path: str = 'reference_gradient.pt',
                             device: str = 'cuda',
                             normalize: bool = True) -> Dict[str, torch.Tensor]:
    """Convenience function to create reference gradient.

    Args:
        model: Model to compute gradients on
        reference_dataset: Dataset for gradient_reference split
        batch_size: Batch size for DataLoader
        output_path: Where to save the reference gradient
        device: Device to run on
        normalize: Whether to normalize to unit norm

    Returns:
        h: The computed reference gradient
    """
    dataloader = DataLoader(
        reference_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order
        num_workers=0   # Single-threaded for reproducibility
    )

    computer = ReferenceGradientComputer(
        model=model,
        reference_dataloader=dataloader,
        device=device,
        normalize=normalize
    )

    return computer.compute_and_save(output_path)
