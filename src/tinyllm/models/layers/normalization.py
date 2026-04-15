"""
Normalization layers for transformer models.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x / norm)
