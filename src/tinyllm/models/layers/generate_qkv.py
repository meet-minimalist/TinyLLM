"""
Query, Key, Value generation utilities for attention mechanisms.
"""

from typing import Tuple

import torch
import torch.nn as nn


class QKVGen(nn.Module):
    """Generate Q, K, V tensors from input for multi-head attention."""

    def __init__(self, emb_dim: int, num_heads: int):
        """
        Initialize QKV Generator.

        Args:
            emb_dim: Embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        assert (
            emb_dim % num_heads == 0
        ), "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Single linear layer for Q, K, V (more efficient than 3 separate layers)
        self.linear = nn.Linear(emb_dim, emb_dim * 3)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Q, K, V tensors from input.

        Args:
            x: Input tensor of shape [batch, seq_len, emb_dim].

        Returns:
            Tuple of (Q, K, V) tensors, each of shape [batch, num_heads, seq_len, head_dim].
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.linear(x)  # [batch, seq_len, emb_dim * 3]
        q, k, v = qkv.chunk(3, dim=-1)  # Split into 3 equal parts

        # Reshape for multi-head attention
        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        return q, k, v
