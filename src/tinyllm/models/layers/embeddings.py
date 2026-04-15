"""
Embedding layers for transformer models.
"""

import torch
import torch.nn as nn
import math


class LearnablePositionalEmbeddings(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, max_seq_len: int, emb_dim: int):
        """
        Initialize Learnable Positional Embeddings.

        Args:
            max_seq_len: Maximum sequence length.
            emb_dim: Embedding dimension.
        """
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, emb_dim) * 0.02)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input.

        Args:
            x: Input tensor of shape [batch, seq_len, emb_dim].

        Returns:
            Tensor with positional embeddings added.
        """
        seq_len = x.shape[1]
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        return x + self.pos_emb[:, :seq_len, :]


class SinusoidalPositionalEmbeddings(nn.Module):
    """Sinusoidal positional embeddings (fixed, not learned)."""

    def __init__(self, max_seq_len: int, emb_dim: int):
        """
        Initialize Sinusoidal Positional Embeddings.

        Args:
            max_seq_len: Maximum sequence length.
            emb_dim: Embedding dimension.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len

        # Create sinusoidal embeddings
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim)
        )

        pe = torch.zeros(1, max_seq_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional embeddings to input.

        Args:
            x: Input tensor of shape [batch, seq_len, emb_dim].

        Returns:
            Tensor with positional embeddings added.
        """
        seq_len = x.shape[1]
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        return x + self.pe[:, :seq_len, :]
