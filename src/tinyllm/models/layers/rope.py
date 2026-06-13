"""
Rotary Positional Embedding (RoPE) implementation.
"""

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(
        self, head_dim: int, max_seq_len: int = 2048, base: int = 10000
    ):
        """
        Initialize RoPE.

        Args:
            head_dim: Dimension of each attention head.
            max_seq_len: Maximum sequence length.
            base: Base frequency for RoPE.
        """
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Precompute frequency tensor
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine and sine tensors for RoPE.

        Args:
            x: Input tensor of shape [batch, seq_len, num_heads, head_dim].
            seq_len: Sequence length to compute embeddings for. If None, uses x.shape[1].

        Returns:
            Tuple of (cos, sin) tensors for applying RoPE.
        """
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim/2]

        # Compute cos and sin
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
        cos = emb.cos()  # [seq_len, head_dim]
        sin = emb.sin()  # [seq_len, head_dim]

        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim].
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        cos: Cosine tensor from RoPE.
        sin: Sine tensor from RoPE.

    Returns:
        Tuple of (q_rotated, k_rotated).
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    cos = (
        cos.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    )  # [1, 1, seq_len, head_dim]
    sin = (
        sin.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    )  # [1, 1, seq_len, head_dim]

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated
