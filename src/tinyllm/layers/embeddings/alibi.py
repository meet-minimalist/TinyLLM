"""
ALiBi (Attention with Linear Biases) positional encoding.

Instead of adding position embeddings to the input, ALiBi adds a
position-dependent bias to the attention scores. This allows the model
to extrapolate to longer sequences at inference time.

Reference:
  Train Short, Test Long: Attention with Linear Biases (ALiBi)
  https://arxiv.org/abs/2108.12409
"""

import torch
import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("alibi")
class AlibiPositionalEmbedding(nn.Module):
    """
    ALiBi position bias generator.

    This module does NOT modify the input tensor. Instead, it provides
    a `get_bias(seq_len)` method that returns a causal bias matrix with
    ALiBi slopes for each head. The attention module must call this
    and add the bias to attention scores.

    Usage in config:
        embedding: "alibi"
    """

    def __init__(self, max_seq_len: int, emb_dim: int, num_heads: int = 8, **kwargs):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.num_heads = kwargs.get("num_heads", num_heads)

        # ALiBi slopes: geometric sequence for each head
        # m_0 = 2^(-8/num_heads), m_h = 1 / 2^(h * 8/num_heads)
        def _get_slopes(n: int) -> torch.Tensor:
            def _get_slopes_power_of_2(n: int) -> list:
                start = 2 ** (-(2 ** -(n.bit_length() - 1)))
                ratio = start
                return [start * (ratio**i) for i in range(n)]

            if n & (n - 1) == 0:  # power of 2
                return torch.tensor(_get_slopes_power_of_2(n))
            else:
                # Closest power of 2
                n_log2 = 2 ** (n.bit_length() - 1)
                slopes_power_of_2 = _get_slopes_power_of_2(n_log2)
                slopes_extra = _get_slopes_power_of_2(2 * n_log2)
                return torch.tensor(
                    slopes_power_of_2 + slopes_extra[0::2][: n - n_log2]
                )

        slopes = _get_slopes(self.num_heads)
        self.register_buffer("_slopes", slopes, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ALiBi is a no-op on the hidden states — bias is added in attention.
        """
        return x

    def get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Return the ALiBi bias matrix.

        Returns:
            Tensor [1, num_heads, seq_len, seq_len] to be added to attention scores.
        """
        # Create position differences: j - i (causal: attends only to past)
        positions = torch.arange(seq_len, device=device).unsqueeze(0) - \
                    torch.arange(seq_len, device=device).unsqueeze(1)
        # Upper triangular (causal): positive for future tokens
        alibi_bias = self._slopes.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
                     positions.unsqueeze(0).float()
        # Causal: mask out future (positions > 0 becomes very negative)
        causal_mask = positions > 0
        alibi_bias = alibi_bias.masked_fill(
            causal_mask.unsqueeze(0), float("-inf")
        )
        return alibi_bias  # [1, num_heads, seq_len, seq_len]
