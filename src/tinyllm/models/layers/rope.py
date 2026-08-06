"""
Rotary Positional Embedding (RoPE) implementation.
"""

import math

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_type: str = None,
        scaling_factor: float = 1.0,
        original_max_seq_len: int = None,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        """
        Initialize RoPE, optionally with YaRN/LongRoPE scaling.

        Args:
            head_dim: Dimension of each attention head.
            max_seq_len: Maximum sequence length.
            base: Base frequency for RoPE.
            scaling_type: None (no scaling), "yarn" (YaRN), or "longrope".
            scaling_factor: Extension factor. >1 extends context length.
            original_max_seq_len: Original max_seq_len before scaling
                (used for the YaRN ramp bounds). Defaults to
                max_seq_len / scaling_factor.
            beta_fast: YaRN high-frequency boundary (dims with more than this
                many rotations over the original context are left unscaled).
            beta_slow: YaRN low-frequency boundary (dims with fewer than this
                many rotations are fully interpolated).
        """
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Attention temperature scaling (YaRN). 1.0 == no effect, so the
        # standard and LongRoPE paths leave cos/sin untouched.
        self.mscale = 1.0

        # Original max length before scaling
        self.original_max_seq_len = original_max_seq_len or int(
            max_seq_len / max(scaling_factor, 1.0)
        )

        if scaling_type == "yarn":
            # YaRN: interpolate frequencies with a ramp
            self._init_yarn(base)
        elif scaling_type == "longrope":
            # LongRoPE: same interpolation, different beta scheduling
            self._init_longrope(base)
        else:
            # Standard RoPE
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _init_yarn(self, base: int):
        """
        Initialize YaRN-scaled frequencies (Peng et al., 2023, Section 3.2).

        Uses wavelength-based frequency-band bounds derived from beta_fast /
        beta_slow: high-frequency dims (many rotations over the original
        context) are extrapolated unchanged, low-frequency dims are fully
        interpolated by 1/scaling_factor, and a linear ramp blends the band
        in between. Also computes the attention temperature ``mscale``.
        """
        dim = self.head_dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )

        # Wavelength-based index bounds. A dim's rotation count over the
        # original context is orig_len * inv_freq / (2*pi); solving for the
        # index where that equals beta gives these closed-form bounds.
        def _bound(num_rotations: float) -> float:
            return (
                self.head_dim
                * math.log(
                    self.original_max_seq_len / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        low_idx = max(0, math.floor(_bound(self.beta_fast)))
        high_idx = min(dim - 1, math.ceil(_bound(self.beta_slow)))

        # Non-uniform ramp: 0 below the band (extrapolate), linear across the
        # band, 1 above it (interpolate).
        ramp = torch.zeros(dim)
        if high_idx > low_idx:
            ramp[low_idx : high_idx + 1] = torch.linspace(
                0.0, 1.0, steps=(high_idx - low_idx + 1)
            )
        ramp[high_idx + 1 :] = 1.0

        # ramp=0 -> keep original inv_freq; ramp=1 -> interpolated (/factor).
        inv_freq_interpolated = inv_freq / self.scaling_factor
        inv_freq = (1.0 - ramp) * inv_freq + ramp * inv_freq_interpolated

        # Temperature scaling: t = sqrt(1 + 0.1*ln(s)); applied as a magnitude
        # scale on cos/sin. mscale = 0.1*ln(s) + 1 matches the paper's
        # sqrt(1/t) folded into the attention logits.
        if self.scaling_factor > 1.0:
            self.mscale = 0.1 * math.log(self.scaling_factor) + 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_yarn_ramp", ramp, persistent=False)

    def _init_longrope(self, base: int):
        """Initialize LongRoPE-scaled frequencies."""
        # LongRoPE uses a similar approach to YaRN with different beta scheduling
        dim = self.head_dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )

        # Beta schedule for LongRoPE
        ratio = self.original_max_seq_len / self.max_seq_len
        t = torch.arange(dim, dtype=torch.float) / max(1, dim - 1)
        beta = 1.0 - (1.0 - ratio) * (1.0 - t**2)

        inv_freq = inv_freq / beta.clamp(min=0.1)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int = None,
        position_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine and sine tensors for RoPE.

        Args:
            x: Input tensor of shape [batch, seq_len, num_heads, head_dim].
            seq_len: Sequence length to compute embeddings for. If None, uses x.shape[1].
            position_ids: Optional [seq_len] tensor of per-token position
                indices. When None, positions are ``0..seq_len-1`` (contiguous).
                In varlen packing this is supplied so positions **reset to 0 at
                each document boundary**, keeping every document's RoPE indices
                within ``[0, max_seq_len)`` regardless of how many documents are
                packed into one row.

        Returns:
            Tuple of (cos, sin) tensors for applying RoPE.
        """
        if seq_len is None:
            seq_len = x.shape[1]

        if position_ids is None:
            t = torch.arange(
                seq_len, device=x.device, dtype=self.inv_freq.dtype
            )
        else:
            t = position_ids.to(device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim/2]

        # Compute cos and sin, scaled by the YaRN attention temperature
        # (mscale == 1.0 for standard / LongRoPE, so those are unaffected).
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
        cos = emb.cos() * self.mscale  # [seq_len, head_dim]
        sin = emb.sin() * self.mscale  # [seq_len, head_dim]

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
