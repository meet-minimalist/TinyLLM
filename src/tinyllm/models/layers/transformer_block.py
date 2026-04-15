"""
Transformer decoder block — single unified implementation.

Supports both GPT-style (learned PE, standard MHA, standard FFN) and
modern LLM-style (RoPE, GQA/MQA, gated FFN, RMSNorm) through config.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from src.tinyllm.models.layers.generate_qkv import QKVGen
from src.tinyllm.models.layers.normalization import RMSNorm


def _build_norm(normalization: str, dim: int) -> nn.Module:
    """Build a normalization layer."""
    if normalization == "rms":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


def _build_attention(
    attention_type: str,
    emb_dim: int,
    num_heads: int,
    drop_prob: float,
    config: dict,
) -> nn.Module:
    """Build an attention layer."""
    if attention_type == "mha":
        return _MHA(emb_dim, num_heads, drop_prob)
    if attention_type == "gqa":
        return _GQA(emb_dim, num_heads, config, drop_prob)
    raise ValueError(f"Unknown attention type: {attention_type}")


def _build_ffn(
    ffn_type: str,
    emb_dim: int,
    ff_multiplier: int,
    drop_rate: float,
    config: dict,
) -> nn.Module:
    """Build an FFN layer."""
    if ffn_type == "standard":
        return _StandardFFN(
            emb_dim, ff_multiplier, drop_rate, config.get("act_fn", "gelu")
        )
    if ffn_type == "gated":
        return _GatedFFN(
            emb_dim, ff_multiplier, drop_rate, config.get("ffn_act", "swish")
        )
    raise ValueError(f"Unknown FFN type: {ffn_type}")


# ── Internal building-block classes ──────────────────────────────────────────


class _MHA(nn.Module):
    """Causal multi-head attention with optional RoPE."""

    def __init__(self, emb_dim: int, num_heads: int, drop_prob: float):
        super().__init__()
        assert (
            emb_dim % num_heads == 0
        ), "emb_dim must be divisible by num_heads"
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_gen = QKVGen(emb_dim, num_heads)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from src.tinyllm.models.layers.rope import apply_rotary_pos_emb

        q, k, v = self.qkv_gen(x)

        # Apply RoPE if provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        seq_len = x.shape[1]
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        if mask is not None:
            attn = attn + mask

        weights = torch.softmax(attn, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], seq_len, -1)
        out = self.out_proj(out)
        return self.layer_norm(x + self.dropout(out))


class _GQA(nn.Module):
    """Grouped query attention with optional RoPE."""

    def __init__(
        self, emb_dim: int, num_heads: int, config: dict, drop_prob: float
    ):
        super().__init__()
        num_kv_heads = config.get("num_kv_heads", num_heads)
        self.head_dim = emb_dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(
            emb_dim,
            num_heads * self.head_dim,
            bias=config.get("qkv_bias", False),
        )
        self.k_proj = nn.Linear(
            emb_dim,
            num_kv_heads * self.head_dim,
            bias=config.get("qkv_bias", False),
        )
        self.v_proj = nn.Linear(
            emb_dim,
            num_kv_heads * self.head_dim,
            bias=config.get("qkv_bias", False),
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            emb_dim,
            bias=config.get("o_proj_bias", False),
        )
        self.dropout = nn.Dropout(drop_prob)

        if config.get("use_qk_norm", False):
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from src.tinyllm.models.layers.rope import apply_rotary_pos_emb

        b, s, _ = x.shape
        q = (
            self.q_proj(x)
            .view(b, s, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(b, s, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(b, s, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = (
            (self.dropout(weights) @ v)
            .transpose(1, 2)
            .contiguous()
            .view(b, s, -1)
        )
        return self.o_proj(out)


class _StandardFFN(nn.Module):
    """Standard FFN with residual and layer norm."""

    def __init__(
        self, emb_dim: int, ff_multiplier: int, drop_rate: float, act_fn: str
    ):
        super().__init__()
        act_dict = {"gelu": nn.GELU(), "relu": nn.ReLU(), "swish": nn.SiLU()}
        self.ff1 = nn.Linear(emb_dim, emb_dim * ff_multiplier)
        self.ff2 = nn.Linear(emb_dim * ff_multiplier, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.act = act_dict.get(act_fn, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.ff1(x)))
        return self.layer_norm(x + self.dropout(self.ff2(h)))


class _GatedFFN(nn.Module):
    """Gated FFN (SwiGLU/GeGLU) with residual and RMS norm."""

    def __init__(
        self, emb_dim: int, ff_multiplier: int, drop_rate: float, act_fn: str
    ):
        super().__init__()
        act_dict = {"swish": nn.SiLU(), "gelu": nn.GELU()}
        hidden = emb_dim * ff_multiplier
        self.gate = nn.Linear(emb_dim, hidden, bias=False)
        self.up = nn.Linear(emb_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, emb_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.norm = RMSNorm(emb_dim)
        self.act = act_dict.get(act_fn, nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gate(x)) * self.up(x)
        return self.norm(x + self.dropout(self.down(h)))


# ── Public unified block ─────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Unified transformer decoder block.

    Configures attention type, FFN type, normalization, and residual
    placement from a config dict.

    Typical presets:
        GPT:     attention_type="mha", ffn_type="standard", normalization="layer_norm"
        Qwen/LLaMA: attention_type="gqa", ffn_type="gated", normalization="rms"
    """

    def __init__(self, config: dict):
        """
        Initialize Transformer Block.

        Args:
            config: Model configuration dictionary. Expected keys:
                - d_model: Model dimension.
                - num_heads: Number of attention heads.
                - attention_type: "mha" or "gqa".
                - ffn_type: "standard" or "gated".
                - ff_multiplier: FFN hidden dimension multiplier.
                - drop_rate / drop_prob: Dropout probability.
                - normalization: "layer_norm" or "rms".
                Additional keys passed to sub-modules as needed
                (num_kv_heads, qkv_bias, use_qk_norm, act_fn, ffn_act, …).
        """
        super().__init__()

        emb_dim = config["d_model"]
        drop_rate = config.get("drop_rate", config.get("drop_prob", 0.0))
        normalization = config.get("normalization", "layer_norm")
        attention_type = config.get("attention_type", "mha")
        ffn_type = config.get("ffn_type", "standard")

        self.attn = _build_attention(
            attention_type, emb_dim, config["num_heads"], drop_rate, config
        )
        self.ffn = _build_ffn(
            ffn_type, emb_dim, config.get("ff_multiplier", 4), drop_rate, config
        )

        self.attn_norm = _build_norm(normalization, emb_dim)
        self.ffn_norm = _build_norm(normalization, emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            mask: Optional attention bias (additive or boolean).
            cos: Cosine tensor for RoPE (used by GQA).
            sin: Sine tensor for RoPE (used by GQA).

        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        x = x + self.attn(self.attn_norm(x), mask=mask, cos=cos, sin=sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
