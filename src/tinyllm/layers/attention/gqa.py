import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.models.base import make_block_causal_mask
from src.tinyllm.models.layers.rope import apply_rotary_pos_emb
from src.tinyllm.models.layers.normalization import RMSNorm
from src.tinyllm.layers.ngpt import l2norm

# Lazy flags — checked once at first forward to avoid repeated failed imports
_HAS_FLASH_ATTN: Optional[bool] = None
_HAS_FLEX_ATTN: Optional[bool] = None


def _check_flash_attn() -> bool:
    global _HAS_FLASH_ATTN
    if _HAS_FLASH_ATTN is None:
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401

            _HAS_FLASH_ATTN = True
        except (ImportError, AttributeError):
            # AttributeError catches packages like FA4 that install as flash_attn
            # but don't expose flash_attn_varlen_func at the top level
            _HAS_FLASH_ATTN = False
    return _HAS_FLASH_ATTN


def _check_flex_attn() -> bool:
    global _HAS_FLEX_ATTN
    if _HAS_FLEX_ATTN is None:
        try:
            from torch.nn.attention.flex_attention import (
                flex_attention,
                create_block_mask,
            )  # noqa: F401

            _HAS_FLEX_ATTN = True
        except ImportError:
            _HAS_FLEX_ATTN = False
    return _HAS_FLEX_ATTN


def _make_doc_block_mask(cu_seqlens: torch.Tensor, seq_len: int):
    from torch.nn.attention.flex_attention import create_block_mask

    doc_ids = torch.zeros(seq_len, dtype=torch.long, device=cu_seqlens.device)
    for i in range(len(cu_seqlens) - 1):
        doc_ids[cu_seqlens[i] : cu_seqlens[i + 1]] = i

    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])

    return create_block_mask(
        mask_fn,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=cu_seqlens.device,
    )


@LAYER_REGISTRY.register("gqa")
class GQA(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        drop_prob: float = 0.0,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        o_proj_bias: bool = False,
        use_qk_norm: bool = False,
        flash: bool = False,
        ngpt: bool = False,
        **kwargs,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.flash = flash
        self.ngpt = ngpt
        self.emb_dim = emb_dim

        self.q_proj = nn.Linear(
            emb_dim, num_heads * self.head_dim, bias=qkv_bias
        )
        self.k_proj = nn.Linear(
            emb_dim, num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            emb_dim, num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim, emb_dim, bias=o_proj_bias
        )
        self.dropout = nn.Dropout(drop_prob)

        if use_qk_norm and not ngpt:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if ngpt:
            from src.tinyllm.layers.ngpt import NGPTScale

            # nGPT replaces RMSNorm on q/k with L2 normalization per head plus
            # a learned per-channel scale. q and k become unit vectors, so
            # q.k lands in [-1, 1] and the usual 1/sqrt(d_k) would shrink the
            # logits further; the paper multiplies by sqrt(d_k) instead.
            scale = 1.0 / math.sqrt(emb_dim)
            self.q_scale = NGPTScale(num_heads * self.head_dim, 1.0, scale)
            self.k_scale = NGPTScale(num_kv_heads * self.head_dim, 1.0, scale)
            self.softmax_scale = math.sqrt(self.head_dim)
        else:
            self.q_scale = None
            self.k_scale = None
            self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def normalize_weights(self):
        from src.tinyllm.layers.ngpt import normalize_linear_

        # q/k/v read from the residual stream (embedding axis = in_features),
        # o_proj writes back into it (embedding axis = out_features).
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            normalize_linear_(proj, dim=1)
        normalize_linear_(self.o_proj, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
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

        # QK-norm must run BEFORE RoPE, as in Qwen3. The two orders are not
        # equivalent: RoPE preserves each head vector's norm, but the norm's
        # learnable per-channel weight multiplies the channels, and RoPE has
        # already mixed them by then.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.q_scale is not None:
            # nGPT: unit-normalize every head vector, then apply the learned
            # per-channel scale. This runs AFTER RoPE, matching the reference
            # implementation — RoPE is a rotation so it cannot change the norm,
            # but the scale has to act on the rotated channels.
            q = l2norm(q) * self.q_scale().view(
                1, self.num_heads, 1, self.head_dim
            )
            k = l2norm(k) * self.k_scale().view(
                1, self.num_kv_heads, 1, self.head_dim
            )

        B, H, S, D = q.shape

        if self.flash and cu_seqlens is not None and _check_flash_attn():
            # ---- Tier 1: flash_attn varlen API — natively supports GQA ----
            # Best performance; requires flash-attn package (fa2/fa3).
            from flash_attn import flash_attn_varlen_func

            q_flat = q.transpose(1, 2).reshape(-1, H, D)
            k_flat = k.transpose(1, 2).reshape(-1, self.num_kv_heads, D)
            v_flat = v.transpose(1, 2).reshape(-1, self.num_kv_heads, D)

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

            attn_out = flash_attn_varlen_func(
                q_flat,
                k_flat,
                v_flat,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=True,
            )
            attn_out = attn_out.reshape(B, S, H, D).transpose(1, 2)
            attn_weights = None

        elif self.flash and cu_seqlens is not None and _check_flex_attn():
            # ---- Tier 2: flex_attention with document block mask ----
            # PyTorch-native Triton kernels; works on all CUDA arches (Ampere,
            # Hopper, Blackwell) without an external package.
            from torch.nn.attention.flex_attention import flex_attention

            if self.num_groups > 1:
                k = k.repeat_interleave(self.num_groups, dim=1)
                v = v.repeat_interleave(self.num_groups, dim=1)
            block_mask = _make_doc_block_mask(cu_seqlens, S)
            attn_out = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=self.softmax_scale,
            )
            attn_weights = None

        else:
            # ---- Tier 3: F.scaled_dot_product_attention fallback ----
            # Always works; cuDNN uses flash attention internally on CUDA.
            # SDPA doesn't natively handle GQA grouping, so expand K/V here.
            if self.num_groups > 1:
                k = k.repeat_interleave(self.num_groups, dim=1)
                v = v.repeat_interleave(self.num_groups, dim=1)

            if cu_seqlens is not None:
                attn_mask = make_block_causal_mask(cu_seqlens)
            else:
                attn_mask = None

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=attn_mask is None,
                scale=self.softmax_scale,
            )
            attn_weights = None

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
        o_proj_output = self.o_proj(attn_out)

        metadata = {
            "q": q,
            "k": k,
            "v": v,
            "attn_logits": None,
            "attn_weights": attn_weights,
            "qkv_out": attn_out,
            "o_proj_out": o_proj_output,
        }
        return o_proj_output, metadata
