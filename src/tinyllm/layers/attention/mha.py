from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.models.base import make_block_causal_mask
from src.tinyllm.models.layers.generate_qkv import QKVGen
from src.tinyllm.models.layers.rope import apply_rotary_pos_emb

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


@LAYER_REGISTRY.register("mha")
class MHA(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        drop_prob: float = 0.0,
        flash: bool = False,
        window_size: int = 0,
        **kwargs
    ):
        super().__init__()
        assert (
            emb_dim % num_heads == 0
        ), "emb_dim must be divisible by num_heads"
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.flash = flash
        self.window_size = window_size  # 0 = no sliding window

        self.qkv_gen = QKVGen(emb_dim, num_heads)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        q, k, v = self.qkv_gen(x)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        B, H, S, D = q.shape

        if self.flash and cu_seqlens is not None and _check_flash_attn():
            # ---- Tier 1: flash_attn varlen API ----
            from flash_attn import flash_attn_varlen_func

            q_flat = q.transpose(1, 2).reshape(-1, H, D)
            k_flat = k.transpose(1, 2).reshape(-1, H, D)
            v_flat = v.transpose(1, 2).reshape(-1, H, D)

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
                softmax_scale=self.scale,
                causal=True,
            )
            attn_out = attn_out.reshape(B, S, H, D).transpose(1, 2)
            attn_weights = None

        elif self.flash and cu_seqlens is not None and _check_flex_attn():
            # ---- Tier 2: flex_attention with document block mask ----
            # PyTorch-native Triton kernels; no external package needed.
            from torch.nn.attention.flex_attention import flex_attention

            block_mask = _make_doc_block_mask(cu_seqlens, S)
            attn_out = flex_attention(
                q, k, v, block_mask=block_mask, scale=self.scale
            )
            attn_weights = None

        else:
            # ---- Tier 3: F.scaled_dot_product_attention fallback ----
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
            )
            attn_weights = None

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        o_proj_output = self.out_proj(attn_out)

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
