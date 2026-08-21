"""
Multi-Head Latent Attention (MLA).

Introduced in DeepSeek-V2, MLA uses low-rank joint compression for keys
and values to dramatically reduce KV cache size while maintaining full
multi-head expressiveness.

Reference:
    https://arxiv.org/abs/2405.04434 (DeepSeek-V2)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.models.layers.rope import apply_rotary_pos_emb


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
            from torch.nn.attention.flex_attention import (  # noqa: F401
                flex_attention,
                create_block_mask,
            )

            _HAS_FLEX_ATTN = True
        except ImportError:
            _HAS_FLEX_ATTN = False
    return _HAS_FLEX_ATTN


def make_block_causal_mask(cu_seqlens: torch.Tensor) -> torch.Tensor:
    S = cu_seqlens[-1].item()
    device = cu_seqlens.device
    causal = ~torch.triu(
        torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
    )
    doc_ids = torch.zeros(S, dtype=torch.long, device=device)
    for i in range(len(cu_seqlens) - 1):
        doc_ids[cu_seqlens[i] : cu_seqlens[i + 1]] = i
    same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
    allowed = causal & same_doc
    return torch.where(
        allowed, 0.0, torch.tensor(float("-inf"), dtype=torch.float32)
    )


@LAYER_REGISTRY.register("mla")
class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention with low-rank KV compression.

    Config:
        emb_dim: Model dimension.
        num_heads: Number of attention heads.
        q_lora_rank: Low-rank dim for Q projection (default: emb_dim//2).
        kv_lora_rank: Low-rank dim for KV latent (default: emb_dim//4).
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        drop_prob: float = 0.0,
        q_lora_rank: int = None,
        kv_lora_rank: int = None,
        flash: bool = False,
        **kwargs,
    ):
        super().__init__()
        head_dim = emb_dim // num_heads
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.flash = flash

        self.q_lora_rank = q_lora_rank or max(64, emb_dim // 2)
        self.kv_lora_rank = kv_lora_rank or max(32, emb_dim // 4)

        # ---- Query projection (low-rank) ----
        self.q_a_proj = nn.Linear(emb_dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, num_heads * head_dim, bias=False
        )

        # ---- KV latent projection (shared low-rank) ----
        self.kv_a_proj = nn.Linear(emb_dim, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, num_heads * head_dim * 2, bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
        cos=None,
        sin=None,
        cu_seqlens=None,
    ) -> Tuple[torch.Tensor, dict]:
        batch, seq_len, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # ---- Q projection (low-rank) ----
        q_latent = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_latent).view(batch, seq_len, H, D).transpose(1, 2)

        # ---- KV latent (shared low-rank) ----
        kv_latent = self.kv_a_layernorm(self.kv_a_proj(x))
        kv_b = (
            self.kv_b_proj(kv_latent)
            .view(batch, seq_len, H, 2 * D)
            .transpose(1, 2)
        )
        k, v = kv_b.chunk(2, dim=-1)

        # Apply RoPE if provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- Attention ----
        if self.flash and cu_seqlens is not None and _check_flash_attn():
            from flash_attn import flash_attn_varlen_func

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            q_flat = q.transpose(1, 2).reshape(-1, H, D)
            k_flat = k.transpose(1, 2).reshape(-1, H, D)
            v_flat = v.transpose(1, 2).reshape(-1, H, D)
            attn_out = flash_attn_varlen_func(
                q_flat,
                k_flat,
                v_flat,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=D**-0.5,
                causal=True,
            )
            attn_out = attn_out.reshape(batch, seq_len, H, D).transpose(1, 2)
            attn_weights = None
        elif self.flash and cu_seqlens is not None and _check_flex_attn():
            from torch.nn.attention.flex_attention import (
                flex_attention,
                create_block_mask,
            )

            doc_ids = torch.zeros(seq_len, dtype=torch.long, device=x.device)
            for i in range(len(cu_seqlens) - 1):
                doc_ids[cu_seqlens[i] : cu_seqlens[i + 1]] = i

            def _mask_fn(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])

            block_mask = create_block_mask(
                _mask_fn,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=x.device,
            )
            attn_out = flex_attention(
                q, k, v, block_mask=block_mask, scale=D**-0.5
            )
            attn_weights = None
        else:
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

        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        )
        o_proj_output = self.out_proj(attn_out)

        metadata = {
            "q": q,
            "k": k,
            "v": v,
            "attn_logits": None,
            "attn_weights": attn_weights,
            "qkv_out": attn_out,
            "o_proj_out": o_proj_output,
            "kv_latent": kv_latent,
        }
        return o_proj_output, metadata
