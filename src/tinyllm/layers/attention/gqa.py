from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.models.base import make_block_causal_mask
from src.tinyllm.models.layers.rope import apply_rotary_pos_emb
from src.tinyllm.models.layers.normalization import RMSNorm

# Lazy flag — set once at first forward to avoid repeated failed imports
_HAS_FLASH_ATTN_PKG: Optional[bool] = None


def _check_flash_attn_pkg() -> bool:
    global _HAS_FLASH_ATTN_PKG
    if _HAS_FLASH_ATTN_PKG is None:
        try:
            import flash_attn  # noqa: F401

            _HAS_FLASH_ATTN_PKG = True
        except ImportError:
            print(
                "flash_attn package not found. Flash attention will not be used."
            )
            _HAS_FLASH_ATTN_PKG = False
    return _HAS_FLASH_ATTN_PKG


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

        if use_qk_norm:
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

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.q_norm is not None:
            q_norm_out = self.q_norm(q)
            k_norm_out = self.k_norm(k)
            q, k = q_norm_out, k_norm_out

        # Expand kv heads for equivalent full-head computation
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        B, H, S, D = q.shape

        if self.flash and cu_seqlens is not None and _check_flash_attn_pkg():
            # ---- flash_attn varlen API (requires flash-attn package) ----
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
                softmax_scale=self.head_dim**-0.5,
                causal=True,
            )
            attn_out = attn_out.reshape(B, S, H, D).transpose(1, 2)
            attn_weights = None
        else:
            # ---- F.scaled_dot_product_attention with optional block mask ----
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
