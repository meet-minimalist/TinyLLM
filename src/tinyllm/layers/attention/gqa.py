from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.models.layers.rope import apply_rotary_pos_emb
from src.tinyllm.models.layers.normalization import RMSNorm


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

        if self.flash:
            if self.num_groups > 1:
                k = k.repeat_interleave(self.num_groups, dim=1)
                v = v.repeat_interleave(self.num_groups, dim=1)
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
            attn_weights = None
        else:
            if self.num_groups > 1:
                k = k.repeat_interleave(self.num_groups, dim=1)
                v = v.repeat_interleave(self.num_groups, dim=1)
            scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            if mask is not None:
                scores = scores + mask
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = self.dropout(attn_weights) @ v

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
