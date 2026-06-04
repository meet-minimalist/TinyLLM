from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.models.layers.generate_qkv import QKVGen
from src.tinyllm.models.layers.rope import apply_rotary_pos_emb


@LAYER_REGISTRY.register("mha")
class MHA(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        drop_prob: float = 0.0,
        flash: bool = False,
        **kwargs
    ):
        super().__init__()
        assert (
            emb_dim % num_heads == 0
        ), "emb_dim must be divisible by num_heads"
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.flash = flash

        self.qkv_gen = QKVGen(emb_dim, num_heads)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        q, k, v = self.qkv_gen(x)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.flash:
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
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            seq_len = x.shape[1]
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(
                causal.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            if mask is not None:
                attn = attn + mask
            attn_weights = torch.softmax(attn, dim=-1)
            attn_out = torch.matmul(self.dropout(attn_weights), v)

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], x.shape[1], -1)
        )
        o_proj_output = self.out_proj(attn_out)

        metadata = {
            "q": q,
            "k": k,
            "v": v,
            "attn_logits": None if self.flash else attn,
            "attn_weights": attn_weights,
            "qkv_out": attn_out,
            "o_proj_out": o_proj_output,
        }
        return o_proj_output, metadata
