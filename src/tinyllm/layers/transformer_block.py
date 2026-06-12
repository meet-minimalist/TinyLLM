from typing import Optional

import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        emb_dim = config["d_model"]
        drop_rate = config.get("drop_rate", config.get("drop_prob", 0.0))
        norm_type = config.get(
            "norm", config.get("normalization", "layer_norm")
        )
        attn_type = config.get("attention", config.get("attention_type", "mha"))
        ffn_type = config.get("ffn", config.get("ffn_type", "standard"))

        norm_cls = LAYER_REGISTRY.get(norm_type)
        attn_cls = LAYER_REGISTRY.get(attn_type)
        ffn_cls = LAYER_REGISTRY.get(ffn_type)

        self.attn_norm = norm_cls(emb_dim)
        self.ffn_norm = norm_cls(emb_dim)

        self.attn = attn_cls(
            emb_dim=emb_dim,
            num_heads=config["num_heads"],
            drop_prob=drop_rate,
            **{
                k: v
                for k, v in config.items()
                if k not in ("d_model", "num_heads", "drop_rate", "drop_prob")
            },
        )
        self.ffn = ffn_cls(
            emb_dim=emb_dim,
            ff_multiplier=config.get(
                "ff_multiplier", config.get("ffn_multiplier", 4)
            ),
            drop_rate=drop_rate,
            **{
                k: v
                for k, v in config.items()
                if k
                not in (
                    "emb_dim",
                    "ff_multiplier",
                    "ffn_multiplier",
                    "drop_rate",
                    "drop_prob",
                )
            },
        )

    def forward(
        self,
        x,
        mask: Optional = None,
        cos: Optional = None,
        sin: Optional = None,
        cu_seqlens: Optional = None,
    ):
        attn_out, _ = self.attn(
            self.attn_norm(x),
            mask=mask,
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x
