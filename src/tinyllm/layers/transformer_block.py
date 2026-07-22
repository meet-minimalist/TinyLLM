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
        self.block_pattern = config.get(
            "block_pattern", "sequential"
        )  # sequential | parallel
        self.depth_upscale = config.get(
            "depth_upscale", 1
        )  # >1 = depth-upscaled

        norm_cls = LAYER_REGISTRY.get(norm_type)
        attn_cls = LAYER_REGISTRY.get(attn_type)
        ffn_cls = LAYER_REGISTRY.get(ffn_type)

        shared_kwargs = {
            k: v
            for k, v in config.items()
            if k
            not in (
                "d_model",
                "num_heads",
                "drop_rate",
                "drop_prob",
                "emb_dim",
                "ff_multiplier",
                "ffn_multiplier",
            )
        }

        # When block_pattern is "parallel" and depth_upscale > 1, we create
        # multiple parallel FFN branches (depth-upscaled) that run in parallel
        # with the attention.
        num_ffn_branches = 1
        if self.depth_upscale > 1:
            num_ffn_branches = self.depth_upscale
            self.block_pattern = "parallel"

        if self.block_pattern == "sequential":
            # === Standard pre-norm: attn → residual → ffn → residual ===
            self.attn_norm = norm_cls(emb_dim)
            self.ffn_norm = norm_cls(emb_dim)

            self.attn = attn_cls(
                emb_dim=emb_dim,
                num_heads=config["num_heads"],
                drop_prob=drop_rate,
                **shared_kwargs,
            )
            self.ffn = ffn_cls(
                emb_dim=emb_dim,
                ff_multiplier=config.get(
                    "ff_multiplier", config.get("ffn_multiplier", 4)
                ),
                drop_rate=drop_rate,
                **shared_kwargs,
            )
        elif self.block_pattern == "parallel":
            # === Parallel: norm → [attn, ffn_1..ffn_N] → sum → residual ===
            self.prenorm = norm_cls(emb_dim)
            self.attn = attn_cls(
                emb_dim=emb_dim,
                num_heads=config["num_heads"],
                drop_prob=drop_rate,
                **shared_kwargs,
            )
            self.ffns = nn.ModuleList(
                [
                    ffn_cls(
                        emb_dim=emb_dim,
                        ff_multiplier=config.get(
                            "ff_multiplier", config.get("ffn_multiplier", 4)
                        ),
                        drop_rate=drop_rate,
                        **shared_kwargs,
                    )
                    for _ in range(num_ffn_branches)
                ]
            )
        else:
            raise ValueError(
                f"Unknown block_pattern: {self.block_pattern}. "
                f"Supported: sequential, parallel"
            )

    def forward(
        self,
        x,
        mask: Optional = None,
        cos: Optional = None,
        sin: Optional = None,
        cu_seqlens: Optional = None,
    ):
        if self.block_pattern == "sequential":
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

        elif self.block_pattern == "parallel":
            # Apply shared prenorm, then run attention + all FFN branches in parallel
            h = self.prenorm(x)
            attn_out, _ = self.attn(
                h,
                mask=mask,
                cos=cos,
                sin=sin,
                cu_seqlens=cu_seqlens,
            )
            ffn_out = sum(ffn(h) for ffn in self.ffns)
            return x + attn_out + ffn_out
