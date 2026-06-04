import torch
import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.layers.transformer_block import TransformerBlock
from src.tinyllm.models.base import BaseLLM
from src.tinyllm.factory.registry import MODEL_REGISTRY


def _first_key(cfg, *keys, default=None):
    for k in keys:
        if isinstance(cfg, dict):
            v = cfg.get(k)
        else:
            v = getattr(cfg, k, None)
        if v is not None:
            return v
    return default


@MODEL_REGISTRY.register("dynamic")
class DynamicModel(BaseLLM):
    def __init__(self, config):
        super().__init__(config)

        cfg = config if isinstance(config, dict) else config.to_dict()
        d_model = cfg["d_model"]
        vocab_size = cfg["vocab_size"]
        max_seq_len = cfg["max_seq_len"]

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        embed_type = cfg.get("embedding", "learned_pe")
        embed_cls = LAYER_REGISTRY.get(embed_type)
        self.position_embedding = embed_cls(
            max_seq_len=max_seq_len, emb_dim=d_model
        )

        blocks_cfg = _first_key(cfg, "blocks", default={})
        ff_mult = _first_key(
            blocks_cfg, "ff_multiplier", "ffn_multiplier", default=4
        )
        block_config = {
            "d_model": d_model,
            "num_heads": _first_key(blocks_cfg, "num_heads", default=4),
            "ff_multiplier": ff_mult,
            "drop_prob": _first_key(blocks_cfg, "drop_prob", default=0.0),
            "attention": _first_key(blocks_cfg, "attention", default="mha"),
            "ffn": _first_key(blocks_cfg, "ffn", default="standard"),
            "norm": _first_key(blocks_cfg, "norm", default="layer_norm"),
            "act_fn": _first_key(blocks_cfg, "act_fn", default="gelu"),
        }
        for k, v in blocks_cfg.items():
            if k not in (
                "count",
                "attention",
                "ffn",
                "norm",
                "num_heads",
                "ff_multiplier",
                "ffn_multiplier",
                "drop_prob",
                "drop_rate",
                "act_fn",
            ):
                block_config[k] = v

        num_blocks = _first_key(blocks_cfg, "count", default=4)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(block_config) for _ in range(num_blocks)]
        )

        head_cfg = _first_key(cfg, "head", default={})
        norm_type = _first_key(head_cfg, "norm", default="layer_norm")
        norm_cls = LAYER_REGISTRY.get(norm_type)
        self.final_norm = norm_cls(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        tie = _first_key(head_cfg, "tie_weights", default=False)
        if tie:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(self, input_ids, mask=None):
        x = self.token_embedding(input_ids)
        x = self.position_embedding(x)

        seq_len = input_ids.shape[1]
        causal = self._causal_mask(seq_len, x.device)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).to(torch.float32)
            mask = (1.0 - mask) * torch.finfo(torch.float32).min
            causal = causal + mask

        for block in self.transformer_blocks:
            x = block(x, causal)

        return self.lm_head(self.final_norm(x))
