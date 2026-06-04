import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.layers.transformer_block import TransformerBlock
from src.tinyllm.models.base import BaseLLM
from src.tinyllm.factory.registry import MODEL_REGISTRY


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

        block_cfg = {
            k: v
            for k, v in cfg.items()
            if k
            not in (
                "model_type",
                "tokenizer_name",
                "vocab_size",
                "max_seq_len",
                "embedding",
                "head",
            )
        }
        block_cfg["d_model"] = d_model
        block_cfg.setdefault("attention", "mha")
        block_cfg.setdefault("ffn", "standard")
        block_cfg.setdefault("norm", "layer_norm")
        block_cfg.setdefault("drop_prob", 0.0)
        block_cfg.setdefault("ff_multiplier", 4)

        num_blocks = cfg.get(
            "num_blocks",
            cfg.get("num_layers", cfg.get("blocks", {}).get("count", 4)),
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(block_cfg) for _ in range(num_blocks)]
        )

        head_cfg = cfg.get("head", {})
        norm_type = head_cfg.get("norm", cfg.get("norm", "layer_norm"))
        norm_cls = LAYER_REGISTRY.get(norm_type)
        self.final_norm = norm_cls(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        tie = cfg.get("tie_weights", head_cfg.get("tie_weights", False))
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
