import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


_ACT_DICT = {"swish": nn.SiLU(), "gelu": nn.GELU()}


@LAYER_REGISTRY.register("gated")
class GatedFFN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ff_multiplier: int = 4,
        drop_rate: float = 0.0,
        act_fn: str = "swish",
        **kwargs
    ):
        super().__init__()
        hidden = emb_dim * ff_multiplier
        self.gate = nn.Linear(emb_dim, hidden, bias=False)
        self.up = nn.Linear(emb_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, emb_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.act = _ACT_DICT.get(act_fn, nn.SiLU())

    def forward(self, x):
        return self.dropout(self.down(self.act(self.gate(x)) * self.up(x)))
