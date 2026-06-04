import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


_ACT_DICT = {"gelu": nn.GELU(), "relu": nn.ReLU(), "swish": nn.SiLU()}


@LAYER_REGISTRY.register("standard")
class StandardFFN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ff_multiplier: int = 4,
        drop_rate: float = 0.0,
        act_fn: str = "gelu",
        **kwargs
    ):
        super().__init__()
        self.ff1 = nn.Linear(emb_dim, emb_dim * ff_multiplier)
        self.ff2 = nn.Linear(emb_dim * ff_multiplier, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.act = _ACT_DICT.get(act_fn, nn.GELU())

    def forward(self, x):
        return self.dropout(self.ff2(self.dropout(self.act(self.ff1(x)))))
