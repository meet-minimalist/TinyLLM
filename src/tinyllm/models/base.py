import torch
import torch.nn as nn


class BaseLLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return torch.where(
            mask.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min, 0.0
        )
