import torch
import torch.nn as nn


def make_block_causal_mask(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Build a block-diagonal causal mask from cumulative sequence lengths.

    Each position can only attend to earlier positions *within the same document*.
    Returns float mask of shape (S, S), where -inf = masked out.
    """
    S = cu_seqlens[-1].item()
    device = cu_seqlens.device
    dtype = torch.float32

    # full causal: lower-triangle (j <= i) is allowed
    causal = ~torch.triu(
        torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
    )

    # document IDs for each position
    doc_ids = torch.zeros(S, dtype=torch.long, device=device)
    for i in range(len(cu_seqlens) - 1):
        doc_ids[cu_seqlens[i] : cu_seqlens[i + 1]] = i

    same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
    allowed = causal & same_doc
    return torch.where(allowed, 0.0, torch.tensor(float("-inf"), dtype=dtype))


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
