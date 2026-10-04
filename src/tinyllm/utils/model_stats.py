"""
Parameter and FLOP accounting.

For MoE the numbers that matter are different from a dense model: *total*
params set memory, *active* params set compute. FLOPs per token use the active
non-embedding count (Kaplan 6N rule, arXiv:2001.08361) plus the attention
score/value term, which grows with context length.

A module can declare parameters that are not used for every token (e.g. MoE
experts beyond top-k) by defining ``inactive_param_count() -> int``. Dense
models need nothing.
"""

import torch.nn as nn

# Dense peak TFLOP/s per GPU (tensor cores, no sparsity) for the training
# dtype. bf16 where the GPU has it, fp16 otherwise (T4, V100, P100).
_PEAK_TFLOPS = {
    "H100": 989.0,
    "H200": 989.0,
    "A100": 312.0,
    "A10G": 125.0,
    "A10": 125.0,
    "L4": 121.0,
    "L40S": 362.0,
    "RTX 4090": 165.0,
    "RTX 3090": 71.0,
    "RTX 3050": 18.0,
    "V100": 125.0,
    "T4": 65.0,
    "P100": 18.7,
}


def count_params(model: nn.Module) -> dict:
    """Return total / embedding / non-embedding / active parameter counts.

    Tied weights are counted once. The embedding count includes an untied
    ``lm_head`` (it is a vocab-sized matrix, same as the input embedding).
    """
    seen = set()
    total = 0
    embedding = 0
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if "embedding" in name or "lm_head" in name:
            embedding += p.numel()

    inactive = 0
    for module in model.modules():
        fn = getattr(module, "inactive_param_count", None)
        if callable(fn):
            inactive += int(fn())

    non_embedding = total - embedding
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": non_embedding,
        "active": total - inactive,
        "active_non_embedding": non_embedding - inactive,
    }


def flops_per_token(
    model_config, active_non_embedding: int, seq_len: int
) -> float:
    """Training FLOPs per token: 6N + output layer + causal attention.

    - 6N over active non-embedding params (Kaplan).
    - The output projection (lm_head, d_model x vocab) is a real matmul even
      when tied to the input embedding, which is only a lookup: 6 * d * V.
      In small models it is a large share — about half the FLOPs at
      d=512 / V=50K, and ~1/3 at d=1024 / V=151K.
    - Full attention costs 12 * L * d_attn * T per token for fwd+bwd; causal
      masking halves it. ``seq_len`` is the average document length the model
      attends over (per-document varlen packing never attends across documents).
    """
    blocks = model_config.get("blocks", {}) or {}
    n_layers = int(blocks.get("count", 0))
    d_model = int(model_config.get("d_model", 0))
    vocab = int(model_config.get("vocab_size", 0))
    return (
        6.0 * active_non_embedding
        + 6.0 * d_model * vocab
        + 6.0 * n_layers * d_model * seq_len
    )


def peak_flops(device_name: str) -> float | None:
    """Dense peak FLOP/s for a CUDA device name, or None if unknown."""
    for key, tflops in _PEAK_TFLOPS.items():
        if key in device_name:
            return tflops * 1e12
    return None


def fmt_count(n: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(int(n))
