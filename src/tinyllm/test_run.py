"""
Quick test: trains the model on randomly generated data.

No data download needed. Use this to verify the full pipeline
(model build → forward → backward → analysis → benchmark) works.

Usage:
    python -m src.tinyllm.test_run
"""

import sys, os, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tinyllm.utils.misc import Config, get_tokenizer
from src.tinyllm.factory.factory import model_factory
from src.tinyllm.logger.logger_utils import configure_logging, logger

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Build a tiny model
model_cfg = Config.parse(
    os.path.join(os.path.dirname(__file__), "configs", "models", "gpt.yaml")
)
model_cfg.d_model = 64
model_cfg.num_heads = 2
model_cfg.num_blocks = 2
model_cfg.vocab_size = 2048
model_cfg.max_seq_len = 128

model = model_factory(model_cfg)
model.to(device)
model.train()

tokenizer = get_tokenizer(model_cfg.tokenizer_name)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")
print(f"Device: {device}")

# Generate random training data
batch_size = 2
seq_len = 64
num_batches = 10

for step in range(num_batches):
    x = torch.randint(
        0, model_cfg.vocab_size, (batch_size, seq_len), device=device
    )
    y = torch.randint(
        0, model_cfg.vocab_size, (batch_size, seq_len), device=device
    )

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1)
    )
    loss.backward()

    # Track gradient norms
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm**0.5

    print(
        f"Step {step+1:2d}/{num_batches}  loss={loss.item():.4f}  grad_norm={total_norm:.4f}"
    )

    # Check for explosion
    if total_norm > 100:
        logger.warning(f"Gradient explosion detected! norm={total_norm:.2f}")

    # Manual optimizer step (simple SGD for test)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
        model.zero_grad()

# Run weight stats
from src.tinyllm.analysis.weight_stats import compute_weight_stats

wstats = compute_weight_stats(model)
print(f"Weight stats computed ({len(wstats)} entries)")

# Run spectral on first weight
from src.tinyllm.analysis.spectral import (
    compute_svd_and_variance,
    compute_weightwatcher_alpha,
)

for name, p in model.named_parameters():
    if p.ndim >= 2:
        r = compute_svd_and_variance(p)
        alpha = compute_weightwatcher_alpha(p)
        print(
            f"Spectral {name}: n_for_95={r['n_for_95pct']}/{min(p.shape)}, alpha={alpha:.2f}"
        )
        break

print(f"\n{'='*40}")
print(f"Test complete — {num_batches} steps on random data.")
print(f"Run real training with:")
print(
    f"  python -m src.tinyllm.train -c configs/training/train_config.yaml -m configs/models/gpt.yaml"
)
