# TinyLLMs

Lightweight, from-scratch LLM training framework with a **single config-driven model** that supports any decoder-only architecture (GPT, Qwen, LLaMA, etc.) via YAML. Built for the FineWeb10B pretokenized dataset with GPT-2 tokenizer.

## Quick Start

```bash
# Train a GPT-style model on FineWeb10B
python -m src.tinyllm.train \
    -c src/tinyllm/configs/training/train_config.yaml \
    -m src/tinyllm/configs/models/gpt.yaml

# Train a Qwen3-style model (same tokenizer/vocab, different architecture)
python -m src.tinyllm.train \
    -c src/tinyllm/configs/training/train_config.yaml \
    -m src/tinyllm/configs/models/qwen3.yaml
```

## Installation

```bash
# Create environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: flash-attn for variable-length varlen
pip install flash-attn

# Optional: liger-kernel for fused Triton kernels
pip install liger-kernel
```

### Environment Variables

Create `.env` at the project root:

```env
WANDB_API_KEY=your-wandb-key
HF_TOKEN=your-huggingface-token  # only needed for gated models
```

## Data

Uses the [FineWeb10B GPT-2 pretokenized dataset](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2).

```bash
# Download validation chunk + 1 training chunk
python -m src.tinyllm.utils.fineweb10b_downloader 1
```

Then update `train_file_pattern` / `test_file_pattern` in your training config to point to the `.bin` files.

## Architecture

```
src/tinyllm/
├── train.py                  # Training entry point
├── configs/
│   ├── models/               # Model architecture YAMLs
│   │   ├── gpt.yaml          # MHA + StandardFFN + LayerNorm
│   │   └── qwen3.yaml        # GQA + GatedFFN + RMSNorm + RoPE
│   └── training/             # Training hyperparameter YAMLs
├── models/
│   ├── builder.py            # DynamicModel — single class, all architectures
│   ├── base.py               # Weight init, causal mask, block-causal mask
│   └── layers/
│       ├── transformer_block.py  # Single block, resolves attn/ffn/norm from registry
│       ├── rope.py               # Rotary Positional Embeddings
│       ├── generate_qkv.py       # Fused QKV projection
│       └── normalization.py      # LayerNorm, RMSNorm
├── layers/
│   ├── registry.py           # LAYER_REGISTRY for all components
│   ├── attention/
│   │   ├── mha.py            # Multi-Head Attention (cu_seqlens + flash support)
│   │   └── gqa.py            # Grouped Query Attention (cu_seqlens + flash support)
│   ├── ffn/
│   │   ├── standard.py       # Linear → Activation → Linear
│   │   └── gated.py          # SwiGLU/GeGLU gated FFN
│   ├── normalization/
│   │   ├── layer_norm.py     # nn.LayerNorm wrapper
│   │   └── rms_norm.py       # RMSNorm
│   └── embeddings/
│       ├── learned_pe.py     # Learned positional embeddings
│       ├── sinusoidal_pe.py  # Sinusoidal positional embeddings
│       └── rope_only.py      # No-op (RoPE handled inside attention)
├── datasets/
│   └── fineweb_helper.py     # Varlen-packed + fixed-batch dataloader for .bin files
├── trainer/
│   └── trainer.py            # Training loop with AMP, grad accum, cu_seqlens
├── optimizer/
│   └── muon.py               # Muon optimizer + Muon-AdamW hybrid
├── analysis/
│   ├── spectral.py           # SVD, variance explained, WeightWatcher alpha
│   ├── weight_stats.py       # Weight mean/std/min/max/L2/sparsity
│   └── activation_stats.py   # Activation capture + stats
├── callbacks/
│   ├── wandb_callback.py     # WandB logging (config, metrics, watch)
│   ├── analysis_callback.py  # Gradient/weight/spectral/activation analysis
│   ├── benchmark_callback.py # HellaSwag eval
│   └── checkpoint_callback.py# Checkpoint save/load
├── benchmarks/
│   ├── hellaswag.py          # HellaSwag evaluation
│   ├── base.py               # Benchmark base class
│   └── runner.py             # Benchmark runner
├── factory/
│   ├── factory.py            # model_factory, optimizer_factory, lr_scheduler_factory
│   └── registry.py           # MODEL_REGISTRY
└── utils/
    ├── misc.py               # Config parser (Box), tokenizer, path helpers
    ├── kernels.py            # liger-kernel + flash-attn integration
    ├── train_utils.py        # autocast context
    └── fineweb10b_downloader.py  # FineWeb10B .bin downloader
```

## Model Config

All architectures use the same `dynamic` model type. What changes is the `blocks` section:

### GPT-style (MHA + StandardFFN + LayerNorm + RoPE)

```yaml
model_type: "dynamic"
tokenizer_name: "gpt2"
d_model: 256
vocab_size: 50304            # GPT-2 vocab (50257) padded to nearest 64
max_seq_len: 1024
embedding: "rope_only"       # RoPE inside attention, no learned position table

blocks:
  count: 4
  attention: "mha"
  ffn: "standard"
  norm: "layer_norm"
  num_heads: 4
  ff_multiplier: 4
  flash: false               # use flash_attn varlen if true + package installed

head:
  norm: "layer_norm"
  tie_weights: true
```

### Qwen3-style (GQA + GatedFFN + RMSNorm + RoPE)

```yaml
model_type: "dynamic"
tokenizer_name: "gpt2"       # same tokenizer, same vocab
d_model: 256
vocab_size: 50304
max_seq_len: 1024
embedding: "rope_only"

blocks:
  count: 4
  attention: "gqa"
  ffn: "gated"
  norm: "rms"
  num_heads: 8
  num_kv_heads: 2
  ff_multiplier: 4
  use_qk_norm: true

head:
  norm: "rms"
  tie_weights: true
```

## Training Config

```yaml
model_type: "gpt"
exp_path: "gpt_training"
mode: "varlen_packed"            # or "fixed_batch"
train_file_pattern: "data/fineweb_train_*.bin"
test_file_pattern: "data/fineweb_val_*.bin"

packed_tokens: 1024              # tokens per batch (must match model max_seq_len)
max_seq_len: 1024
device: "cuda:0"

optimizer:
  type: "muon_adamw"             # Muon for 2D, AdamW for 1D/biases/norms
  muon_lr: 0.0002
  adamw_lr: 0.001
  weight_decay: 0.1

init_lr: 0.001
num_training_steps: 100_000
lr_scheduler_type: "cosine"

fp16_training: true
use_grad_accum: true
iters_to_accumulate: 4

use_wandb: true
log_every: 10

max_batches: 0                   # limit for testing (0 = unlimited)
max_tokens: 0

analysis:
  every_n_steps: 500
  track_weights: true
  track_gradients: true
  spectral: false

kernels:
  use_liger: false               # requires: pip install liger-kernel
  use_flash_attn: false          # requires: pip install flash-attn
```

## Features

### Dynamic Model
- Single `DynamicModel` class reads any architecture from YAML config
- No per-model Python files — all variants via config
- `LAYER_REGISTRY` resolves attention, FFN, norm, embedding by name

### Attention (two paths)
- **No flash** (`flash: false`): `F.scaled_dot_product_attention` with block-diagonal causal mask built from `cu_seqlens` — correct document-boundary masking in varlen mode
- **flash-attn package** (`flash: true` + `pip install flash-attn`): `flash_attn_varlen_func` with native varlen support

### Optimizer
- **Muon**: Newton-Schulz preconditioning for 2D weight matrices
- **AdamW**: all 1D / bias / norm / embedding parameters
- `_CombinedOptimizer` wraps both as a single `torch.optim.Optimizer` (compatible with LR schedulers)

### fp8 training (H100 / Ada SM89+)
Opt-in fp8 matmuls via [torchao float8](https://github.com/pytorch/ao). It is an
**add-on to the bf16 path**, not a replacement — master weights and the optimizer
stay fp32, bf16 autocast still wraps everything, and only the big Linear matmuls
run in fp8. Enable it in the training config:

```yaml
precision: "bf16"        # keep bf16; fp8 is orthogonal
fp8:
  enabled: true          # requires torchao + a Hopper/Ada GPU (SM89+)
  recipe: "tensorwise"   # tensorwise (fastest) | rowwise (safer numerics)
  filter_fqns: ["lm_head"]  # kept in bf16 (also any Linear with dims not %16)
```

Requires `pip install torchao` (in `requirements.txt` / `setup.sh`). It is a safe
**no-op** on non-Hopper GPUs, CPU, or when torchao is absent — it logs a warning
and trains in plain bf16. If the fp8 loss curve drifts from your bf16 baseline,
switch `recipe` to `rowwise`.

### Data Pipeline
- Varlen-packed mode: documents concatenated into fixed-length batches with `cu_seqlens`
- Fixed-batch mode: traditional `(B, S)` batches
- Memory-mapped I/O, async BOS scanning, configurable limits (`max_batches`, `max_tokens`)

### Analysis & Observability
- **WandB**: loss, LR, tokens/sec each step; weight histograms every 100 steps
- **Weight stats**: mean, std, min, max, L2, sparsity per parameter
- **Gradient norms**: per-parameter via backward hooks (captured before zero_grad)
- **Spectral analysis**: SVD, variance explained ratio, condition number
  - Condition number = σ₀ / σₙ (largest ÷ smallest singular value)
  - ≈ 1 → isotropic, well-conditioned; >> 1 → near-singular (training instability risk); ∞ → rank-deficient
  - Logged per 2D weight layer under `spectral/{name}/condition_number`
- **WeightWatcher alpha**: power-law tail exponent of the eigenvalue spectrum per 2D layer
  - α < 2 → heavy-tailed (structured features, potentially overfit)
  - α 2–4 → well-trained, good generalization (optimal)
  - α > 6 → near random initialization (undertrained)
  - Computed via Hill estimator + KS minimization on the eigenvalue distribution
- **Timing**: per-analysis latency logged to WandB
- **HellaSwag**: periodic evaluation benchmark

### Testing

```bash
# Quick smoke test on random data (no download needed)
python -m src.tinyllm.test_run

# Limited data test
# Add to train_config.yaml:
#   max_batches: 5
#   max_tokens: 50000
```

## Key Design Decisions

| Decision | Choice |
|---|---|
| Single model class vs per-model files | `DynamicModel` only — `gpt.py` and `qwen3.py` deleted |
| Attention return convention | Always `(output, metadata_dict)` with q, k, v, attn_weights (or None if flash), qkv_out, o_proj_out |
| Gradient capture | `register_hook` on each param (fires during backward, stored before zero_grad) |
| Embedding for varlen | `rope_only` — no learned position table; RoPE applied inside attention per-document |
| Tokenizer | GPT-2 for all architectures (FineWeb10B is GPT-2 pretokenized) |
| Vocab size | 50304 = GPT-2's 50257 padded to nearest multiple of 64 |
| Positional encoding | RoPE applied per-document inside attention layers |
