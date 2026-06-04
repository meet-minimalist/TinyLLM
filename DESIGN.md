# TinyLLMs — Architecture & Design

## Overview

TinyLLMs is a framework for training small-scale decoder-only language models.
It is designed for **analysis and experimentation**: every intermediate activation,
weight matrix, and spectral property can be logged to Weights & Biases for
post-hoc analysis.

## Directory Layout

```
src/tinyllm/
├── train.py                     # Entry point — wires everything together
├── infer.py                     # Inference / text generation
│
├── configs/
│   ├── models/                  # Per-architecture YAML configs
│   │   ├── gpt.yaml             # GPT-style (learned PE + MHA + standard FFN)
│   │   └── qwen3.yaml           # Qwen3-style (RoPE + GQA + gated FFN + RMSNorm)
│   └── training/                # Training hyperparameter configs
│       └── train_config.yaml
│
├── models/
│   ├── base.py                  # BaseLLM — shared init_weights, forward helpers
│   └── builder.py               # DynamicModel — reads any config, builds model
│
├── layers/                      # Modular, registered layer components
│   ├── __init__.py              # Registers all layers on import
│   ├── registry.py              # LAYER_REGISTRY — decorator-based registration
│   ├── transformer_block.py     # Generic block; resolves attn/ffn/norm from registry
│   ├── attention/
│   │   ├── mha.py               # Multi-Head Attention (registered "mha")
│   │   └── gqa.py               # Grouped Query Attention (registered "gqa")
│   ├── ffn/
│   │   ├── standard_ffn.py      # Standard FFN (registered "standard")
│   │   └── gated_ffn.py         # Gated FFN / SwiGLU (registered "gated")
│   ├── normalization/
│   │   ├── layernorm.py         # LayerNorm (registered "layer_norm")
│   │   └── rmsnorm.py           # RMSNorm (registered "rms")
│   └── embeddings/
│       ├── learned_pe.py        # Learnable pos embeddings (registered "learned_pe")
│       ├── sinusoidal_pe.py     # Sinusoidal PE (registered "sinusoidal")
│       └── rope_only.py         # No-op PE, for RoPE-only models (registered "rope_only")
│
├── datasets/
│   ├── fineweb_helper.py        # FineWeb pretokenized binary loader
│   └── dataset_helper.py        # HuggingFace dataset loader
│
├── optimizer/
│   ├── muon.py                  # Muon + MuonAdamW hybrid optimizer
│   └── __init__.py
│
├── analysis/                    # Weight & activation analysis toolkit
│   ├── spectral.py              # SVD, eigenvalues, variance explained, WeightWatcher alpha
│   ├── weight_stats.py          # Weight/gradient norm, sparsity, min/max/mean/std
│   ├── activation_stats.py      # Activation hooking & statistics
│   └── __init__.py
│
├── benchmarks/
│   ├── base.py                  # BaseBenchmark ABC
│   ├── hellaswag.py             # HellaSwag (subsampled)
│   ├── runner.py                # Runs enabled benchmarks
│   └── __init__.py
│
├── callbacks/
│   ├── base_callback.py
│   ├── callback_handler.py
│   ├── checkpoint_callback.py
│   ├── wandb_callback.py
│   ├── analysis_callback.py     # Periodic gradient/weight/spectral logging
│   ├── benchmark_callback.py    # Periodic benchmark evaluation
│   └── __init__.py
│
├── trainer/
│   ├── trainer.py               # Main training loop
│   └── __init__.py
│
├── factory/
│   ├── registry.py              # MODEL_REGISTRY infrastructure
│   ├── factory.py               # model_factory, optimizer_factory, lr_scheduler_factory
│   └── __init__.py
│
├── loss_fn/
│   └── loss_helper.py           # Cross-entropy loss
│
├── logger/
│   └── logger_utils.py          # Logging configuration
│
└── utils/
    ├── misc.py                  # Config parser, tokenizer helper, exp path
    ├── train_utils.py           # AMP context helper
    ├── kernels.py               # Optional liger-kernel / flash-attn integration
    └── ww_utils.py              # WeightWatcher alpha (reference implementation)
```

## Core Design Principles

### 1. Layer Registry (`layers/registry.py`)

Every layer component (attention, FFN, normalization) registers itself via
decorator:

```python
@LAYER_REGISTRY.register("mha")
class MHA(nn.Module):
    ...
```

The `TransformerBlock` resolves components dynamically:

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        attn_cls = LAYER_REGISTRY.get(config["attention"])
        self.attn = attn_cls(**attn_kwargs)
```

To add a new attention variant (e.g., MLA), just create a new file in
`layers/attention/`, register it, and reference it in the config YAML.

### 2. Config-Driven Model Building (`models/builder.py`)

The model config specifies every architectural choice as strings that map to
registered layers:

```yaml
model:
  vocab_size: 50304
  d_model: 256
  max_seq_len: 1024
  embedding: "learned_pe"          # registered in LAYER_REGISTRY
  blocks:
    count: 4
    attention: "gqa"               # resolved from LAYER_REGISTRY
    ffn: "gated"                   # resolved from LAYER_REGISTRY
    norm: "rms"                    # resolved from LAYER_REGISTRY
    num_heads: 8
    num_kv_heads: 2
    ff_multiplier: 4
  head:
    norm: "rms"
    tie_weights: true
```

`build_model_from_config()` reads this YAML and constructs the full model
without any hardcoded architecture decisions.

### 3. Attention Layer Return Convention

All attention layers return a consistent `(output, metadata_dict)` tuple:

```python
class MHA(nn.Module):
    def forward(self, x, mask=None, cos=None, sin=None):
        ...
        return output, {
            "q": q, "k": k, "v": v,
            "attn_weights": attn_weights,
            "qkv_out": qkv_out,
            "o_proj_out": o_proj_output,
        }
```

This allows the analysis callbacks to capture intermediates uniformly using
forward hooks, without special-casing per attention type.

### 4. Muon + AdamW Hybrid Optimizer (`optimizer/muon.py`)

- **Muon** is applied to all 2D weight matrices (nn.Linear weights without bias)
- **AdamW** is applied to everything else (1D params, biases, embeddings, norms)

The optimizer factory splits parameters automatically:

```python
def MuonAdamW(model, muon_lr, adamw_lr, weight_decay, ...):
    muon_params = [p for name, p in model.named_parameters()
                   if p.ndim >= 2 and "bias" not in name]
    adamw_params = [p for name, p in model.named_parameters()
                    if p.ndim < 2 or "bias" in name]
    return torch.optim.AdamW(...)  # with Muon step applied to muon_params
```

### 5. WandB & Analysis Pipeline (`analysis/`, `callbacks/analysis_callback.py`)

Three levels of tracking, all running at the same
`analysis_every_n_steps` interval:

| Level | What | Cost |
|-------|------|------|
| Weight stats | Per-weight-matrix: mean, std, min, max, norm, sparsity | Cheap (O(n params)) |
| Activation stats | Per-layer: mean, std, min, max, norm of activations | Medium (requires forward hook) |
| Spectral analysis | Per-weight-matrix: SVD → eigenvalues, variance explained (95%/99%), condition number, WeightWatcher alpha | Expensive (O(n³)) |

Each component logs its own timing to WandB so you can identify bottlenecks.

### 6. WeightWatcher Alpha (`analysis/spectral.py:compute_weightwatcher_alpha`)

The WeightWatcher α (alpha) exponent measures the "heavy-tailedness" of a
weight matrix's eigenvalue distribution. It is computed via the Hill estimator
with KS-minimization on the empirical spectral density.

**Interpretation of α:**

| α range | Interpretation |
|---------|---------------|
| α < 2   | Very heavy-tailed. Many large eigenvalues → high redundancy / overfitting risk |
| 2–4     | Reasonable fat tails. Healthy training signal |
| 4–10    | Moderate tails. May be undertrained |
| > 10    | Near-white-noise spectrum (random initialization). α ≈ 48 for randn(50304, 256) |
| → ∞     | Perfect white noise, no learned structure |

**What α reveals during training:**

- **α decreasing over time** → the layer is learning meaningful structure
- **α drops sharply** → possible phase transition or optimization instability
- **α very low in early layers** → early layers specialize in general patterns
- **α very low in late layers** → late layers overfit to training data
- **α stays high** → the layer isn't learning effectively (dead layer, wrong LR, etc.)

Monitor `spectral/{layer_name}/alpha` in WandB across training steps.

### 7. Gradient Norm Tracking (`analysis/weight_stats.py:compute_gradient_norms`)

Gradient norms are captured via **backward hooks** (not after zero_grad). Each
parameter registers a hook during `backward()` that stores its gradient norm.
At analysis time, the callback computes:

- `gradients/global/gradient_norm`: L2 norm of all gradients (key metric)
- `gradients/global/exploded_gradients`: count of params with grad norm > 1e4
- `gradients/global/zero_gradients`: count of params with grad norm = 0
- `gradients/layer/{name}`: per-parameter gradient norm

**What to look for in WandB:**
- **Exploding gradients**: `exploded_gradients > 0` → gradient clipping needed
- **Vanishing gradients**: `global_gradient_norm` trending to 0 → activations dying
- **Spiky gradient norm**: optimization instability, lower LR or increase warmup

### 8. Flash Attention

Two levels of flash attention support, controlled by the `flash` parameter
in attention layer config:

1. **PyTorch SDPA** (`flash: true`, default mechanism):
   Uses `torch.nn.functional.scaled_dot_product_attention` with
   `is_causal=True`. On CUDA with compatible GPUs, this automatically
   dispatches to flash attention kernels (memory-efficient, O(1) in seq len).

2. **flash-attn package** (optional, via `utils/kernels.py`):
   The `flash_attn` package provides more flexible attention (sliding window,
   ALiBi, etc.). Install with `pip install flash-attn` and use via the
   kernel integration module.

When `flash: true`, `attn_weights` is `None` in the metadata dict (the full
attention matrix is never materialized). When `flash: false`, the full
attention matrix is returned for analysis.

### 9. Optional Triton Kernel Integration (`utils/kernels.py`)

The `utils/kernels.py` module provides optional integration with:
- **liger-kernel**: fused CrossEntropy, RoPE, SwiGLU, RMSNorm, LayerNorm
- **flash-attn**: faster attention

Both are optional (not in core requirements). Install with:
```
pip install liger-kernel flash-attn
```

Set config:
```yaml
kernels:
  use_flash_attn: true    # uses F.scaled_dot_product_attention
  use_liger: true         # patches model with liger fused ops
```

### 10. Dataset Design

Currently supports two data sources, selected by config:

- **FineWeb pretokenized binary** (`fineweb_helper.py`): Ultra-fast, pre-tokenized
  shards in `.bin` format. Supports varlen_packed and fixed_batch modes.
- **HuggingFace datasets** (`dataset_helper.py`): Tokenizes on-the-fly or
  pre-tokenizes with caching.

To switch datasets, change the training config (no code changes needed).

## How to Add a New Feature

### New Attention Variant
1. Create `layers/attention/my_attn.py`
2. Decorate class with `@LAYER_REGISTRY.register("my_attn")`
3. Implement `__init__(self, emb_dim, num_heads, drop_prob=0.0, flash=False, **kwargs)`
4. Implement `forward(self, x, mask=None, cos=None, sin=None) -> (output, metadata_dict)`
5. Reference it in config: `attention: "my_attn"`

### New FFN Variant
1. Create `layers/ffn/my_ffn.py`
2. Decorate class with `@LAYER_REGISTRY.register("my_ffn")`
3. Implement `__init__(self, emb_dim, ff_multiplier=4, drop_rate=0.0, **kwargs)`
4. Implement `forward(self, x) -> Tensor`
5. Reference it in config: `ffn: "my_ffn"`

### New Model Architecture
No Python code needed — just create a new YAML config combining existing layer types.
Add a new config YAML under `configs/models/` and run with `model_type: "dynamic"`.

### New Benchmark
1. Create `benchmarks/my_bench.py`
2. Subclass `BaseBenchmark`, implement `run()`
3. Add config entry in training YAML

### New Dataset Source
1. Implement a class that yields `(input_ids, targets, ...)` batches
2. Import and use it in `train.py` (swap out the dataloader)

## Config Reference

### Model Config (e.g., `configs/models/gpt.yaml`)
| Field | Type | Description |
|-------|------|-------------|
| `model_type` | str | Must be `"dynamic"` |
| `tokenizer_name` | str | HuggingFace tokenizer name |
| `vocab_size` | int | Vocabulary size |
| `d_model` | int | Model dimension |
| `max_seq_len` | int | Maximum sequence length |
| `embedding` | str | `"learned_pe"`, `"rope_only"`, or `"sinusoidal"` |
| `blocks.count` | int | Number of transformer layers |
| `blocks.attention` | str | `"mha"` or `"gqa"` |
| `blocks.ffn` | str | `"standard"` or `"gated"` |
| `blocks.norm` | str | `"layer_norm"` or `"rms"` |
| `blocks.num_heads` | int | Number of attention heads |
| `blocks.num_kv_heads` | int | KV heads for GQA |
| `blocks.ff_multiplier` | int | FFN hidden dim multiplier |
| `blocks.flash` | bool | Use flash attention (default: false) |
| `blocks.use_qk_norm` | bool | QK layer norm (default: false) |
| `blocks.drop_prob` | float | Dropout probability |
| `blocks.act_fn` | str | `"gelu"`, `"swish"`, or `"relu"` |
| `head.norm` | str | Final norm type |
| `head.tie_weights` | bool | Tie LM head with embedding |

### Training Config (e.g., `configs/training/train_config.yaml`)
| Field | Type | Description |
|-------|------|-------------|
| `model_type` | str | Must match model config |
| `mode` | str | `"varlen_packed"` or `"fixed_batch"` |
| `train_file_pattern` | str | Glob for training shards |
| `test_file_pattern` | str | Glob for validation shards |
| `optimizer.type` | str | `"muon_adamw"` or `"adamw"` |
| `optimizer.muon_lr` | float | Learning rate for Muon params |
| `optimizer.adamw_lr` | float | Learning rate for AdamW params |
| `optimizer.muon_momentum` | float | Momentum for Muon (default: 0.95) |
| `optimizer.weight_decay` | float | Weight decay |
| `analysis.every_n_steps` | int | How often to run analysis |
| `analysis.track_weights` | bool | Log weight statistics |
| `analysis.track_gradients` | bool | Log gradient norms (via backward hooks) |
| `analysis.spectral` | bool | Enable SVD-based spectral analysis |
| `analysis.track_alpha` | bool | Compute WeightWatcher alpha |
| `analysis.variance_thresholds` | list | Variance thresholds for spectral (e.g. [0.95, 0.99]) |
| `analysis.track_activations` | bool | Log activation statistics |
| `benchmarks.<name>.enabled` | bool | Enable benchmark |
| `benchmarks.<name>.num_samples` | int | Subsample size |
| `benchmarks.<name>.every_n_steps` | int | Run interval |
| `lr_scheduler_type` | str | `"cosine"`, `"linear"`, `"constant"`, etc. |

## Timing & Performance

Every analysis component logs its duration to WandB as
`time/{component_name}` (e.g., `time/spectral_analysis`,
`time/benchmark_hellaswag`). Monitor these to adjust the analysis frequency.
