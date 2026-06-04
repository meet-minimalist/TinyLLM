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
│   │   ├── gpt.yaml
│   │   └── qwen3.yaml
│   └── training/                # Training hyperparameter configs
│       └── train_config.yaml
│
├── models/                      # Full model definitions
│   ├── base.py                  # BaseLLM — shared init_weights, forward helpers
│   ├── builder.py               # build_model_from_config() — dynamic composition
│   ├── gpt.py                   # GPT wrapper (delegates to builder)
│   └── qwen3.py                 # Qwen3 wrapper (delegates to builder)
│
├── layers/                      # Modular, registered layer components
│   ├── __init__.py              # Exports all layers & TransformerBlock
│   ├── registry.py              # LAYER_REGISTRY — decorator-based registration
│   ├── transformer_block.py     # Generic block; resolves attn/ffn/norm from registry
│   │
│   ├── attention/
│   │   ├── mha.py               # Multi-Head Attention (registered as "mha")
│   │   └── gqa.py               # Grouped Query Attention (registered as "gqa")
│   │
│   ├── ffn/
│   │   ├── standard_ffn.py      # Standard FFN (registered as "standard")
│   │   └── gated_ffn.py         # Gated FFN / SwiGLU (registered as "gated")
│   │
│   ├── normalization/
│   │   ├── layernorm.py         # torch.nn.LayerNorm wrapper (registered as "layer_norm")
│   │   └── rmsnorm.py           # RMSNorm (registered as "rms")
│   │
│   └── embeddings/
│       ├── learned_pe.py        # Learnable positional embeddings
│       ├── sinusoidal_pe.py     # Sinusoidal positional embeddings
│       └── rope.py              # Rotary Positional Embedding
│
├── datasets/
│   ├── fineweb_helper.py        # FineWeb pretokenized binary loader
│   ├── dataset_helper.py        # HuggingFace dataset loader
│   └── __init__.py
│
├── optimizer/
│   ├── muon.py                  # Muon optimizer + MuonAdamW hybrid
│   └── __init__.py
│
├── analysis/                    # Weight & activation analysis toolkit
│   ├── spectral.py              # SVD, eigenvalue decomposition, variance explained
│   ├── weight_stats.py          # Weight norm, sparsity, min/max/mean/std
│   ├── activation_stats.py      # Activation hooking & statistics
│   └── __init__.py
│
├── benchmarks/                  # Lightweight downstream eval
│   ├── base.py                  # BaseBenchmark ABC
│   ├── hellaswag.py             # HellaSwag (subsampled, ~200 examples)
│   ├── runner.py                # Runs enabled benchmarks at intervals
│   └── __init__.py
│
├── callbacks/                   # Training lifecycle hooks
│   ├── base_callback.py
│   ├── callback_handler.py
│   ├── checkpoint_callback.py
│   ├── wandb_callback.py
│   ├── analysis_callback.py     # Periodic spectral/weight/activation logging
│   ├── benchmark_callback.py    # Periodic benchmark evaluation
│   └── __init__.py
│
├── trainer/
│   ├── trainer.py               # Main training loop
│   └── __init__.py
│
├── factory/
│   ├── registry.py              # Shared MODEL_REGISTRY + LAYER_REGISTRY infrastructure
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
    └── ww_utils.py              # WeightWatcher alpha computation (reference)
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

### 6. Spectral Analysis (`analysis/spectral.py`)

For a weight matrix W of shape (M, N):

```python
U, S, Vt = torch.linalg.svd(W.float())
evals = S ** 2
total = evals.sum()
ratio = evals.cumsum(0) / total
n_95 = int((ratio < 0.95).sum()) + 1    # eigenvalues to explain 95% variance
n_99 = int((ratio < 0.99).sum()) + 1    # eigenvalues to explain 99% variance
```

This directly measures activation/weight redundancy — if 95% variance is
explained by 10% of eigenvalues, the matrix is highly redundant.

### 7. Lightweight Benchmarks (`benchmarks/`)

Benchmarks run every `benchmark_every_n_steps` with a small subsample:

- **HellaSwag**: ~200 examples, ~10–20 seconds
- Configurable per-benchmark: `num_samples`, `every_n_steps`

Each benchmark implements the `BaseBenchmark` interface:

```python
class BaseBenchmark(ABC):
    @abstractmethod
    def run(self, model, tokenizer, device) -> dict:
        """Returns {"accuracy": ..., "loss": ...}"""
```

### 8. Dataset Design

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
3. Return `(output, metadata_dict)` from `forward()`
4. Reference it in config: `attention: "my_attn"`

### New Benchmark
1. Create `benchmarks/my_bench.py`
2. Subclass `BaseBenchmark`, implement `run()`
3. Add config entry in training YAML
4. Register in `benchmarks/runner.py`

### New Dataset Source
1. Implement a class that yields `(input_ids, targets, ...)` batches
2. Import and use it in `train.py` (swap out the dataloader)

## Config Reference

### Model Config (e.g., `configs/models/gpt.yaml`)
| Field | Type | Description |
|-------|------|-------------|
| `model_type` | str | Registered model name ("gpt", "qwen3") |
| `tokenizer_name` | str | HuggingFace tokenizer name |
| `vocab_size` | int | Vocabulary size |
| `d_model` | int | Model dimension |
| `max_seq_len` | int | Maximum sequence length |
| `embedding` | str | Embedding type ("learned_pe", "rope_only") |
| `blocks.count` | int | Number of transformer layers |
| `blocks.attention` | str | Attention type ("mha", "gqa") |
| `blocks.ffn` | str | FFN type ("standard", "gated") |
| `blocks.norm` | str | Normalization type ("layer_norm", "rms") |
| `blocks.num_heads` | int | Number of attention heads |
| `blocks.num_kv_heads` | int | KV heads for GQA |
| `blocks.ff_multiplier` | int | FFN hidden dim multiplier |
| `head.norm` | str | Final norm type |
| `head.tie_weights` | bool | Tie LM head with embedding |

### Training Config (e.g., `configs/training/train_config.yaml`)
| Field | Type | Description |
|-------|------|-------------|
| `model_type` | str | Must match model config |
| `mode` | str | "varlen_packed" or "fixed_batch" |
| `train_file_pattern` | str | Glob for training shards |
| `test_file_pattern` | str | Glob for validation shards |
| `optimizer.type` | str | "muon_adamw" or "adamw" |
| `analysis.every_n_steps` | int | How often to run analysis |
| `analysis.spectral` | bool | Enable SVD-based spectral analysis |
| `benchmarks` | dict | Per-benchmark config |
| `lr_scheduler_type` | str | "cosine", "linear", "constant", etc. |

## Timing & Performance

Every analysis component logs its duration to WandB as
`time/{component_name}` (e.g., `time/spectral_analysis`,
`time/benchmark_hellaswag`). Monitor these to adjust the analysis frequency.
