# TinyLLMs

Lightweight, from-scratch implementations of popular LLM architectures for research and learning. Build models like GPT-2, Qwen, LLaMA, and Mistral from YAML configs using a unified, modular building-block system.

## Quick Start

```bash
# Train GPT on TinyStories
python -m src.tinyllm.train \
    -c src/tinyllm/configs/training/train_config.yaml \
    -m src/tinyllm/configs/models/gpt.yaml

# Inference with a trained checkpoint
python -m src.tinyllm.infer \
    -m src/tinyllm/configs/models/gpt.yaml \
    -c checkpoints/model.pt \
    -p "Once upon a time" \
    --max_tokens 100
```

## Installation

```bash
pip install -r requirements.txt
```

## Architecture

Models are assembled from reusable, composable building blocks. A single unified `TransformerBlock` handles all architecture variants through a config dict:

```yaml
# GPT-2 style
attention_type: "mha"
ffn_type: "standard"
normalization: "layer_norm"

# Qwen / LLaMA style
attention_type: "gqa"
ffn_type: "gated"
normalization: "rms"
```

```
src/tinyllm/
├── train.py                  # Training entry point
├── infer.py                  # Inference entry point
├── callbacks/                # Checkpoint, WandB lifecycle hooks
├── configs/
│   ├── models/               # Model architecture YAMLs
│   │   ├── gpt.yaml
│   │   └── qwen3.yaml
│   └── training/             # Training hyperparameter YAMLs
├── datasets/                 # Dataset loading + bucketed batching
├── factory/                  # Model & LR scheduler registry
├── logger/                   # Logging utilities
├── loss_fn/                  # Cross-entropy loss helpers
├── models/
│   ├── gpt.py                # GPT model (thin config wrapper)
│   ├── qwen3.py              # Qwen3 model (thin config wrapper)
│   └── layers/               # Reusable building blocks
│       ├── transformer_block.py   # Single unified decoder block
│       ├── embeddings.py          # Learnable + sinusoidal PE
│       ├── rope.py                # Rotary Positional Embeddings
│       ├── normalization.py       # RMSNorm
│       └── generate_qkv.py        # Q/K/V projection
├── trainer/                  # Training loop with AMP + grad accum
└── utils/                    # Config parsing, tokenizer, misc
```

## Features

### Models & Architectures
- [x] GPT-2 style decoder-only transformer
- [x] Qwen3 / LLaMA style with GQA + RoPE + SwiGLU
- [x] Unified `TransformerBlock` — one class for all architecture variants
- [x] Config-driven model construction from YAML

### Attention Mechanisms
- [x] Multi-Head Attention (MHA)
- [x] Grouped Query Attention (GQA)
- [x] Rotary Positional Embeddings (RoPE)
- [x] Learnable positional embeddings
- [x] Sinusoidal positional embeddings
- [x] Causal masking
- [x] Optional QK normalization

### Feed-Forward Networks
- [x] Standard FFN (Linear → Activation → Linear)
- [x] Gated FFN (SwiGLU / GeGLU)
- [x] Configurable activation functions (GELU, ReLU, SiLU)

### Normalization
- [x] Layer Normalization
- [x] RMSNorm

### Training
- [x] Config-driven training from YAML
- [x] Automatic mixed precision (AMP / FP16)
- [x] Gradient accumulation
- [x] Cosine, linear, constant, inverse-sqrt LR schedulers with warmup
- [x] Label smoothing
- [x] Bucketed batching (group similar-length sequences)
- [x] Cross-entropy loss with ignore index

### Observability
- [x] Weights & Biases integration
- [x] File-based logging
- [x] Model summary (torchinfo)
- [x] Checkpoint management (keep last N)

### Inference
- [x] Autoregressive token-by-token generation
- [x] Checkpoint loading

## Usage

### Define a Model

Create a YAML file in `configs/models/`:

```yaml
model_type: "gpt"
tokenizer_name: "gpt2"

vocab_size: 50304
emb_dim: 768
max_seq_len: 1024
num_heads: 12
drop_prob: 0.1
ff_multiplier: 4
num_blocks: 12
tie_weights: true
act_fn: "gelu"
```

Or for a Qwen3-style model:

```yaml
model_type: "qwen3"
tokenizer_name: "Qwen/Qwen2.5-0.5B"

d_model: 768
vocab_size: 50304
max_seq_len: 2048
num_layers: 12
num_heads: 12
num_kv_heads: 4
drop_rate: 0.1
ffn_multiplier: 4
tie_word_embeddings: true
use_qk_norm: true
ffn_act: "swish"
```

### Define Training Config

```yaml
model_type: "gpt"
exp_path: "experiments"
num_epochs: 10
batch_size: 32
max_seq_len: 1024
dataset_name: "roneneldan/TinyStories"
lr_scheduler_type: "cosine"
init_lr: 3e-4
warmup_epochs: 1
device: "cuda:0"
fp16_training: true
use_grad_accum: true
iters_to_accumulate: 4
use_wandb: true
```

### Register a Custom Model

```python
from src.tinyllm.factory.registry import MODEL_REGISTRY
from src.tinyllm.models.layers import TransformerBlock, RMSNorm, RotaryPositionalEmbedding

@MODEL_REGISTRY.register("my_model")
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Assemble from building blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
```

## TODO

- [ ] Flash Attention integration
- [ ] Mistral / Sliding Window Attention
- [ ] Mixture of Experts (MoE)
- [ ] Evaluation harness (MMLU, HellaSwag, etc.)
- [ ] Pretraining on larger datasets (FineWeb, Dolma)
- [ ] Resume training from checkpoint
