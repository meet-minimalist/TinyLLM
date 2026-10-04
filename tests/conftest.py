import os
import sys

import numpy as np
import pytest
import torch
from box import Box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.tinyllm.datasets.fineweb_helper import (  # noqa: E402
    HEADER_MAGIC,
    HEADER_SIZE,
    HEADER_VERSION,
)

BOS = 50256
VOCAB = 256


def write_shard(path, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0], header[1], header[2] = HEADER_MAGIC, HEADER_VERSION, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def make_tokens(seed: int, n_docs: int) -> np.ndarray:
    """Documents of random length, each starting with BOS. Token ids stay
    below VOCAB except BOS, which the tiny test model maps via modulo."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_docs):
        length = int(rng.integers(5, 120))
        docs.append(np.concatenate([[BOS], rng.integers(0, VOCAB - 1, length)]))
    return np.concatenate(docs)


@pytest.fixture
def shards(tmp_path):
    """Three small train shards and one val shard."""
    for i in range(3):
        write_shard(tmp_path / f"train_{i}.bin", make_tokens(i, 60))
    write_shard(tmp_path / "val_0.bin", make_tokens(99, 30))
    return tmp_path


def tiny_model_config() -> Box:
    return Box(
        {
            "model_type": "dynamic",
            "name": "tiny",
            "tokenizer_name": "gpt2",
            "d_model": 32,
            # BOS (50256) is a real token id in the shards, so the embedding
            # must cover it.
            "vocab_size": 50304,
            "max_seq_len": 64,
            "embedding": "rope_only",
            "blocks": {
                "count": 2,
                "attention": "gqa",
                "num_heads": 4,
                "num_kv_heads": 2,
                "ffn": "gated",
                "norm": "rms",
                "ff_multiplier": 2,
                "act_fn": "swish",
                "use_qk_norm": True,
            },
            "head": {"norm": "rms", "tie_weights": True},
        }
    )


def tiny_train_config(shard_dir, **overrides) -> Box:
    cfg = {
        "mode": "varlen_packed",
        "train_file_pattern": str(shard_dir / "train_*.bin"),
        "test_file_pattern": str(shard_dir / "val_*.bin"),
        "packed_tokens": 128,
        "max_seq_len": 64,
        "device": "cpu",
        "num_workers": 0,
        "prefetch_queue_size": 0,
        "optimizer": {
            "type": "muon_adamw",
            "muon_lr": 0.02,
            "adamw_lr": 0.003,
            "weight_decay": 0.1,
        },
        "num_epochs": 1,
        "num_training_steps": 6,
        "warmup_steps": 2,
        "lr_scheduler_type": "wsd",
        "lr_schedule": {"decay_steps": 0.5},
        "precision": "fp32",
        "use_compile": False,
        "log_every": 1,
        "max_grad_norm": 1.0,
        "eval_max_batches": 2,
        "use_wandb": False,
    }
    cfg.update(overrides)
    return Box(cfg)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
