"""
Dataset loading with optional pre-tokenization for maximum GPU throughput.

Key optimization: pre-tokenize the dataset once at startup so the DataLoader
serves pure tensors with zero tokenizer overhead during training.
"""

import os
import random
import pickle
from typing import Generator, List, Tuple

import datasets
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizer

cache_dir = "./data_cache"
hf_dataset_cache_dir = "./data_cache/hf_cache/"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(hf_dataset_cache_dir, exist_ok=True)


class _TokenizedDataset(Dataset):
    """Dataset of pre-tokenized (input_ids, mask, labels) tensors."""

    def __init__(self, hf_dataset, tokenizer, max_len, cache_path=None):
        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.input_ids = data["input_ids"]
            self.attention_mask = data["attention_mask"]
            self.labels = data["labels"]
            return

        # Tokenize everything upfront
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        # Batch tokenization is much faster than per-sample encode()
        texts = [item["text"] for item in hf_dataset]
        batched = tokenizer(
            texts,
            max_length=max_len - 2,
            truncation=True,
            padding=False,
            return_attention_mask=False,
        )

        for ids in batched["input_ids"]:
            ids = [bos_id] + ids + [eos_id]
            mask = [1] * len(ids)
            lbl = ids[1:] + [pad_id]
            self.input_ids.append(torch.tensor(ids, dtype=torch.int32))
            self.attention_mask.append(torch.tensor(mask, dtype=torch.int32))
            self.labels.append(torch.tensor(lbl, dtype=torch.int64))

        # Save to cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "input_ids": self.input_ids,
                        "attention_mask": self.attention_mask,
                        "labels": self.labels,
                    },
                    f,
                )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


class BatchSamplerSimilarLength(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_tokens = self.batch_size * self.seq_len
        self.shuffle = shuffle

        # Use pre-computed lengths if available
        if hasattr(dataset, "input_ids"):
            self.indices = [
                (i, len(ids)) for i, ids in enumerate(dataset.input_ids)
            ]
        else:
            self.indices = [
                (i, len(data["text"].split())) for i, data in enumerate(dataset)
            ]

        if self.shuffle:
            random.shuffle(self.indices)

        sorted_indices = sorted(self.indices, key=lambda x: x[1])

        self.all_batch_idx = []
        single_batch_idx = []
        cumulative = 0

        for idx, token_len in sorted_indices:
            cumulative += token_len
            single_batch_idx.append(idx)

            if cumulative > self.total_tokens:
                self.all_batch_idx.append(single_batch_idx.copy())
                single_batch_idx.clear()
                cumulative = 0

        if single_batch_idx:
            self.all_batch_idx.append(single_batch_idx)

        if self.shuffle:
            random.shuffle(self.all_batch_idx)

    def __iter__(self) -> Generator[List[int], None, None]:
        for batch_idx in self.all_batch_idx:
            random.shuffle(batch_idx)
            yield batch_idx

    def __len__(self) -> int:
        return len(self.all_batch_idx)


def _prefetched_collate(pad_token_id: int):
    """Fast collate for pre-tokenized data."""

    def collate(batch):
        input_ids, attn, labels = zip(*batch)
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        attn = pad_sequence(attn, batch_first=True, padding_value=0)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=pad_token_id
        )
        return input_ids, attn, labels

    return collate


def _on_the_fly_collate(tokenizer, max_len):
    """Slower collate that tokenizes each batch on the fly."""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        texts = [item["text"] for item in batch]

        # Batch tokenize (faster than per-sample encode)
        encoded = tokenizer(
            texts,
            max_length=max_len - 2,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        input_ids = []
        attention_mask = []
        labels = []

        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            ids = [bos_id] + ids + [eos_id]
            mask = [1] + mask + [1]
            lbl = ids[1:] + [pad_id]
            input_ids.append(torch.tensor(ids, dtype=torch.int32))
            attention_mask.append(torch.tensor(mask, dtype=torch.int32))
            labels.append(torch.tensor(lbl, dtype=torch.int64))

        return (
            pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
            pad_sequence(attention_mask, batch_first=True, padding_value=0),
            pad_sequence(labels, batch_first=True, padding_value=pad_id),
        )

    return collate


class DatasetHelper:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        seq_len: int,
        num_workers: int = 4,
        persistent_workers: bool = True,
        use_pin_memory: bool = True,
        sample_similar_len: bool = True,
        split: str = "train",
        dataset_name: str = None,
        dataset_object=None,
        tokenized_dataset=None,
        prefetch_factor: int = 4,
        tokenize_upfront: bool = True,
        cache_dir: str = None,
    ):
        """
        Construct a dataset loader.

        Args:
            tokenizer: Tokenizer instance.
            batch_size: Batch size.
            seq_len: Maximum sequence length.
            num_workers: DataLoader workers.
            persistent_workers: Keep workers alive between epochs.
            use_pin_memory: Pin memory for faster CPU→GPU transfer.
            sample_similar_len: Bucket similar-length sequences together.
            split: Dataset split.
            dataset_name: HF dataset name.
            dataset_object: Pre-loaded HF dataset.
            tokenized_dataset: Already-tokenized dataset.
            prefetch_factor: Batches to prefetch per worker.
                Higher values (4-8) keep the GPU fed. Default: 4.
            tokenize_upfront: If True, tokenize the entire dataset at startup
                and cache the results. This eliminates per-batch tokenizer
                overhead — the #1 cause of low GPU utilization.
            cache_dir: Directory to cache tokenized data. If None, no caching.
        """
        if split not in ["train", "test", "validation"]:
            raise RuntimeError("Split must be 'train', 'validation' or 'test'.")

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_len = seq_len

        # ── Determine data source ─────────────────────────────────────
        if tokenized_dataset is not None:
            dataset = tokenized_dataset
            collate_fn = _prefetched_collate(self.pad_token_id)

        elif tokenize_upfront:
            # Load raw dataset
            if dataset_name is not None:
                raw_dataset = load_dataset(
                    dataset_name, split=split, cache_dir=hf_dataset_cache_dir
                )
            elif dataset_object is not None:
                raw_dataset = dataset_object
            else:
                raise RuntimeError(
                    "Provide dataset_name, dataset_object, or tokenized_dataset."
                )

            # Tokenize everything upfront
            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                safe_name = (
                    dataset_name.replace("/", "_")
                    if dataset_name
                    else "dataset"
                )
                cache_path = os.path.join(
                    cache_dir, f"{safe_name}_{split}_{seq_len}.pkl"
                )

            dataset = _TokenizedDataset(
                raw_dataset, tokenizer, seq_len, cache_path=cache_path
            )
            collate_fn = _prefetched_collate(self.pad_token_id)

        else:
            # On-the-fly tokenization (slow — only for debugging)
            if dataset_name is not None:
                dataset = load_dataset(
                    dataset_name, split=split, cache_dir=hf_dataset_cache_dir
                )
            elif dataset_object is not None:
                dataset = dataset_object
            else:
                raise RuntimeError(
                    "Provide dataset_name, dataset_object, or tokenized_dataset."
                )
            collate_fn = _on_the_fly_collate(tokenizer, seq_len)

        # ── Build DataLoader ──────────────────────────────────────────
        loader_kwargs = {
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers and num_workers > 0,
            "pin_memory": use_pin_memory,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        }

        if sample_similar_len:
            batch_sampler = BatchSamplerSimilarLength(
                dataset, batch_size, seq_len, shuffle=(split == "train")
            )
            loader_kwargs["batch_sampler"] = batch_sampler
        else:
            loader_kwargs["batch_size"] = batch_size
            loader_kwargs["shuffle"] = split == "train"

        self.dataloader = DataLoader(dataset, **loader_kwargs)

    def get_loader(self):
        return self.dataloader


if __name__ == "__main__":
    from src.tinyllm.utils.misc import get_tokenizer

    tok = get_tokenizer("openai-community/gpt2")

    # Pre-tokenized (fast — recommended for training)
    loader = DatasetHelper(
        tokenizer=tok,
        dataset_name="roneneldan/TinyStories",
        batch_size=32,
        seq_len=256,
        num_workers=8,
        tokenize_upfront=False,
        prefetch_factor=4,
        split="train",
        cache_dir=cache_dir,
    ).get_loader()

    import time

    t0 = time.time()
    for i, (ids, mask, labels) in enumerate(loader):
        print(
            f"Batch {i}: ids shape {ids.shape}, mask shape {mask.shape}, labels shape {labels.shape}"
        )
        if i == 10:
            break
    elapsed = time.time() - t0
    print(f"10 batches in {elapsed:.2f}s ({elapsed/10:.3f}s/batch)")
    print(f"  ids shape: {ids.shape}")
