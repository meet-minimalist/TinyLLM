"""
# @ Author: Meet Patel
# @ Create Time: 2024-06-30 13:55:42
# @ Modified by: Meet Patel
# @ Modified time: 2024-07-20 16:53:09
# @ Description:
"""

import random
from typing import Generator, List, Tuple

import datasets
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizer
from datasets import DatasetBuilder


class BatchSamplerSimilarLength(Sampler):
    def __init__(
        self,
        dataset_iterator: datasets.Dataset,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
    ):
        """Initializer to load the dataset and sort as per the source sequences
        (german sequence) and prepare the indices in a bucketed manner where the
        sequences with similar lengths are grouped together.

        Args:
            dataset_iterator (datasets.Dataset): Dataset iterator.
            batch_size (int): Batch size to be used to compute upper limit of
                tokens in a given batch. This will be directly correlated to
                available GPU memory.
            seq_len (int): Sequence length to be used to compute upper limit of
                tokens.
            shuffle (bool, optional): Shuffle the dataset before sorting and
                after getting the buckets. Defaults to True.
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_tokens_in_batch = self.batch_size * self.seq_len
        self.shuffle = shuffle

        self.indices = [
            (i, len(data["text"].split(" ")))
            for i, data in enumerate(dataset_iterator)
        ]

        if self.shuffle:
            random.shuffle(self.indices)

        sorted_indices = sorted(self.indices, key=lambda x: x[1])

        self.all_batch_idx = []
        single_batch_idx = []
        cummulative_token_len = 0

        for idx, token_len in sorted_indices:
            cummulative_token_len += token_len

            single_batch_idx.append(idx)

            if cummulative_token_len > self.total_tokens_in_batch:
                self.all_batch_idx.append(single_batch_idx.copy())
                single_batch_idx.clear()
                cummulative_token_len = 0

        if self.shuffle:
            random.shuffle(self.all_batch_idx)

    def __iter__(self) -> Generator[int, int, int]:
        """
        Function will fetch list of indices to be used to generate a batch.

        Yields:
            List[int]: Yields list of indices for batch generation.
        """
        for batch_idx in self.all_batch_idx:
            random.shuffle(batch_idx)
            yield batch_idx

    def __len__(self) -> int:
        """Function to get the total number of batches which can be generated.

        Returns:
            int: Number of batches from the given dataset.
        """
        return len(self.all_batch_idx)


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
        dataset_object: DatasetBuilder = None,
        tokenized_dataset: object = None,
        prefetch_tokenized: bool = False,
        prefetch_factor: int = 2,
    ):
        """
        Construct a dataset loader within this class.

        Supports three modes:
         - load by `dataset_name` (HF dataset) and tokenize on the fly
         - use a provided `dataset_object` (already loaded HF dataset)
         - use a `tokenized_dataset` (dicts with `input_ids`, `attention_mask`, `labels`)

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer instance.
            batch_size  (int): Batch size to be used to compute upper limit of
                tokens.
            seq_len (int): Sequence length to be used to compute upper limit of
                tokens in a given batch.
            num_workers (int): Number of workers for dataset loader.
            persistent_workers (bool): Use persistent_worker in torch dataloader.
            use_pin_memory (bool): Use pin_memory in torch dataloader.
            sample_similar_len (bool): Enabling sampling of similar length
                sequences in same batch.
            split (str, optional): Dataset split. Either "train" or "test".
                Defaults to "train".
            dataset_name (str, optional): Name of HF dataset to load.
            dataset_object (DatasetBuilder, optional): Preloaded dataset object.
            tokenized_dataset (object, optional): Prefetched tokenized dataset.
            prefetch_tokenized (bool, optional): If True, expects tokenized_dataset to
                contain tensors or lists for fast collating.
            prefetch_factor (int, optional): Number of batches to prefetch. Defaults to 2.

        Raises:
            RuntimeError: If unsupported split is provided or no dataset source is provided.
        """
        if split not in ["train", "test", "validation"]:
            raise RuntimeError(
                f"Split for dataloader shall be from 'train', 'validation' or 'test' only."
            )

        # Decide where data comes from
        if tokenized_dataset is None:
            if dataset_name is None and dataset_object is None:
                raise RuntimeError(
                    f"Either dataset_name or dataset_object must be provided when tokenized_dataset is None."
                )
            if dataset_name is not None:
                dataset = load_dataset(dataset_name, split=split)
            else:
                dataset = dataset_object
            self._use_prefetched = False
            collate_fn = self.collate_batch
        else:
            dataset = tokenized_dataset
            self._use_prefetched = bool(prefetch_tokenized)

            def _prefetched_collate(batch):
                # Expect batch to be dicts with input_ids (list/int or tensor), attention_mask, labels
                input_ids = [
                    (
                        torch.tensor(b["input_ids"], dtype=torch.int32)
                        if not isinstance(b["input_ids"], torch.Tensor)
                        else b["input_ids"].to(torch.int32)
                    )
                    for b in batch
                ]
                attn = [
                    (
                        torch.tensor(
                            b.get("attention_mask", [1] * len(b["input_ids"])),
                            dtype=torch.int32,
                        )
                        if not isinstance(
                            b.get("attention_mask", None), torch.Tensor
                        )
                        else b["attention_mask"].to(torch.int32)
                    )
                    for b in batch
                ]
                labels = [
                    (
                        torch.tensor(
                            b.get("labels", b["input_ids"]), dtype=torch.int64
                        )
                        if not isinstance(b.get("labels", None), torch.Tensor)
                        else b["labels"].to(torch.int64)
                    )
                    for b in batch
                ]
                input_ids = pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.pad_token_id.item(),
                ).to(torch.int32)
                attn = pad_sequence(attn, batch_first=True, padding_value=0).to(
                    torch.int32
                )
                labels = pad_sequence(
                    labels,
                    batch_first=True,
                    padding_value=self.pad_token_id.item(),
                ).to(torch.int64)
                return input_ids, attn, labels

            collate_fn = _prefetched_collate

        self.tokenizer = tokenizer
        self.pad_token_id = torch.tensor([tokenizer.pad_token_id])
        self.bos_token_id = torch.tensor([tokenizer.bos_token_id])
        self.eos_token_id = torch.tensor([tokenizer.eos_token_id])

        self.max_len = seq_len
        if sample_similar_len:
            batch_sampler = BatchSamplerSimilarLength(
                dataset, batch_size, seq_len, shuffle=True
            )
            self.dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                pin_memory=use_pin_memory,
                prefetch_factor=prefetch_factor,
            )
        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                pin_memory=use_pin_memory,
                prefetch_factor=prefetch_factor,
            )

    def collate_batch(self, batch_data: List) -> Tuple[torch.Tensor]:
        """Function to tokenize sequences and prepare the mask for training.

        Args:
            batch_data (List): List of Tuple of source and target sequences.

        Returns:
            Tuple[torch.Tensor]: Tuple of source token, target token, source
                mask, target mask and target labels. All have shape of [batch, seq]
        """
        batched_input_ids, batched_attention_mask, batched_labels = [], [], []

        for data in batch_data:
            text = data["text"]
            input_ids = self.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self.max_len
                - 2,  # minus 2 because we will be adding bos and eos tokens as well.
                truncation=True,
            )[0]
            input_ids = torch.cat(
                [self.bos_token_id, input_ids, self.eos_token_id], dim=0
            )
            labels = torch.cat([input_ids[1:], self.pad_token_id], dim=0)
            attn_mask = torch.ones_like(input_ids)
            batched_input_ids.append(input_ids)
            batched_labels.append(labels)
            batched_attention_mask.append(attn_mask)
        batched_input_ids = pad_sequence(
            batched_input_ids,
            batch_first=True,
            padding_value=self.pad_token_id.item(),
        ).to(torch.int32)
        batched_attention_mask = pad_sequence(
            batched_attention_mask, batch_first=True, padding_value=0
        ).to(torch.int32)
        batched_labels = pad_sequence(
            batched_labels,
            batch_first=True,
            padding_value=self.pad_token_id.item(),
        ).to(torch.int64)
        return batched_input_ids, batched_attention_mask, batched_labels

    def get_loader(self):
        """
        Get instance of dataloader.

        Returns:
            DataLoader: Dataloader based on split.
        """
        return self.dataloader


if __name__ == "__main__":
    from src.tinyllm.utils.misc import get_tokenizer

    model_type = "gpt2"
    batch_size = 2
    seq_len = 128
    num_workers = 4
    persistent_workers = True
    use_pin_memory = True
    sample_similar_len = True
    split = "validation"
    prefetch_factor = 2

    tokenizer = get_tokenizer(model_type)

    test_loader = DatasetHelper(
        tokenizer=tokenizer,
        dataset_name="roneneldan/TinyStories",
        batch_size=batch_size,
        seq_len=seq_len,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        use_pin_memory=use_pin_memory,
        sample_similar_len=sample_similar_len,
        split=split,
        prefetch_factor=prefetch_factor,
    ).get_loader()

    for batch_idx, (input_ids, attn_mask, labels) in enumerate(test_loader):
        print(f"Input ids: {input_ids}")
        print(f"Attention mask: {attn_mask}")
        print(f"Labels: {labels}")
        print(f"Input ids shape: {input_ids.shape}")
        print(f"Attention mask shape: {attn_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        break
