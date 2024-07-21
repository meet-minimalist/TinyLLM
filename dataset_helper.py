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
                tokens.
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
        tokenizer,
        batch_size,
        seq_len,
        num_workers,
        persistent_workers,
        split="train",
    ):
        if split not in ["train", "validation"]:
            raise RuntimeError(
                f"Split for dataloader shall be from 'train' or 'validation' only."
            )
        data_loader = load_dataset("roneneldan/TinyStories", split=split)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        batch_sampler = BatchSamplerSimilarLength(
            data_loader, batch_size, seq_len, shuffle=True
        )
        self.dataloader = DataLoader(
            data_loader,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_batch,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def collate_batch(self, batch_data: List) -> Tuple[torch.Tensor]:
        """Function to tokenize sequences and prepare the mask for training.

        Args:
            batch_data (List): List of Tuple of source and target sequences.

        Returns:
            Tuple[torch.Tensor]: Tuple of source token, target token, source
                mask, target mask and target labels.
        """
        input_ids, attention_mask, labels = [], [], []

        for data in batch_data:
            text = data["text"]
            tokenized_data = self.tokenizer(text)
            input_ids.append(
                torch.tensor(tokenized_data["input_ids"], dtype=torch.int32)
            )
            labels.append(
                torch.tensor(
                    tokenized_data["input_ids"][1:] + [self.pad_token_id],
                    dtype=torch.int32,
                )
            )
            attention_mask.append(
                torch.tensor(
                    tokenized_data["attention_mask"], dtype=torch.int32
                )
            )
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.pad_token_id
        )
        labels = labels.to(torch.long)
        return input_ids, attention_mask, labels

    def get_loader(self):
        return self.dataloader
