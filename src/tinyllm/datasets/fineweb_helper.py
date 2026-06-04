import torch
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, Union
import glob
import os
import threading
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass

# =============================================================================
# Constants
# =============================================================================
BOS_ID = 50256
HEADER_SIZE = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1


# =============================================================================
# Configuration
# =============================================================================
@dataclass(slots=True)
class DataLoaderConfig:
    mode: str = "varlen_packed"  # "varlen_packed" | "fixed_batch"
    file_pattern: str = "data/fineweb_*.bin"
    device: str = "cuda"

    # Varlen Packed mode
    packed_tokens: int = 8192
    max_seq_len: int = 2048
    align_to_bos: bool = True

    # Fixed Batch mode
    batch_size: int = 4
    seq_len: int = 512

    # Limits (useful for testing — 0 = no limit)
    max_batches: int = 0  # Stop after this many batches (0 = unlimited)
    max_tokens: int = 0  # Stop after this many total tokens (0 = unlimited)

    # Shared
    bos_token: int = BOS_ID
    num_workers: int = 0
    prefetch_factor: int = 2


# =============================================================================
# Low-level I/O Utilities
# =============================================================================
def _load_data_shard_lazy(file: Path):
    """Memory-maps header only. Does NOT load tokens into RAM."""
    header = torch.from_file(
        str(file), shared=False, size=HEADER_SIZE, dtype=torch.int32
    )
    assert header[0] == HEADER_MAGIC, f"magic mismatch: {header[0]}"
    assert header[1] == HEADER_VERSION, f"version unsupported: {header[1]}"

    num_tokens = int(header[2])
    return {
        "path": file,
        "num_tokens": num_tokens,
        "_header": header,  # keep alive
    }


def _read_tokens_slice(
    shard: dict, start_idx: int, end_idx: int
) -> torch.Tensor:
    """Zero-copy disk read for a specific token range."""
    if start_idx >= end_idx:
        return torch.empty(0, dtype=torch.uint16)

    file = shard["path"]
    offset = HEADER_SIZE * 4 + start_idx * 2  # 2 bytes per uint16
    num_tokens = end_idx - start_idx

    with file.open("rb", buffering=0) as f:
        f.seek(offset)
        # Allocate on CPU, pin if targeting GPU later
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "read size mismatch"
    return tokens


# =============================================================================
# Async BOS Scanner (Varlen mode only)
# =============================================================================
class BOSScanner:
    def __init__(
        self, shard: dict, bos_token: int, chunk_size: int = 6_000_000
    ):
        self.shard = shard
        self.bos_token = bos_token
        self.chunk_size = chunk_size
        self._bos_idx: Optional[np.ndarray] = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._scan, daemon=True)
        self._thread.start()

    def _scan(self):
        file = self.shard["path"]
        num_tokens = self.shard["num_tokens"]
        positions = []

        with file.open("rb", buffering=0) as f:
            f.seek(HEADER_SIZE * 4)
            for start in range(0, num_tokens, self.chunk_size):
                length = min(self.chunk_size, num_tokens - start)
                chunk = torch.empty(length, dtype=torch.uint16)
                f.readinto(chunk.numpy())
                bos = (
                    (chunk == self.bos_token).nonzero(as_tuple=True)[0].numpy()
                )
                positions.append(bos + start)

        self._bos_idx = (
            np.concatenate(positions)
            if positions
            else np.array([], dtype=np.int64)
        )
        self._done.set()

    def wait_ready(self):
        self._done.wait()

    def get_indices(self) -> np.ndarray:
        self.wait_ready()
        return self._bos_idx


# =============================================================================
# Unified Iterable Dataset
# =============================================================================
class NanoGPTDataset(IterableDataset):
    """
    Memory-efficient dataloader supporting both varlen-packed and fixed-batch modes.
    Switch modes by changing DataLoaderConfig.mode
    """

    def __init__(self, cfg: DataLoaderConfig):
        self.cfg = cfg
        self._batch_count = 0
        self._token_count = 0
        if cfg.file_pattern is None:
            raise ValueError(
                "file_pattern must be specified in DataLoaderConfig"
            )
        self._files = [Path(f) for f in sorted(glob.glob(cfg.file_pattern))]
        if not self._files:
            raise FileNotFoundError(
                f"No files match pattern: {cfg.file_pattern}"
            )

    def _should_stop(self) -> bool:
        cfg = self.cfg
        if cfg.max_batches > 0 and self._batch_count >= cfg.max_batches:
            return True
        if cfg.max_tokens > 0 and self._token_count >= cfg.max_tokens:
            return True
        return False

    def _iter_varlen(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yields: (inputs [N], targets [N], cu_seqlens [M+1])"""
        for file_path in self._files:
            if self._should_stop():
                return
            shard = _load_data_shard_lazy(file_path)
            scanner = BOSScanner(shard, self.cfg.bos_token)
            bos_idx = scanner.get_indices()
            if len(bos_idx) == 0:
                continue

            idx = 0
            n = len(bos_idx)
            total = shard["num_tokens"]
            target_len = self.cfg.packed_tokens + 1  # +1 for input/target shift

            while idx < n:
                starts, ends = [], []
                cur_len = 0

                while cur_len < target_len:
                    start = bos_idx[idx]
                    next_bos = bos_idx[idx + 1] if idx + 1 < n else total

                    # Clamp by: next doc, max_seq_len, or remaining batch space
                    end = min(
                        next_bos,
                        start + self.cfg.max_seq_len,
                        start + target_len - cur_len,
                    )
                    starts.append(start)
                    ends.append(end)
                    cur_len += end - start
                    idx += 1
                    if idx >= n and cur_len < target_len:
                        break  # shard exhausted

                if cur_len < 2:
                    continue  # need at least input+target

                # Read & concatenate segments
                segments = [
                    _read_tokens_slice(shard, s, e)
                    for s, e in zip(starts, ends)
                ]
                buf = torch.cat(segments)

                # Shift for targets
                inputs = buf[:-1]
                targets = buf[1:]

                # Build cu_seqlens
                cum_len = [0]
                for s, e in zip(starts, ends):
                    cum_len.append(cum_len[-1] + (e - s))
                cu_seqlens = torch.tensor(
                    cum_len, dtype=torch.int32, pin_memory=True
                )

                # Move to device
                dev = self.cfg.device
                self._batch_count += 1
                self._token_count += inputs.shape[0]
                yield (
                    inputs.to(dev, non_blocking=True),
                    targets.to(dev, non_blocking=True),
                    cu_seqlens.to(dev, non_blocking=True),
                )
                if self._should_stop():
                    return

    def _iter_fixed(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields: (inputs [B, S], targets [B, S])"""
        for file_path in self._files:
            if self._should_stop():
                return
            shard = _load_data_shard_lazy(file_path)
            total = shard["num_tokens"]
            chunk_len = self.cfg.batch_size * (self.cfg.seq_len + 1)
            pos = 0

            while pos + chunk_len <= total:
                if self._should_stop():
                    return
                buf = _read_tokens_slice(shard, pos, pos + chunk_len)
                buf = buf.view(self.cfg.batch_size, self.cfg.seq_len + 1)

                inputs = buf[:, :-1]
                targets = buf[:, 1:]
                pos += self.cfg.batch_size * self.cfg.seq_len

                dev = self.cfg.device
                self._batch_count += 1
                self._token_count += inputs.numel()
                yield (
                    inputs.to(dev, non_blocking=True),
                    targets.to(dev, non_blocking=True),
                )

    def __iter__(self) -> Iterator[Union[Tuple, Tuple]]:
        if self.cfg.mode == "varlen_packed":
            yield from self._iter_varlen()
        elif self.cfg.mode == "fixed_batch":
            yield from self._iter_fixed()
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")


# =============================================================================
# Factory Function
# =============================================================================
def create_nanogpt_dataloader(cfg: DataLoaderConfig) -> DataLoader:
    """Creates PyTorch DataLoader with config-driven behavior."""
    dataset = NanoGPTDataset(cfg)

    # WSL/Linux: use 'spawn' context for CUDA compatibility
    mp_context = None
    if cfg.num_workers > 0:
        if os.name == "nt" or "WSL" in os.uname().release:
            # Windows/WSL: default context is usually fine
            mp_context = None
        else:
            # Linux: use 'spawn' to avoid fork() CUDA issues
            mp_context = "spawn"

    # CRITICAL: Disable pin_memory when using workers
    # The main process will handle CPU→GPU transfer safely
    use_pin_memory = (cfg.device == "cuda") and (cfg.num_workers == 0)

    return DataLoader(
        dataset,
        batch_size=None,  # dataset already yields batches
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        pin_memory=use_pin_memory,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=mp_context,
    )


if __name__ == "__main__":
    import time

    train_cfg = DataLoaderConfig(
        mode="varlen_packed",
        file_pattern="/mnt/d/d/DeepLearning/datasets/fineweb10b-pretokenized/fineweb_train_*.bin",
        device="cuda",
        packed_tokens=8192,
        max_seq_len=2048,
        align_to_bos=True,
        num_workers=2,
        prefetch_factor=4,
    )

    train_loader = create_nanogpt_dataloader(train_cfg)

    t0 = time.time()
    for i, (input, targets, cu_seqlens) in enumerate(train_loader):
        print(
            f"Batch {i}: input shape {input.shape}, targets shape {targets.shape}, cu_seqlens shape {cu_seqlens.shape}"
        )
        if i == 10:
            break
    elapsed = time.time() - t0
    print(f"10 batches in {elapsed:.2f}s ({elapsed/10:.3f}s/batch)")
