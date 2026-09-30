import queue
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
    prefetch_queue_size: int = 2  # background thread queue depth (0 = disabled)

    # Data parallelism + resume
    rank: int = 0
    world_size: int = 1
    skip_batches: int = 0  # batches this rank already consumed (resume)


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
        return torch.empty(0, dtype=torch.long)

    file = shard["path"]
    offset = HEADER_SIZE * 4 + start_idx * 2  # 2 bytes per uint16
    num_tokens = end_idx - start_idx

    with file.open("rb", buffering=0) as f:
        f.seek(offset)
        # Read as uint16 then convert to long for Embedding compatibility
        buf = torch.empty(num_tokens, dtype=torch.uint16)
        nbytes = f.readinto(buf.numpy())
        assert nbytes == 2 * num_tokens, "read size mismatch"
        tokens = buf.to(torch.long)
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

    Batches form one deterministic global sequence over all shards, indexed
    0, 1, 2, ... Each consumer (one DataLoader worker on one rank) takes every
    ``n_consumers``-th batch, so ranks and workers never see the same data.
    Planning a batch (which token ranges it holds) is cheap; only the batches a
    consumer keeps are read from disk. That makes resume cheap too: a consumer
    skips the batches it already delivered without reading them.
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

    def _consumer(self) -> Tuple[int, int, int]:
        """Return (consumer_id, n_consumers, batches_to_skip) for this worker.

        The DataLoader takes batches from its workers round-robin, starting at
        worker 0. So if this rank already delivered ``skip_batches`` batches,
        worker ``w`` produced ``ceil((skip_batches - w) / W)`` of them.
        """
        info = torch.utils.data.get_worker_info()
        w, W = (0, 1) if info is None else (info.id, info.num_workers)
        cfg = self.cfg
        consumer = cfg.rank * W + w
        n_consumers = cfg.world_size * W
        skip = max(0, -(-(cfg.skip_batches - w) // W))
        return consumer, n_consumers, skip

    # --- Batch plans: cheap descriptors, no token reads -----------------------
    def _plan_varlen(self) -> Iterator[Tuple[dict, list, list]]:
        """Yields (shard, starts, ends): the token ranges packed into one batch."""
        for file_path in self._files:
            shard = _load_data_shard_lazy(file_path)
            scanner = BOSScanner(shard, self.cfg.bos_token)
            bos_idx = scanner.get_indices()
            if len(bos_idx) == 0:
                continue

            idx = 0
            n = len(bos_idx)
            total = shard["num_tokens"]
            target_len = self.cfg.packed_tokens + 1  # +1 for input/target shift
            # Offset to resume a partly-consumed document from. None means the
            # next document starts at its own BOS. This carries across batches,
            # so a document longer than one batch spans several of them.
            doc_pos = None

            while idx < n:
                starts, ends = [], []
                cur_len = 0

                while cur_len < target_len and idx < n:
                    start = bos_idx[idx] if doc_pos is None else doc_pos
                    next_bos = bos_idx[idx + 1] if idx + 1 < n else total

                    # Clamp by: next doc, max_seq_len, or remaining batch space
                    end = min(
                        next_bos,
                        start + self.cfg.max_seq_len,
                        start + target_len - cur_len,
                    )
                    starts.append(int(start))
                    ends.append(int(end))
                    cur_len += end - start

                    # Only move to the next document once this one is fully
                    # consumed. Advancing unconditionally discarded whatever
                    # the two length clamps cut off, which threw away ~17% of
                    # the corpus — every document longer than max_seq_len lost
                    # its tail, and so did the last document of every batch.
                    if end >= next_bos:
                        idx += 1
                        doc_pos = None
                    else:
                        doc_pos = end

                if cur_len < 2:
                    continue  # need at least input+target
                yield shard, starts, ends

    def _plan_fixed(self) -> Iterator[Tuple[dict, int]]:
        """Yields (shard, pos): the start of one [batch_size, seq_len+1] block."""
        for file_path in self._files:
            shard = _load_data_shard_lazy(file_path)
            total = shard["num_tokens"]
            chunk_len = self.cfg.batch_size * (self.cfg.seq_len + 1)
            pos = 0
            while pos + chunk_len <= total:
                yield shard, pos
                pos += self.cfg.batch_size * self.cfg.seq_len

    # --- Materialise one planned batch ----------------------------------------
    def _read_varlen(self, shard, starts, ends):
        """Returns (inputs [1, N], targets [1, N], cu_seqlens [M+1]) on CPU."""
        segments = [
            _read_tokens_slice(shard, s, e) for s, e in zip(starts, ends)
        ]
        buf = torch.cat(segments)

        # Shift for targets
        inputs = buf[:-1]
        targets = buf[1:]

        # Build cu_seqlens (reflects buf length = packed_tokens+1).
        # Clamp to input length (inputs = buf[:-1], removing the +1
        # shift token). That clamp pulls the final boundary down by
        # one, so a last segment of length 1 collapses to length 0 and
        # leaves a duplicated boundary (~1 batch in 500).
        # flash_attn_varlen_func treats a zero-length segment as
        # undefined behaviour, so drop duplicates here — before the
        # tensor is built, to keep it pinned in one allocation.
        max_len = inputs.shape[0]
        cum_len = [0]
        for s, e in zip(starts, ends):
            boundary = min(cum_len[-1] + (e - s), max_len)
            if boundary != cum_len[-1]:
                cum_len.append(boundary)
        cu_seqlens = torch.tensor(
            cum_len,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )

        # Add batch dim for model compatibility. GPU transfer is handled by
        # the trainer so that pin_memory + non_blocking works with workers.
        return inputs.unsqueeze(0), targets.unsqueeze(0), cu_seqlens

    def _read_fixed(self, shard, pos):
        """Returns (inputs [B, S], targets [B, S]) on CPU."""
        chunk_len = self.cfg.batch_size * (self.cfg.seq_len + 1)
        buf = _read_tokens_slice(shard, pos, pos + chunk_len)
        buf = buf.view(self.cfg.batch_size, self.cfg.seq_len + 1)
        return buf[:, :-1], buf[:, 1:]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        if self.cfg.mode == "varlen_packed":
            plans, read = self._plan_varlen(), self._read_varlen
        elif self.cfg.mode == "fixed_batch":
            plans, read = self._plan_fixed(), self._read_fixed
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")

        consumer, n_consumers, skip = self._consumer()
        for global_idx, plan in enumerate(plans):
            if global_idx % n_consumers != consumer:
                continue
            if skip > 0:
                skip -= 1
                continue
            if self._should_stop():
                return
            batch = read(*plan)
            self._batch_count += 1
            self._token_count += batch[0].numel()
            yield batch


# =============================================================================
# Thread-based Prefetcher
# =============================================================================
class _ThreadPrefetcher:
    """
    Wraps any iterable and pulls batches into a fixed-size queue on a daemon
    thread so the GPU can run the current batch while the CPU prepares the next.

    Safe on Windows — uses threads, not multiprocessing, so there are no CUDA
    context or pagefile issues.
    """

    def __init__(self, loader, queue_size: int = 2):
        self._loader = loader
        self._queue_size = queue_size

    def __iter__(self):
        q = queue.Queue(maxsize=self._queue_size)
        _sentinel = object()

        def _worker():
            try:
                for batch in self._loader:
                    q.put(batch)
            finally:
                q.put(_sentinel)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is _sentinel:
                break
            yield item
        t.join()


# =============================================================================
# Factory Function
# =============================================================================
def create_nanogpt_dataloader(cfg: DataLoaderConfig) -> DataLoader:
    """Creates PyTorch DataLoader with config-driven behavior."""
    dataset = NanoGPTDataset(cfg)

    mp_context = None
    if cfg.num_workers > 0:
        if os.name == "nt":
            # Windows native: default start method is already 'spawn'
            mp_context = None
        else:
            # Linux / WSL2: explicitly use 'spawn' — fork() after CUDA init
            # causes undefined behaviour even when workers don't call CUDA directly
            mp_context = "spawn"

    use_workers = cfg.num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=None,  # dataset already yields batches
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if use_workers else None,
        # pin_memory only meaningful with workers: dataset yields CPU tensors,
        # workers pin them, main process does non_blocking GPU transfer.
        pin_memory=use_workers and cfg.device.startswith("cuda"),
        persistent_workers=use_workers,
        multiprocessing_context=mp_context,
    )

    # On Windows (num_workers=0), wrap with a thread prefetcher so the CPU
    # packs the next batch while the GPU runs the current one.
    # On Linux with num_workers>0, PyTorch's own prefetch_factor handles this.
    if not use_workers and cfg.prefetch_queue_size > 0:
        return _ThreadPrefetcher(loader, queue_size=cfg.prefetch_queue_size)
    return loader


if __name__ == "__main__":
    import time

    train_cfg = DataLoaderConfig(
        mode="varlen_packed",
        file_pattern="D:/d/DeepLearning/datasets/fineweb10b-pretokenized/fineweb_val_*.bin",
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
