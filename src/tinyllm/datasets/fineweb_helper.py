import queue
import torch
from pathlib import Path
from typing import Iterator, Optional, Tuple
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


def _first_separator(shard: dict, bos_token: int) -> Optional[int]:
    """Position of the first document separator in a shard, or None.

    A shard can start in the middle of a document (the tail of the previous
    shard's last document); those tokens are skipped. Documents are short, so
    this reads only a few KB: the window doubles until a separator is found.
    """
    total = shard["num_tokens"]
    start, window = 0, 4096
    while start < total:
        end = min(total, start + window)
        hits = (_read_tokens_slice(shard, start, end) == bos_token).nonzero()
        if len(hits):
            return start + int(hits[0])
        start, window = end, window * 2
    return None


# =============================================================================
# Unified Iterable Dataset
# =============================================================================
class NanoGPTDataset(IterableDataset):
    """
    Memory-efficient dataloader supporting both varlen-packed and fixed-batch modes.
    Switch modes by changing DataLoaderConfig.mode

    Batches form one fixed global sequence over all shards, numbered 0, 1, 2,
    ... Every batch is a fixed-size run of consecutive tokens, so where batch
    ``g`` lives (file and offset) is arithmetic on the shard sizes — no shard
    needs to be scanned. In varlen mode the document boundaries a batch needs
    (for ``cu_seqlens``) are found in that batch's own tokens after reading it.

    Each consumer (one DataLoader worker on one rank) takes every
    ``n_consumers``-th batch, so ranks and workers never see the same data, and
    resume jumps straight to the next unread batch.
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

    # --- Batch layout: arithmetic only ----------------------------------------
    def _layout(self) -> Iterator[Tuple[dict, int, int, int, int]]:
        """Yields (shard, first, stride, span, n_batches) per file.

        Batch ``i`` of a file covers tokens ``[first + i*stride, +span)``,
        clipped to the file. Computed lazily, file by file.
        """
        cfg = self.cfg
        for file_path in self._files:
            shard = _load_data_shard_lazy(file_path)
            total = shard["num_tokens"]
            if cfg.mode == "varlen_packed":
                first = _first_separator(shard, cfg.bos_token)
                if first is None:
                    continue
                # packed_tokens inputs + 1 for the target shift. Batches do
                # not overlap: the shift token is a target only.
                span = stride = cfg.packed_tokens + 1
                remaining = total - first
                n = -(-remaining // stride)
                if remaining % stride == 1:
                    n -= 1  # a 1-token tail has no input/target pair
            else:
                first, stride = 0, cfg.batch_size * cfg.seq_len
                span = cfg.batch_size * (cfg.seq_len + 1)
                n = (total - span) // stride + 1 if total >= span else 0
            if n > 0:
                yield shard, first, stride, span, n

    # --- Materialise one batch ------------------------------------------------
    def _read_varlen(self, shard, start, end):
        """Returns (inputs [1, N], targets [1, N], cu_seqlens [M+1]) on CPU."""
        buf = _read_tokens_slice(shard, start, end)
        inputs, targets = buf[:-1], buf[1:]
        n_in = inputs.shape[0]

        # Segments: a new one starts at every separator (document start) and
        # every max_seq_len tokens inside a document piece. The first piece may
        # continue a document from the previous batch.
        doc_starts = (buf == self.cfg.bos_token).nonzero().flatten().tolist()
        piece_starts = [0] + [i for i in doc_starts if i > 0]
        bounds = []
        for s, e in zip(piece_starts, piece_starts[1:] + [buf.shape[0]]):
            bounds.extend(range(s, e, self.cfg.max_seq_len))
        bounds.append(buf.shape[0])

        # Clamp to the input length (inputs drop the +1 shift token). That can
        # collapse a 1-token last segment to length 0 — a duplicated boundary,
        # which flash_attn_varlen_func treats as undefined behaviour — so drop
        # duplicates before building the (pinned) tensor.
        cum_len = [0]
        for b in bounds[1:]:
            b = min(b, n_in)
            if b != cum_len[-1]:
                cum_len.append(b)
        cu_seqlens = torch.tensor(
            cum_len,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )
        # Add batch dim for model compatibility. GPU transfer is handled by
        # the trainer so that pin_memory + non_blocking works with workers.
        return inputs.unsqueeze(0), targets.unsqueeze(0), cu_seqlens

    def _read_fixed(self, shard, start, end):
        """Returns (inputs [B, S], targets [B, S]) on CPU."""
        buf = _read_tokens_slice(shard, start, end)
        buf = buf.view(self.cfg.batch_size, self.cfg.seq_len + 1)
        return buf[:, :-1], buf[:, 1:]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        if self.cfg.mode not in ("varlen_packed", "fixed_batch"):
            raise ValueError(f"Unknown mode: {self.cfg.mode}")
        read = (
            self._read_varlen
            if self.cfg.mode == "varlen_packed"
            else self._read_fixed
        )

        consumer, n_consumers, skip = self._consumer()
        g = consumer + skip * n_consumers  # next global batch for this reader
        base = 0  # global index of the current file's first batch
        for shard, first, stride, span, n in self._layout():
            while g < base + n:
                if self._should_stop():
                    return
                start = first + (g - base) * stride
                end = min(start + span, shard["num_tokens"])
                batch = read(shard, start, end)
                self._batch_count += 1
                self._token_count += batch[0].numel()
                yield batch
                g += n_consumers
            base += n


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
