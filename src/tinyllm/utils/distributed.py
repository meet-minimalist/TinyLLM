"""
Single-node data parallelism helpers (DDP via ``torchrun``).

The same entry point runs on one GPU (``python -m src.tinyllm.train ...``) and
on N GPUs (``torchrun --nproc_per_node=N -m src.tinyllm.train ...``).
``torchrun`` sets RANK / LOCAL_RANK / WORLD_SIZE; without them this is a no-op.
"""

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def init_distributed(
    device: str, timeout_minutes: int = 60
) -> tuple[DistInfo, str]:
    """Initialise the process group if launched by torchrun.

    Returns the DistInfo and the device string this rank must use. The
    timeout is long because rank 0 runs benchmarks and saves checkpoints while
    the other ranks wait at the next collective.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo(), device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )
    return DistInfo(rank, local_rank, world_size), device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce(tensor: torch.Tensor, op=None) -> torch.Tensor:
    """In-place all-reduce (sum by default); a no-op when not distributed."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM)
    return tensor


def any_rank(flag: bool, device) -> bool:
    """True if ``flag`` is set on any rank. Every rank must call this."""
    if not (dist.is_available() and dist.is_initialized()):
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
