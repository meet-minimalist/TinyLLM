import pytest
import torch

from src.tinyllm.datasets.fineweb_helper import DataLoaderConfig, NanoGPTDataset


def _cfg(shards, mode, **kw):
    return DataLoaderConfig(
        mode=mode,
        file_pattern=str(shards / "train_*.bin"),
        device="cpu",
        packed_tokens=128,
        max_seq_len=64,
        batch_size=2,
        seq_len=32,
        **kw,
    )


def _batches(cfg):
    return list(NanoGPTDataset(cfg))


def _same(a, b):
    return len(a) == len(b) and all(
        all(torch.equal(x, y) for x, y in zip(ba, bb)) for ba, bb in zip(a, b)
    )


@pytest.mark.parametrize("mode", ["varlen_packed", "fixed_batch"])
def test_skip_resumes_exactly(shards, mode):
    full = _batches(_cfg(shards, mode))
    assert len(full) > 10
    for k in (0, 1, 7, len(full) - 1):
        resumed = _batches(_cfg(shards, mode, skip_batches=k))
        assert _same(resumed, full[k:]), f"skip={k}"


@pytest.mark.parametrize("mode", ["varlen_packed", "fixed_batch"])
def test_ranks_partition_the_data(shards, mode):
    full = _batches(_cfg(shards, mode))
    world = 3
    per_rank = [
        _batches(_cfg(shards, mode, rank=r, world_size=world))
        for r in range(world)
    ]
    # Rank r holds global batches r, r+world, r+2*world, ...
    for r in range(world):
        assert _same(per_rank[r], full[r::world])
    assert sum(len(b) for b in per_rank) == len(full)


def test_rank_skip_combines(shards):
    full = _batches(_cfg(shards, "varlen_packed"))
    got = _batches(_cfg(shards, "varlen_packed", rank=1, world_size=2, skip_batches=3))
    assert _same(got, full[1::2][3:])


def test_varlen_segments_are_valid(shards):
    for inputs, targets, cu in _batches(_cfg(shards, "varlen_packed")):
        assert inputs.shape == targets.shape
        assert cu[0] == 0 and cu[-1] == inputs.shape[-1]
        lens = cu[1:] - cu[:-1]
        assert (lens > 0).all() and (lens <= 64).all()
