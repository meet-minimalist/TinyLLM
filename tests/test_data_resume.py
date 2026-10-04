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


def _reference_varlen(paths, packed_tokens, max_seq_len, bos):
    """The original scan-then-pack algorithm, kept as an oracle: find every
    separator in the shard, then walk documents in order, cutting segments at
    the next document, at max_seq_len and at the batch end."""
    import numpy as np

    from src.tinyllm.datasets.fineweb_helper import (
        _load_data_shard_lazy,
        _read_tokens_slice,
    )

    out = []
    for path in paths:
        shard = _load_data_shard_lazy(path)
        total = shard["num_tokens"]
        tokens = _read_tokens_slice(shard, 0, total)
        bos_idx = np.flatnonzero(tokens.numpy() == bos)
        idx, n, doc_pos, target = 0, len(bos_idx), None, packed_tokens + 1
        while idx < n:
            starts, ends, cur = [], [], 0
            while cur < target and idx < n:
                start = bos_idx[idx] if doc_pos is None else doc_pos
                next_bos = bos_idx[idx + 1] if idx + 1 < n else total
                end = min(next_bos, start + max_seq_len, start + target - cur)
                starts.append(start)
                ends.append(end)
                cur += end - start
                idx, doc_pos = (idx + 1, None) if end >= next_bos else (idx, end)
            if cur < 2:
                continue
            buf = torch.cat([tokens[s:e] for s, e in zip(starts, ends)])
            cu = [0]
            for s, e in zip(starts, ends):
                b = min(cu[-1] + (e - s), buf.numel() - 1)
                if b != cu[-1]:
                    cu.append(b)
            out.append((buf[:-1][None], buf[1:][None], torch.tensor(cu, dtype=torch.int32)))
    return out


@pytest.mark.parametrize("packed,max_len", [(128, 64), (300, 50), (97, 1000)])
def test_scan_free_packing_matches_reference(tmp_path, packed, max_len):
    from conftest import BOS, make_tokens, write_shard

    paths = []
    for i in range(4):
        # Shards 1-3 start mid-document, like real FineWeb shards.
        p = tmp_path / f"s_{i}.bin"
        write_shard(p, make_tokens(i, 200)[5 * i :])
        paths.append(p)
    ref = _reference_varlen(paths, packed, max_len, BOS)
    got = list(
        NanoGPTDataset(
            DataLoaderConfig(
                file_pattern=str(tmp_path / "s_*.bin"),
                device="cpu",
                packed_tokens=packed,
                max_seq_len=max_len,
            )
        )
    )
    assert _same(got, ref)
