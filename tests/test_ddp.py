"""DDP on 2 CPU ranks (gloo) must match 1 process with grad accumulation 2.

Rank r takes global batches r, r+2, ...; one DDP step averages the two ranks'
gradients. A single process with accumulation 2 takes batches 2k and 2k+1 per
step and averages them the same way, so the weights must agree.
"""

import os
import socket
import sys

import pytest
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, shard_dir, out_dir):
    sys.path.insert(0, HERE)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        USE_LIBUV="0",  # Windows builds of torch ship TCPStore without libuv
    )
    from pathlib import Path

    from conftest import tiny_train_config
    from test_trainer_resume import build_trainer

    from src.tinyllm.utils.distributed import cleanup_distributed, init_distributed

    dist_info, _ = init_distributed("cpu")
    t = build_trainer(
        tiny_train_config(Path(shard_dir)), Path(out_dir) / "ddp", dist=dist_info
    )
    t.train()
    torch.save(t.raw_model.state_dict(), os.path.join(out_dir, f"rank{rank}.pt"))
    cleanup_distributed()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="no torch.distributed")
def test_ddp_matches_grad_accumulation(shards, tmp_path):
    from conftest import tiny_train_config
    from test_trainer_resume import build_trainer

    mp.spawn(
        _worker,
        args=(2, _free_port(), str(shards), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    r0 = torch.load(tmp_path / "rank0.pt")
    r1 = torch.load(tmp_path / "rank1.pt")

    single = build_trainer(
        tiny_train_config(shards, use_grad_accum=True, iters_to_accumulate=2),
        tmp_path / "single",
    )
    single.train()
    ref = single.raw_model.state_dict()

    for k in ref:
        assert torch.equal(r0[k], r1[k]), f"ranks diverged at {k}"
        torch.testing.assert_close(r0[k], ref[k], rtol=1e-4, atol=1e-5, msg=k)
