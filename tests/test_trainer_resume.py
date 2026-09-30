"""A run that stops and resumes must end bit-identical to one that never stopped."""

import os

import pytest
import torch

from src.tinyllm.datasets.fineweb_helper import DataLoaderConfig, create_nanogpt_dataloader
from src.tinyllm.factory.factory import lr_scheduler_factory, model_factory, optimizer_factory
from src.tinyllm.trainer.trainer import Trainer
from src.tinyllm.utils.checkpoint import CheckpointManager, load_checkpoint
from src.tinyllm.utils.distributed import DistInfo

from conftest import tiny_model_config, tiny_train_config


def build_trainer(train_cfg, run_dir, dist=None, resume_state=None, seed=0):
    dist = dist or DistInfo()
    torch.manual_seed(seed)
    model_cfg = tiny_model_config()
    model = model_factory(model_cfg)
    optimizer = optimizer_factory(model, train_cfg)
    sched = lr_scheduler_factory(
        train_cfg.lr_scheduler_type,
        optimizer,
        num_training_steps=train_cfg.num_training_steps,
        num_warmup_steps=train_cfg.warmup_steps,
        sched_cfg=train_cfg.lr_schedule,
    )

    def loader_cfg(pattern, skip=0):
        return DataLoaderConfig(
            mode="varlen_packed",
            file_pattern=pattern,
            device="cpu",
            packed_tokens=train_cfg.packed_tokens,
            max_seq_len=train_cfg.max_seq_len,
            prefetch_queue_size=0,
            max_batches=train_cfg.get("max_batches", 0),
            rank=dist.rank,
            world_size=dist.world_size,
            skip_batches=skip,
        )

    return Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=sched,
        tokenizer=None,
        make_train_loader=lambda skip: create_nanogpt_dataloader(
            loader_cfg(train_cfg.train_file_pattern, skip)
        ),
        test_loader=create_nanogpt_dataloader(loader_cfg(train_cfg.test_file_pattern)),
        train_config=train_cfg,
        model_config=model_cfg,
        device="cpu",
        callbacks=[],
        dist_info=dist,
        ckpt_manager=CheckpointManager(str(run_dir), is_main=dist.is_main),
        resume_state=resume_state,
    )


def _params(trainer):
    return {k: v.detach().clone() for k, v in trainer.raw_model.state_dict().items()}


@pytest.mark.parametrize("accum", [1, 2])
def test_stop_and_resume_is_exact(shards, tmp_path, accum):
    extra = {"use_grad_accum": accum > 1, "iters_to_accumulate": accum}

    straight = build_trainer(tiny_train_config(shards, **extra), tmp_path / "a")
    straight.train()

    # First half: stop after 3 steps (as if the session ended), then resume
    # in a fresh process-like setup with a different init seed.
    first = build_trainer(
        tiny_train_config(shards, num_training_steps=6, **extra), tmp_path / "b"
    )
    first.max_steps = 3
    first.train()
    ckpt = CheckpointManager(str(tmp_path / "b")).latest_path()
    assert ckpt and os.path.basename(ckpt) == "step_00000003.pt"

    second = build_trainer(
        tiny_train_config(shards, **extra),
        tmp_path / "b",
        resume_state=load_checkpoint(ckpt),
        seed=123,
    )
    assert second.step == 3 and second.batches_in_epoch == 3 * accum
    second.train()

    assert second.step == straight.step == 6
    a, b = _params(straight), _params(second)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_time_limit_stops_and_saves(shards, tmp_path):
    t = build_trainer(tiny_train_config(shards), tmp_path)
    # Over the limit at the first check. (A tiny max_runtime_minutes is
    # flaky: Windows' ~15 ms clock can report 0 elapsed.)
    t._time_up = lambda: True
    t.train()
    assert t.step == 0
    assert CheckpointManager(str(tmp_path)).latest_path().endswith("step_00000000.pt")


def test_checkpoint_pruning_keeps_milestones(tmp_path):
    mgr = CheckpointManager(str(tmp_path), keep_last=2, keep_every_steps=4)
    for step in range(1, 10):
        mgr.save({"x": step}, step)
    names = sorted(os.listdir(tmp_path / "checkpoints"))
    # Milestones (4, 8) are kept; of the rest, only the newest 2 (7, 9).
    assert names == [
        "latest",
        "step_00000004.pt",
        "step_00000007.pt",
        "step_00000008.pt",
        "step_00000009.pt",
    ]
    assert mgr.latest_path().endswith("step_00000009.pt")
