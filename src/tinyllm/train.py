"""
Training entry point.

Single GPU:
    python -m src.tinyllm.train -c <train.yaml> -m <model.yaml> [--run_name NAME]

N GPUs on one node:
    torchrun --nproc_per_node=N -m src.tinyllm.train -c <train.yaml> -m <model.yaml>

Resume: give the run a stable name (``run_name`` in the config or
``--run_name``) and keep ``resume: auto`` (the default). Running the same
command again continues from the newest checkpoint in
``<output_dir>/<exp>/<run_name>/checkpoints`` — or from the hub copy when
``checkpoint.hub_repo_id`` is set and the local disk is empty.
"""

import argparse
import os

from dotenv import load_dotenv

import torch

from src.tinyllm.datasets.fineweb_helper import (
    DataLoaderConfig,
    create_nanogpt_dataloader,
)
from src.tinyllm.utils.misc import get_tokenizer, Config, get_exp_path
from src.tinyllm.utils.kernels import apply_kernel_patches, audit_kernel_patches
from src.tinyllm.utils.checkpoint import CheckpointManager, load_checkpoint
from src.tinyllm.utils.distributed import (
    barrier,
    cleanup_distributed,
    init_distributed,
)
from src.tinyllm.utils.model_stats import (
    count_params,
    flops_per_token,
    fmt_count,
)
from src.tinyllm.callbacks.wandb_callback import WandbCallback
from src.tinyllm.callbacks.analysis_callback import AnalysisCallback
from src.tinyllm.callbacks.benchmark_callback import BenchmarkCallback
from src.tinyllm.trainer.trainer import Trainer
from src.tinyllm.logger.logger_utils import configure_logging, logger
from src.tinyllm.factory.factory import (
    model_factory,
    optimizer_factory,
    lr_scheduler_factory,
)

load_dotenv()


def resolve_precision(requested: str, device: str) -> str:
    """ "auto" -> bf16 on Ampere+ (compute capability >= 8), fp16 on older
    GPUs (T4, V100, P100 — free-tier hardware), fp32 on CPU."""
    if requested != "auto":
        return requested
    if not device.startswith("cuda"):
        return "fp32"
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    return "bf16" if major >= 8 else "fp16"


def run(args):
    model_config = Config.parse(args.model_config_path)
    train_config = Config.parse(args.config_path)
    if args.run_name:
        train_config["run_name"] = args.run_name
    if args.resume:
        train_config["resume"] = args.resume

    dist_info, device_str = init_distributed(train_config.device)
    train_config["device"] = device_str

    # Experiment folder derives from the model's `name` (in the model YAML);
    # `exp_path` in the training config is an optional override. All runs live
    # under a common `output_dir` (default "experiments/", which is gitignored).
    # A `run_name` makes the run folder stable, which is what resume needs;
    # without one each launch gets a fresh timestamped folder.
    exp_name = (
        train_config.get("exp_path")
        or model_config.get("name")
        or model_config.get("model_type", "experiment")
    )
    runs_root = train_config.get("output_dir", "experiments")
    exp_path, log_file = get_exp_path(
        os.path.join(runs_root, exp_name),
        run_name=train_config.get("run_name"),
        create=dist_info.is_main,
    )
    configure_logging(log_file if dist_info.is_main else None)
    if not dist_info.is_main:
        logger.setLevel("WARNING")
    logger.info(
        f"Run dir: {exp_path} | world_size={dist_info.world_size} | device={device_str}"
    )

    packed = train_config.get("packed_tokens", 8192)
    # RoPE resets positions per document (see builder._doc_position_ids), so a
    # packed row may hold many documents and packed_tokens is NOT bounded by
    # max_seq_len — only each document is (capped by the packer). Absolute
    # position embeddings (learned / sinusoidal) index positions across the
    # whole pack, so they still require max_seq_len >= packed_tokens.
    embedding = model_config.get("embedding", "learned_pe")
    if embedding != "rope_only":
        assert model_config.max_seq_len >= packed, (
            f"model max_seq_len ({model_config.max_seq_len}) must be >= "
            f"training packed_tokens ({packed}) for '{embedding}' position "
            f"embeddings, which index absolute positions across the whole "
            f"packed row. Use embedding: 'rope_only' to pack more tokens/step."
        )
    # max_seq_len is a property of the model (position-embedding / RoPE capacity),
    # so the model config is its single source of truth. The data pipeline's
    # per-document cap defaults to it; an optional train_config.max_seq_len may
    # only *lower* it (to train on shorter documents than the model supports).
    data_max_seq_len = (
        train_config.get("max_seq_len") or model_config.max_seq_len
    )
    assert data_max_seq_len <= model_config.max_seq_len, (
        f"train_config max_seq_len ({data_max_seq_len}) must be <= model "
        f"max_seq_len ({model_config.max_seq_len})."
    )
    train_config["max_seq_len"] = data_max_seq_len  # resolve for downstream
    precision = resolve_precision(
        train_config.get("precision", "bf16"), device_str
    )
    train_config["precision"] = precision
    assert not (
        device_str == "cpu" and precision in ("fp16", "bf16")
    ), f"{precision.upper()} training requires a CUDA device."
    logger.info(f"Precision: {precision}")

    device = torch.device(device_str)

    model = model_factory(model_config)
    model = apply_kernel_patches(model, train_config)
    # Training reads pre-tokenized shards; only the benchmarks need the
    # tokenizer, so skip loading it (and the HF download) when they are off.
    benchmarks_on = any(
        (b or {}).get("enabled", False)
        for b in (train_config.get("benchmarks", {}) or {}).values()
    )
    tokenizer = (
        get_tokenizer(model_config.tokenizer_name) if benchmarks_on else None
    )

    def _loader_cfg(is_train: bool, skip_batches: int = 0) -> DataLoaderConfig:
        return DataLoaderConfig(
            mode=train_config.get("mode", "varlen_packed"),
            file_pattern=train_config.get(
                "train_file_pattern" if is_train else "test_file_pattern", None
            ),
            device=device_str,
            packed_tokens=packed,
            max_seq_len=train_config.get("max_seq_len", 2048),
            align_to_bos=is_train,
            num_workers=train_config.get("num_workers", 0),
            prefetch_factor=train_config.get("prefetch_factor", 2),
            prefetch_queue_size=train_config.get("prefetch_queue_size", 2),
            max_batches=train_config.get("max_batches", 0) if is_train else 0,
            max_tokens=train_config.get("max_tokens", 0) if is_train else 0,
            rank=dist_info.rank,
            world_size=dist_info.world_size,
            skip_batches=skip_batches,
        )

    def make_train_loader(skip_batches: int):
        return create_nanogpt_dataloader(_loader_cfg(True, skip_batches))

    test_loader = create_nanogpt_dataloader(_loader_cfg(False))

    # ---- Resume / init_from -------------------------------------------------
    ckpt_cfg = train_config.get("checkpoint", {}) or {}
    ckpt_manager = CheckpointManager(
        exp_path,
        keep_last=ckpt_cfg.get("keep_last", 3),
        keep_every_steps=ckpt_cfg.get("keep_every_steps", 0),
        hub_repo_id=ckpt_cfg.get("hub_repo_id"),
        is_main=dist_info.is_main,
    )
    resume_state = None
    resume = train_config.get("resume", "auto")
    if resume and resume != "none":
        if resume == "auto":
            ckpt_path = ckpt_manager.latest_path()
            barrier()  # rank 0 may have just downloaded it from the hub
            if ckpt_path is None:
                ckpt_path = ckpt_manager.latest_path()
        else:
            ckpt_path = resume
        if ckpt_path:
            logger.info(f"Resuming from {ckpt_path}")
            resume_state = load_checkpoint(ckpt_path)
            if not train_config.get("resume_wandb_id"):
                train_config["resume_wandb_id"] = resume_state.get(
                    "wandb_run_id"
                )

    init_from = train_config.get("init_from")
    if resume_state is None and init_from:
        # Weights only: a new stage (mid-training, SFT) starts from a trained
        # model with a fresh optimizer, schedule and data position.
        logger.info(f"Initialising weights from {init_from}")
        state = load_checkpoint(init_from)
        model.load_state_dict(state.get("model", state))

    if getattr(train_config, "use_wandb", False) and not train_config.get(
        "resume_wandb_id"
    ):
        # Fix the W&B run id up front and store it in every checkpoint, so a
        # resumed run keeps logging into the same W&B run.
        import wandb

        train_config["resume_wandb_id"] = wandb.util.generate_id()

    optimizer = optimizer_factory(model, train_config)

    # ---- Size, compute and step budget -------------------------------------
    from src.tinyllm.utils.chinchilla import (
        chinchilla_num_steps,
        tokens_per_step,
    )

    stats = count_params(model)
    fpt = flops_per_token(
        model_config, stats["active_non_embedding"], data_max_seq_len
    )
    logger.info(
        f"Params: total {fmt_count(stats['total'])}, "
        f"embedding {fmt_count(stats['embedding'])}, "
        f"non-embedding {fmt_count(stats['non_embedding'])}, "
        f"active {fmt_count(stats['active'])} "
        f"(active non-embedding {fmt_count(stats['active_non_embedding'])}). "
        f"~{fpt / 1e9:.2f} GFLOPs/token (train)."
    )

    tps = tokens_per_step(train_config, dist_info.world_size)
    cc_cfg = train_config.get("chinchilla", {}) or {}
    tpp = cc_cfg.get("tokens_per_param", 20)
    n_params = stats["total"]
    cc_steps, cc_tokens = chinchilla_num_steps(n_params, tps, tpp)
    logger.info(
        f"Chinchilla estimate: {n_params / 1e6:.2f}M params x {tpp} tok/param "
        f"= {cc_tokens / 1e9:.2f}B tokens; at {tps:,} tokens/step "
        f"-> {cc_steps:,} steps to reach the budget."
    )
    if cc_cfg.get("enabled", False):
        train_config["num_training_steps"] = cc_steps
        logger.info(
            f"chinchilla.enabled: num_training_steps set to {cc_steps:,}."
        )

    num_training_steps = train_config.get("num_training_steps", 0)
    if num_training_steps == 0:
        # The loaders are iterable (no len), so without an explicit step count
        # the schedule falls back to max_batches, or effectively "forever".
        accum = (
            train_config.get("iters_to_accumulate", 1)
            if train_config.get("use_grad_accum", False)
            else 1
        )
        max_batches = train_config.get("max_batches", 0)
        num_epochs = train_config.get("num_epochs", 1)
        num_training_steps = (
            (max_batches // accum) * num_epochs
            if max_batches
            else 1_000_000_000
        )
    logger.info(
        f"Training for {num_training_steps:,} optimizer steps "
        f"x {tps:,} tokens/step = {num_training_steps * tps / 1e9:.2f}B tokens "
        f"(~{num_training_steps * tps * fpt:.2e} FLOPs)."
    )

    # warmup_steps as float (0-1) = fraction of training, int = absolute steps
    num_warmup_steps = train_config.get("warmup_steps", 0)
    if isinstance(num_warmup_steps, float) and 0 < num_warmup_steps < 1:
        num_warmup_steps = int(num_warmup_steps * num_training_steps)

    lr_scheduler = lr_scheduler_factory(
        train_config.lr_scheduler_type,
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        sched_cfg=train_config.get("lr_schedule", {}) or {},
    )

    # Callbacks run on the main rank only (W&B, analysis, benchmarks).
    callbacks = []
    if dist_info.is_main:
        if getattr(train_config, "use_wandb", False):
            callbacks.append(WandbCallback())

        analysis_config = train_config.get("analysis", {})
        if analysis_config.get("every_n_steps", 0) > 0:
            callbacks.append(AnalysisCallback(analysis_config))

        benchmark_config = train_config.get("benchmarks", {})
        if benchmark_config:
            callbacks.append(BenchmarkCallback(benchmark_config))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
        make_train_loader=make_train_loader,
        test_loader=test_loader,
        train_config=train_config,
        model_config=model_config,
        device=device,
        callbacks=callbacks,
        dist_info=dist_info,
        ckpt_manager=ckpt_manager,
        resume_state=resume_state,
        flops_per_token=fpt,
    )
    del resume_state  # the trainer copied what it needs; free host memory

    if dist_info.is_main:
        audit_kernel_patches(
            trainer.model, fused_ce=trainer._fused_ce, train_config=train_config
        )
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLLM Training")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "-m",
        "--model_config_path",
        type=str,
        required=True,
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Stable run folder name; required for resume: auto.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help='"auto" (default), "none", or a checkpoint path.',
    )
    run(parser.parse_args())
