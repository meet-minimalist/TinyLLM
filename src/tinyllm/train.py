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
from src.tinyllm.callbacks.checkpoint_callback import CheckpointCallback
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


def run(args):
    model_config = Config.parse(args.model_config_path)
    train_config = Config.parse(args.config_path)

    exp_path, log_file = get_exp_path(train_config.exp_path)
    configure_logging(log_file)

    packed = train_config.get("packed_tokens", 8192)
    assert model_config.max_seq_len >= packed, (
        f"model max_seq_len ({model_config.max_seq_len}) must be >= "
        f"training packed_tokens ({packed}). "
        f"The position embedding needs at least packed_tokens positions."
    )
    assert (
        model_config.max_seq_len == train_config.max_seq_len
    ), f"model max_seq_len ({model_config.max_seq_len}) must equal train_config max_seq_len ({train_config.max_seq_len})"
    precision = getattr(train_config, "precision", "bf16")
    assert not (
        train_config.device == "cpu" and precision in ("fp16", "bf16")
    ), f"{precision.upper()} training requires a CUDA device."

    device = torch.device(train_config.device)

    model = model_factory(model_config)
    model = apply_kernel_patches(model, train_config)
    tokenizer = get_tokenizer(model_config.tokenizer_name)

    def _build_loader(is_train: bool) -> DataLoaderConfig:
        return DataLoaderConfig(
            mode=train_config.get("mode", "varlen_packed"),
            file_pattern=train_config.get(
                "train_file_pattern" if is_train else "test_file_pattern", None
            ),
            device=train_config.device,
            packed_tokens=train_config.get("packed_tokens", 8192),
            max_seq_len=train_config.get("max_seq_len", 2048),
            align_to_bos=is_train,
            num_workers=train_config.get("num_workers", 0),
            prefetch_factor=train_config.get("prefetch_factor", 2),
            prefetch_queue_size=train_config.get("prefetch_queue_size", 2),
            max_batches=train_config.get("max_batches", 0),
            max_tokens=train_config.get("max_tokens", 0),
        )

    train_loader = create_nanogpt_dataloader(_build_loader(True))
    test_loader = create_nanogpt_dataloader(_build_loader(False))

    optimizer = optimizer_factory(model, train_config)

    num_warmup_steps = train_config.get("warmup_steps", 0)
    # num_training_steps for the LR scheduler (needs a concrete value)
    num_training_steps = train_config.get("num_training_steps", 0)
    if num_training_steps == 0:
        try:
            steps_per_epoch = len(train_loader)
            num_training_steps = (
                train_config.get("num_epochs", 1) * steps_per_epoch
            )
        except Exception:
            max_batches = train_config.get("max_batches", 0)
            num_epochs = train_config.get("num_epochs", 1)
            num_training_steps = (max_batches or 1_000_000_000) * num_epochs

    # warmup_steps as float (0-1) = fraction of training, int = absolute steps
    if isinstance(num_warmup_steps, float) and 0 < num_warmup_steps < 1:
        num_warmup_steps = int(num_warmup_steps * num_training_steps)

    lr_scheduler = lr_scheduler_factory(
        train_config.lr_scheduler_type,
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    callbacks = [
        CheckpointCallback(
            exp_path,
            "model",
            max_to_keep=3,
            save_every_steps=train_config.get("save_every_steps", 0),
        ),
    ]
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
        train_loader=train_loader,
        test_loader=test_loader,
        train_config=train_config,
        model_config=model_config,
        device=device,
        callbacks=callbacks,
    )
    # After trainer construction, use trainer.model instead of model, since
    # model object is compiled using torch.compile.

    audit_kernel_patches(
        trainer.model, fused_ce=trainer._fused_ce, train_config=train_config
    )
    trainer.train()


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
    run(parser.parse_args())
