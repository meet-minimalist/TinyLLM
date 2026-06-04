import argparse

import torch

from src.tinyllm.datasets.fineweb_helper import (
    DataLoaderConfig,
    create_nanogpt_dataloader,
)
from src.tinyllm.utils.misc import get_tokenizer, Config, get_exp_path
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


def run(args):
    model_config = Config.parse(args.model_config_path)
    train_config = Config.parse(args.config_path)

    exp_path, log_file = get_exp_path(train_config.exp_path)
    configure_logging(log_file)

    assert (
        model_config.max_seq_len == train_config.max_seq_len
    ), "model_config.max_seq_len must equal train_config.max_seq_len"
    assert not (
        train_config.device == "cpu"
        and getattr(train_config, "fp16_training", False)
    ), "FP16 training is only available for CUDA devices."

    device = torch.device(train_config.device)

    model = model_factory(model_config)
    tokenizer = get_tokenizer(model_config.tokenizer_name)

    train_cfg = DataLoaderConfig(
        mode=train_config.get("mode", "varlen_packed"),
        file_pattern=train_config.get("train_file_pattern", None),
        device=train_config.device,
        packed_tokens=train_config.get("packed_tokens", 8192),
        max_seq_len=train_config.get("max_seq_len", 2048),
        align_to_bos=True,
        num_workers=train_config.get("num_workers", 2),
        prefetch_factor=train_config.get("prefetch_factor", 4),
    )
    train_loader = create_nanogpt_dataloader(train_cfg)

    test_cfg = DataLoaderConfig(
        mode=train_config.get("mode", "varlen_packed"),
        file_pattern=train_config.get("test_file_pattern", None),
        device=train_config.device,
        packed_tokens=train_config.get("packed_tokens", 8192),
        max_seq_len=train_config.get("max_seq_len", 2048),
        align_to_bos=False,
        num_workers=train_config.get("num_workers", 2),
        prefetch_factor=train_config.get("prefetch_factor", 4),
    )
    test_loader = create_nanogpt_dataloader(test_cfg)

    optimizer = optimizer_factory(model, train_config)

    num_warmup_steps = train_config.get("warmup_steps", 0)
    try:
        steps_per_epoch = len(train_loader)
        num_training_steps = train_config.get("num_epochs", 1) * steps_per_epoch
    except Exception:
        num_training_steps = train_config.get("num_training_steps", 100)

    lr_scheduler = lr_scheduler_factory(
        train_config.lr_scheduler_type,
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    callbacks = [
        CheckpointCallback(exp_path, "model", max_to_keep=3),
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
