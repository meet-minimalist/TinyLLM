"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 10:36:26
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-20 16:56:20
 # @ Description:
 """

import os

import torch
from tqdm import tqdm
import argparse

from dataset_helper import DatasetHelper
from models.helper import (
    train_config_factory,
    model_config_factory,
    model_factory,
)
from utils.misc import get_tokenizer, lr_scheduler_factory, init_wandb
from utils.checkpoint_handler import CheckpointHandler
from utils.logger_utils import configure_logging, logger
from torch.amp import GradScaler
from utils.loss_helper import compute_ce_loss
from torchinfo import summary


def run(args):
    train_config = train_config_factory(args.model_name)
    exp_path = train_config.base_exp_path
    configure_logging(train_config.log_file)
    model_config = model_config_factory(args.model_name)
    assert (
        model_config.max_seq_len == train_config.max_seq_len
    ), "Model config and training config should have same value for max_seq_len."  # For Positional Embeddings.
    assert not (
        train_config.device == "cpu" and train_config.fp16_training
    ), "FP16 Training is only available for CUDA devices."
    device = torch.device(train_config.device)
    model = model_factory(args.model_name, model_config)
    model.to(device)
    model = torch.compile(model)

    input_ids = torch.zeros(2, 128).to(torch.int32).to(device)
    attn_mask = torch.zeros(2, 128).to(torch.int32).to(device)
    summary(model, input_data=[input_ids, attn_mask])
    model_param_count = sum([torch.numel(p) for p in model.parameters()])
    print(
        f"Training {train_config.model_type} model with {model_param_count:,} params."
    )

    tokenizer = get_tokenizer(train_config.model_type)

    train_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.max_seq_len,
        train_config.num_workers,
        train_config.persistent_workers,
        train_config.use_pin_memory,
        train_config.sample_similar_len,
        "train",
    )
    train_loader = train_helper.get_loader()
    valid_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.max_seq_len,
        train_config.num_workers,
        train_config.persistent_workers,
        train_config.use_pin_memory,
        train_config.sample_similar_len,
        "validation",
    )
    valid_loader = valid_helper.get_loader()

    ckpt_handler = CheckpointHandler(exp_path, "model", max_to_keep=3)
    lr_scheduler = lr_scheduler_factory(
        train_config.lr_scheduler_type,
        init_lr=train_config.init_lr,
        epochs=train_config.num_epochs,
        warmup_epochs=train_config.warmup_epochs,
        steps_per_epoch=len(train_loader),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0, weight_decay=0.1, fused=True
    )

    if train_config.fp16_training:
        scaler = GradScaler()
    if train_config.use_wandb:
        import wandb

        init_wandb(train_config, model_config, train_config.resume_wandb_id)
        if train_config.track_gradients:
            wandb.watch(model)

    g_step = 0
    for eps_num in range(train_config.num_epochs):
        model.train()
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(
            train_loader
        ):
            batch_size = input_ids.shape[0]
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train_config.fp16_training:
                # Runs the forward pass with autocasting.
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids, attn_mask)
                    loss = compute_ce_loss(
                        logits,
                        labels,
                        train_config.label_smoothing,
                        tokenizer.pad_token_id,
                    )
                    loss = loss / train_config.iters_to_accumulate
                loss = scaler.scale(loss)
                loss.backward()
                ppl = torch.exp(loss)
            else:
                logits = model(input_ids, attn_mask)
                loss = compute_ce_loss(
                    logits,
                    labels,
                    train_config.label_smoothing,
                    tokenizer.pad_token_id,
                )
                ppl = torch.exp(loss)

            lr = lr_scheduler.step(g_step, optimizer)

            if train_config.use_grad_accum:
                if (
                    (batch_idx + 1) % train_config.iters_to_accumulate == 0
                ) or (batch_idx + 1 == len(train_loader)):
                    if train_config.fp16_training:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                else:
                    # Dont zero the gradients. We need to accumulate them.
                    pass
            else:
                if train_config.fp16_training:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            logger.info(
                f"Epoch: {eps_num+1}/{train_config.num_epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                f"Batch Size: {batch_size}, Loss: {loss:.4f}, "
                f"PPL: {ppl:.4f}, LR: {lr:.4f}"
            )
            metrics = {
                "Epoch": eps_num + 1,
                "Batch": batch_idx + 1,
                "Loss": loss,
                "Perplexity": ppl,
                "LR": lr,
            }
            if train_config.use_wandb:
                wandb.log(metrics, step=g_step)

            if (g_step + 1) % 1000 == 0:
                # Save checkpoint at every 1000 steps.
                checkpoint = {
                    "global_step": g_step,
                    "last_train_loss": loss,
                    "last_train_ppl": ppl,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": (
                        scaler.state_dict()
                        if train_config.fp16_training
                        else None
                    ),
                }
                ckpt_handler.save(checkpoint)

            g_step += 1

        model.eval()
        total_eval_loss = 0
        total_eval_ppl = 0
        with torch.no_grad():
            for input_ids, attn_mask, labels in tqdm(valid_loader):
                input_ids = input_ids.to(device, non_blocking=True)
                attn_mask = attn_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(input_ids, attn_mask)

                # We would take mean across all sequence length and all batches.
                loss = compute_ce_loss(
                    logits,
                    labels,
                    0,
                    tokenizer.pad_token_id,
                )
                ppl = torch.exp(loss)
                total_eval_loss += loss.item()
                total_eval_ppl += ppl.item()

        avg_eval_loss = total_eval_loss / len(valid_loader)
        avg_eval_ppl = total_eval_ppl / len(valid_loader)
        logger.info(
            f"Epoch {eps_num+1}, Evaluation Loss: {avg_eval_loss:.4f}, "
            f"Evaluation Perplexity: {avg_eval_ppl:.4f}"
        )

        if train_config.use_wandb:
            metrics = {"Test Loss": loss}
            wandb.log(metrics, step=g_step)

        # Save the model
        torch.save(model.state_dict(), "model.pth")

        checkpoint = {
            "epoch": eps_num,
            "global_step": g_step,
            "test_loss": avg_eval_loss,
            "test_ppl": avg_eval_ppl,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": (
                scaler.state_dict() if train_config.fp16_training else None
            ),
        }
        ckpt_handler.save(checkpoint)

    if train_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLLM Training helper.")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to train.",
    )
    args = parser.parse_args()
    run(args)
