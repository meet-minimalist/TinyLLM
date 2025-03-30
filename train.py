"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 10:36:26
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-20 16:56:20
 # @ Description:
 """

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import importlib
import argparse

from dataset_helper import DatasetHelper
from models.helper import (
    train_config_factory,
    model_config_factory,
    model_factory,
)
from utils.misc import get_tokenizer, lr_scheduler_factory
from torch.amp import GradScaler
from utils.loss_helper import compute_ce_loss


def run(args):
    train_config = train_config_factory(args.model_name)
    model_config = model_config_factory(args.model_name)
    device = torch.device(train_config.device)
    model = model_factory(args.model_name, model_config)
    model.to(device)
    model = torch.compile(model)

    tokenizer = get_tokenizer(train_config.model_type)

    train_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.max_seq_len,
        train_config.num_workers,
        train_config.persistent_workers,
        train_config.use_pin_memory,
        "validation",
    )
    train_loader = train_helper.get_loader()
    valid_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.max_seq_len,
        train_config.num_workers,
        train_config.persistent_workers,
        train_config.use_pin_memory,
        "validation",
    )
    valid_loader = valid_helper.get_loader()

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

    g_step = 0
    for eps_num in range(train_config.num_epochs):
        model.train()
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(
            train_loader
        ):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            if train_config.fp16_training:
                # Runs the forward pass with autocasting.
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=(
                        torch.float16
                        if torch.cuda.is_available()
                        else torch.float32
                    ),
                ):
                    logits = model(input_ids, attn_mask)
                    loss = compute_ce_loss(
                        logits,
                        labels,
                        train_config.label_smoothing,
                        tokenizer.pad_token_id,
                    )
                    loss = loss / train_config.iters_to_accumulate
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids, attn_mask)
                loss = compute_ce_loss(
                    logits,
                    labels,
                    train_config.label_smoothing,
                    tokenizer.pad_token_id,
                )

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

            print(
                f"Epoch: {eps_num+1}/{train_config.num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss:.4f}, LR: {lr:.4f}"
            )

        # model.eval()
        # total_eval_loss = 0
        # for batch in eval_dataloader:
        #     with torch.no_grad():
        #         input_ids = batch['input_ids']
        #         attention_mask = batch['attention_mask']
        #         labels = batch['labels']
        #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        #         loss = outputs.loss
        #         total_eval_loss += loss.item()

        # avg_eval_loss = total_eval_loss / len(eval_dataloader)
        # print(f"Epoch {epoch+1}, Evaluation Loss: {avg_eval_loss}")

    # Save the model
    # model.save_pretrained("path/to/save/model")
    # tokenizer.save_pretrained("path/to/save/tokenizer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLLM Training helper.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to train.",
    )
    args = parser.parse_args()
    run(args)
