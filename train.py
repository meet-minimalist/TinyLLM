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
from torch.nn import CrossEntropyLoss

import train_config
from dataset_helper import DatasetHelper
from models.helper import config_factory, model_factory
from utils.misc import get_tokenizer, lr_scheduler_factory


def run():
    cuda = torch.device("cuda")

    model_config = config_factory(train_config.model_type)
    model = model_factory(train_config.model_type, model_config)
    model.to(cuda)

    tokenizer = get_tokenizer(train_config.model_type)

    train_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.avg_seq_len_in_batch,
        train_config.num_workers,
        train_config.persistent_workers,
        "validation",
    )
    train_loader = train_helper.get_loader()
    valid_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.avg_seq_len_in_batch,
        train_config.num_workers,
        train_config.persistent_workers,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.1)
    loss_fn = CrossEntropyLoss(
        reduction="mean",
        label_smoothing=train_config.label_smoothing,
        ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
    )

    g_step = 0
    for eps_num in range(train_config.num_epochs):
        model.train()
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(
            train_loader
        ):
            optimizer.zero_grad()
            input_ids = input_ids.to(cuda)
            attn_mask = attn_mask.to(cuda)
            labels = labels.to(cuda)

            logits = model(input_ids, attn_mask)

            batch_size = logits.shape[0]
            logits = logits.view(-1, logits.shape[2])
            labels = labels.view(-1).to(torch.long)

            # We would take mean across all sequence length and all batches.
            loss = loss_fn(logits, labels) * batch_size
            loss.backward()

            lr = lr_scheduler.step(g_step, optimizer)
            optimizer.step()

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
    run()
