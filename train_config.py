"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 10:39:42
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-20 16:52:01
 # @ Description:
 """

from models.helper import Config


class GPTTrainConfig(Config):
    """
    Training config for GPT models.
    """

    model_type = "gpt"

    num_epochs = 10
    batch_size = 16
    avg_seq_len_in_batch = 128
    num_workers = 4
    persistent_workers = True

    lr_scheduler_type = "cosine"
    init_lr = 1e-3
    warmup_epochs = 0
    label_smoothing = 0.1
