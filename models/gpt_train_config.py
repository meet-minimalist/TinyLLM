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
    batch_size = 4
    avg_seq_len_in_batch = 1024
    max_seq_len = 1024
    num_workers = 4
    persistent_workers = True

    lr_scheduler_type = "cosine"
    init_lr = 1e-3
    warmup_epochs = 2
    label_smoothing = 0.1

    use_wandb = True
    resume_wandb_id = None
    track_gradients = False
    fp16_training = True


if __name__ == "__main__":
    g = GPTTrainConfig()
    g.print_config()
