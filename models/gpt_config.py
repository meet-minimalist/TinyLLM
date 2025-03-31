"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-06 22:45:44
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-12 22:10:36
 # @ Description:
 """

from models.helper import Config


class GPTConfig(Config):
    """
    Model Config for GPT model.
    """

    # vocab_size = 50257  # pad it to make it a multiple of 64 == 50304
    vocab_size = 50304
    emb_dim = 128
    max_seq_len = 512
    num_heads = 2
    drop_prob = 0.1
    ff_multiplier = 1
    num_blocks = 2
    tie_weights = True


if __name__ == "__main__":
    g = GPTConfig()
    g.print_config()
