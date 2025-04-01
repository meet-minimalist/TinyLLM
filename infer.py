"""
 # @ Author: Meet Patel
 # @ Create Time: 2025-04-04 20:14:11
 # @ Modified by: Meet Patel
 # @ Modified time: 2025-04-01 20:14:11
 # @ Description:
 """

import os

import torch
import argparse
import time

from models.helper import (
    model_config_factory,
    model_factory,
)
from utils.misc import get_tokenizer


def run(args):
    model_config = model_config_factory(args.model_name)
    device = torch.device(args.device)
    model = model_factory(args.model_name, model_config)
    model.to(device)
    model = torch.compile(model)
    model.eval()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("Checkpoint not found.")

    model.load_state_dict(
        torch.load(args.checkpoint, weights_only=False)["model"]
    )

    tokenizer = get_tokenizer(args.model_name)

    input_string = args.prompt

    predicted_string = ""
    cnt = 0
    start = time.time()
    while True:
        if cnt >= args.max_tokens:
            print("Max tokens reached.")
            break
        input_data = tokenizer.tokenize(input_string)
        input_ids = input_data["input_ids"].to(device)
        mask = input_data["attention_mask"].to(device)
        logits = model(input_ids, mask)
        token = torch.argmax(logits[:, -1, :])
        if token == tokenizer.eos_token_id:
            print("EOS reached.")
            break
        token_char = tokenizer.decode(token)
        predicted_string += token_char
        input_string += token_char
        cnt += 1
    delta = time.time() - start

    print(f"Total time: {delta} sec")
    print(f"Input Prompt: {args.prompt}")
    print(f"Generated output: {predicted_string}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLLM Training helper.")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to train.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model execution.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint of the model.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Once upon a time in,",
        help="Initial prompt to be provided to the model to start the generation process.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()
    run(args)
