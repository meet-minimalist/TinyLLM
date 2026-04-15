"""
Inference entry point — loads a trained checkpoint and generates text.

Usage:
    python -m src.tinyllm.infer \
        -m configs/models/gpt.yaml \
        -c /path/to/checkpoint.pt \
        -p "Once upon a time" \
        --max_tokens 100
"""

import argparse
import os
import time

import torch

from src.tinyllm.utils.misc import get_tokenizer, Config
from src.tinyllm.factory.factory import model_factory


@torch.no_grad()
def generate(
    model, tokenizer, prompt: str, max_tokens: int = 50, device: str = "cpu"
) -> str:
    """Generate text token-by-token from a prompt."""
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = []

    for _ in range(max_tokens):
        mask = torch.ones_like(input_ids, device=device)
        logits = model(input_ids, mask)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        tid = next_token.item()
        if tid == tokenizer.eos_token_id:
            break

        generated.append(tid)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generated)


def run(args):
    model_config = Config.parse(args.model_config_path)
    device = torch.device(args.device)

    model = model_factory(model_config)
    model.to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Support both {"model": ...} and raw state_dict
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)

    tokenizer = get_tokenizer(model_config.tokenizer_name)

    start = time.time()
    output = generate(model, tokenizer, args.prompt, args.max_tokens, device)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s")
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLLM Inference")
    parser.add_argument(
        "-m",
        "--model_config_path",
        type=str,
        required=True,
        help="Model config YAML",
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Checkpoint file"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="Device (cpu/cuda)"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=50, help="Max tokens to generate"
    )
    run(parser.parse_args())
