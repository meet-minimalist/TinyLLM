from contextlib import nullcontext
import torch


def get_autocast_ctx(device: torch.device, use_amp: bool):
    if use_amp:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()
