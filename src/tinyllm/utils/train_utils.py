from contextlib import nullcontext
import torch


def get_autocast_ctx(device: torch.device, precision: str):
    """Return autocast context for the given precision string.

    precision: "bf16" | "fp16" | "fp32"
    BF16 is preferred — same range as FP32, no GradScaler needed.
    """
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()
