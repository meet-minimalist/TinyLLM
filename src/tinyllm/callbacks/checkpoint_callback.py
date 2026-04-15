"""
Checkpoint callback — saves model checkpoints at epoch end.
"""

from src.tinyllm.callbacks.base_callback import BaseCallback
import os
import torch


class CheckpointCallback(BaseCallback):
    """Saves checkpoints at the end of each epoch, keeping only the last N."""

    def __init__(
        self, ckpt_dir: str, model_name: str = "model", max_to_keep: int = 3
    ):
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.max_to_keep = max_to_keep
        self.ckpt_path_history = []

    def _ckpt_path(self, epoch: int, global_step: int, test_loss: float) -> str:
        if epoch is not None and test_loss is not None:
            name = f"{self.model_name}_ep{epoch}_loss{test_loss:.4f}.pt"
        elif global_step is not None:
            name = f"{self.model_name}_step{global_step}.pt"
        else:
            name = f"{self.model_name}.pt"
        return os.path.join(self.ckpt_dir, name)

    def on_epoch_end(self, **kwargs):
        epoch = kwargs.get("epoch")
        global_step = kwargs.get("global_step")
        test_loss = kwargs.get("test_loss")

        ckpt_path = self._ckpt_path(
            epoch, global_step, test_loss or float("inf")
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "test_loss": test_loss,
            "model": kwargs.get("model"),
            "optimizer": kwargs.get("optimizer"),
            "scaler": kwargs.get("scaler"),
        }
        torch.save(checkpoint, ckpt_path)
        self.ckpt_path_history.append(ckpt_path)

        # Remove oldest checkpoints beyond max_to_keep
        while len(self.ckpt_path_history) > self.max_to_keep:
            old_path = self.ckpt_path_history.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
