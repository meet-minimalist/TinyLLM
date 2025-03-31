"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-08-05 20:35:25
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-08-05 20:35:28
 # @ Description:
 """

import os
from typing import Dict

import torch


class CheckpointHandler:
    def __init__(
        self, ckpt_dir: str, model_name: str = "model", max_to_keep: int = 3
    ):
        """Initializer for CheckpointHandler.
        This will save model whenever called and it will only keep track of
        last n number of epochs data only.

        Args:
            ckpt_dir (str): Directory where all the checkpoints are saved.
            model_name (str, optional): Model name to use while saving the
                checkpoint. Defaults to "model".
            max_to_keep (int, optional): Number of last checkpoints to retain.
                Defaults to 3.
        """
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.max_to_keep = max_to_keep
        self.ckpt_path_history = []

    def __get_ckpt_path(
        self,
        global_step: int,
        eps: int,
        last_train_loss: float,
        last_train_ppl: float,
        test_loss: float,
    ) -> str:
        """Function to get the checkpoint path based on given epoch and loss value.

        Args:
            global_step (int): Global training step.
            eps (int): Epoch number.
            last_train_loss (float): Training loss of most recent batch.
            last_train_ppl (float): Training PPL of most recent batch.
            test_loss (float): Test loss value evaluated the end of epoch.

        Returns:
            str: Checkpoint path.
        """
        if eps is not None and test_loss is not None:
            ckpt_name = (
                f"{self.model_name}_eps_{eps}_test_loss_{test_loss:.4f}.pt"
            )
        elif (
            global_step is not None
            and last_train_loss is not None
            and last_train_ppl is not None
        ):
            ckpt_name = f"{self.model_name}_step_{global_step}_train_loss_{last_train_loss:.4f}_train_ppl_{last_train_ppl:.4f}.pt"
        else:
            ckpt_name = f"{self.model_name}.pt"
        cur_ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        return cur_ckpt_path

    def save(self, checkpoint_state: Dict) -> None:
        """Function to save the current checkpoint based on provided checkpoint
        dict.

        Args:
            checkpoint_state (Dict): Checkpoint dict which contains epoch_num,
                test_loss value, checkpoint statedict.
        """
        eps = checkpoint_state.get("epoch", None)
        test_loss = checkpoint_state.get("test_loss", None)
        global_step = checkpoint_state.get("global_step", None)
        last_train_loss = checkpoint_state.get("last_train_loss", None)
        last_train_ppl = checkpoint_state.get("last_train_ppl", None)

        cur_ckpt_path = self.__get_ckpt_path(
            global_step, eps, last_train_loss, last_train_ppl, test_loss
        )

        torch.save(checkpoint_state, cur_ckpt_path)

        self.ckpt_path_history.append(cur_ckpt_path)

        if len(self.ckpt_path_history) > self.max_to_keep:
            remove_ckpt_path = self.ckpt_path_history.pop(0)
            os.remove(remove_ckpt_path)
