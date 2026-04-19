"""
# @ Author: Meet Patel
# @ Create Time: 2024-11-03 15:38:28
# @ Modified by: Meet Patel
# @ Modified time: 2026-01-12 20:45:58
# @ Description:
"""

"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 12:08:18
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-20 16:53:57
 # @ Description:
 """

import datetime
import os
import yaml
from box import Box
from dataclasses import dataclass
from typing import Any, Dict

from src.tinyllm.logger.logger_utils import logger


def get_tokenizer(tokenizer_name: str):
    """
    Get the tokenizer based on the tokenizer_name.

    This function tries to use `transformers`' tokenizer when available.
    When `transformers` is not installed (e.g., in light-weight CI), it falls
    back to a minimal in-memory tokenizer sufficient for smoke tests and
    unit-tests.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_exp_path(base_dir: str) -> str:
    """Function to get the directory to same the experiment related data.

    Args:
        base_dir (str): Directory to store all experiments.

    Returns:
        str: Path for current experiment.
    """
    start_time = datetime.datetime.now()
    time_stamp = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    if base_dir is None:
        base_exp_path = f"./exp/{time_stamp}"
    else:
        base_exp_path = os.path.join(base_dir, time_stamp)
    os.makedirs(base_exp_path, exist_ok=True)
    log_file = os.path.join(base_exp_path, "log.txt")
    return base_exp_path, log_file


class Config:
    @classmethod
    def parse(cls, file_path: str) -> Box:
        """Parse a YAML file and return a Box instance."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        return Box(data)

    @staticmethod
    def print_content(file_path: str):
        """Print the YAML content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        print(f"--- Content of {file_path} ---")
        print(yaml.dump(data, default_flow_style=False))
        print("-" * 30)
