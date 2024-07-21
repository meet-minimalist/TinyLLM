"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 12:08:18
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-20 16:53:57
 # @ Description:
 """

import logging

from transformers import PreTrainedTokenizer

from utils.lr_utils.cosine_annealing_lr import CosineAnnealing
from utils.lr_utils.exp_decay_lr import ExpDecay
from utils.lr_utils.lr_scheduler import LearningRateScheduler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_tokenizer(model_type: str) -> PreTrainedTokenizer:
    """
    Get the tokenizer based on the model_type.

    Args:
        model_type (str): Name of the model.

    Raises:
        NotImplementedError: If unsupported model_type is provided then this
            exception will be thrown.

    Returns:
        PreTrainedTokenizer: Returns the tokenizer instance for a given model.
    """
    if model_type == "gpt":
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise NotImplementedError(
            f"No tokenizer implemented for model type: {model_type}"
        )


def lr_scheduler_factory(
    scheduler_type: str, *args, **kwargs
) -> LearningRateScheduler:
    """Function to get learning rate scheduler based on provided type.

    Args:
        scheduler_type (str): Type of scheduler.

    Raises:
        NotImplementedError: If the type of scheduler is not implemented then this
            will be raised.

    Returns:
        LearningRateScheduler: LR Scheduler instance.
    """
    mapping = {"cosine": CosineAnnealing, "exp": ExpDecay}

    scheduler_class = mapping.get(scheduler_type, None)
    if scheduler_class is None:
        raise NotImplementedError(
            f"Scheduler of type: {scheduler_type} is not implemented."
        )

    return scheduler_class(*args, **kwargs)


def configure_logging(log_file: str) -> None:
    """
    Configure the logger to dump the logs in given file.

    Args:
        log_file (str): Log file to store the logs.
    """
    for handler in logger.handlers:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
