"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 10:37:50
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-12 22:06:15
 # @ Description:
 """

from __future__ import annotations

from abc import ABC

from utils.misc import logger


class Config(ABC):
    @classmethod
    def print_config(cls: Config):
        """
        Print the static variables (class variables) of the class.
        """
        class_vars = {}
        for name, value in cls.__dict__.items():
            if callable(value):
                continue
            if isinstance(value, classmethod) or isinstance(
                value, staticmethod
            ):
                continue
            if name.startswith("__") or name == "_abc_impl":
                continue
            class_vars[name] = value

        logger.info("#" * 49)
        logger.info(f"#{cls.__name__.center(47)}#")
        for name, value in class_vars.items():
            logger.info(f"#\t{name} \t\t: {value}\t\t#")
        logger.info("#" * 49)


def model_factory(model_type: str, config: Config):
    """
    Get the model based on model_type and model config.

    Args:
        model_type (str): Name of the model type.
        config (Config): Model architectural config.

    Raises:
        NotImplementedError: If unsupported model_type is provided then this
            exception will be thrown.

    Returns:
        Model: Torch model as per the model_type and config.
    """
    if model_type == "gpt":
        from models.gpt import GPTModel

        return GPTModel(config)
    else:
        raise NotImplementedError(
            f"No model implemented for model type: {model_type}"
        )


def config_factory(model_type: str) -> Config:
    """
    Get the model config based on the model_type.

    Args:
        model_type (str): Name of the model type.

    Raises:
        NotImplementedError: If unsupported model_type is provided then this
            exception will be thrown.

    Returns:
        Config: Config class return for given model_type.
    """
    if model_type == "gpt":
        from models.gpt_config import GPTConfig

        return GPTConfig
    else:
        raise NotImplementedError(
            f"No model config implemented for model type: {model_type}"
        )
