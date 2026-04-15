"""
# @ Author: Meet Patel
# @ Create Time: 2026-01-08 22:10:15
# @ Modified by: Meet Patel
# @ Modified time: 2026-01-12 20:42:14
# @ Description:
"""

from src.tinyllm.logger.logger_utils import logger


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, name: str = None):
        def _register(cls):
            key = name if name is not None else cls.__name__
            if key in self._registry:
                logger.warning(f"{key} is already registered in {self._name}")
            self._registry[key] = cls
            return cls

        return _register

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f"{name} is not registered in {self._name}")
        return self._registry[name]


MODEL_REGISTRY = Registry("Model")
LR_SCHEDULER_REGISTRY = Registry("LRScheduler")
