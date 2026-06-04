from src.tinyllm.logger.logger_utils import logger


class LayerRegistry:
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
            raise KeyError(
                f"'{name}' not registered in {self._name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]


LAYER_REGISTRY = LayerRegistry("Layer")
