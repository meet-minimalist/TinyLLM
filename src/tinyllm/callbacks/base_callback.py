"""
Base callback class for the training lifecycle.
"""

from abc import ABC


class BaseCallback(ABC):
    """Base class for all callbacks used in the training process."""

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_train_step_begin(self, **kwargs):
        pass

    def on_train_step_end(self, **kwargs):
        pass
