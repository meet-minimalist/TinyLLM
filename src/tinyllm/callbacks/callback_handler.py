"""
Callback handler that dispatches lifecycle events to registered callbacks.
"""


class CallbackHandler:
    """Manages the execution of multiple callbacks."""

    def __init__(self, callbacks: list):
        self.callbacks = callbacks

    def on_train_begin(self, **kwargs):
        for cb in self.callbacks:
            cb.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        for cb in self.callbacks:
            cb.on_train_end(**kwargs)

    def on_epoch_begin(self, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_end(**kwargs)

    def on_train_step_begin(self, **kwargs):
        for cb in self.callbacks:
            cb.on_train_step_begin(**kwargs)

    def on_train_step_end(self, **kwargs):
        for cb in self.callbacks:
            cb.on_train_step_end(**kwargs)
