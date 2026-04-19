"""
WandB callback — logs metrics to Weights & Biases.
"""

from src.tinyllm.callbacks.base_callback import BaseCallback


class WandbCallback(BaseCallback):
    """Logs training/evaluation metrics to Weights & Biases."""

    def __init__(self):
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "WandbCallback requires wandb. Run: pip install wandb"
            )

    def on_train_begin(self, **kwargs):
        train_config = kwargs.get("train_config")
        model_config = kwargs.get("model_config")
        model = kwargs.get("model")

        config_dict = {}
        if train_config:
            config_dict.update(train_config.to_dict())
        if model_config:
            config_dict.update(model_config.to_dict())

        resume_id = getattr(train_config, "resume_wandb_id", None)
        self.wandb.init(
            project="TinyLLM",
            config=config_dict,
            resume="allow",
            id=resume_id,
        )

        if (
            getattr(train_config, "track_weight_stats", False)
            and model is not None
        ):
            try:
                self.wandb.watch(model, log="all", log_freq=100)
            except Exception:
                pass

    def on_train_end(self, **kwargs):
        self.wandb.finish()

    def on_train_step_end(self, **kwargs):
        metrics = kwargs.get("metrics", {})
        global_step = kwargs.get("global_step", 0)
        self.wandb.log(metrics, step=global_step)

    def on_epoch_end(self, **kwargs):
        metrics = kwargs.get("metrics", {})
        global_step = kwargs.get("global_step", 0)
        self.wandb.log(metrics, step=global_step)
