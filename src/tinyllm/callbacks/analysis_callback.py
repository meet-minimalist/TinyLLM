import time

from src.tinyllm.callbacks.base_callback import BaseCallback
from src.tinyllm.logger.logger_utils import logger


class AnalysisCallback(BaseCallback):
    """
    Runs weight stats, activation stats, and spectral analysis
    at configurable intervals. Logs timing for each component.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.every_n_steps = config.get("every_n_steps", 500)
        self.track_weights = config.get("track_weights", True)
        self.track_activations = config.get("track_activations", False)
        self.track_spectral = config.get("spectral", False)
        self.variance_thresholds = config.get(
            "variance_thresholds", [0.95, 0.99]
        )
        self.activation_captures = {}
        self._hooks = []

    def on_train_begin(self, **kwargs):
        model = kwargs.get("model")
        if self.track_activations and model is not None:
            self._register_activation_hooks(model)

    def _register_activation_hooks(self, model):
        for name, module in model.named_modules():
            if "attn" in name and hasattr(module, "forward"):
                from src.tinyllm.analysis.activation_stats import (
                    ActivationCapture,
                )

                capture = ActivationCapture(name)
                self.activation_captures[name] = capture
                self._hooks.append(module.register_forward_hook(capture))

    def on_train_step_end(self, **kwargs):
        global_step = kwargs.get("global_step", 0)
        if global_step == 0 or global_step % self.every_n_steps != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        log_dict = {}
        timing_log = {}

        # 1. Weight stats
        if self.track_weights:
            from src.tinyllm.analysis.weight_stats import compute_weight_stats

            t0 = time.perf_counter()
            wstats = compute_weight_stats(model)
            timing_log["time/weight_stats"] = time.perf_counter() - t0
            for name, stats in wstats.items():
                if name == "_time":
                    continue
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"weights/{name}/{k}"] = v

        # 2. Activation stats
        if self.track_activations and self.activation_captures:
            from src.tinyllm.analysis.activation_stats import (
                compute_activation_stats,
            )

            t0 = time.perf_counter()
            captures = {
                name: cap.captured
                for name, cap in self.activation_captures.items()
                if cap.captured
            }
            if captures:
                astats = compute_activation_stats(captures)
                timing_log["time/activation_stats"] = time.perf_counter() - t0
                for k, v in astats.items():
                    if k == "_time":
                        continue
                    log_dict[f"activations/{k}"] = v

        # 3. Spectral analysis
        if self.track_spectral:
            from src.tinyllm.analysis.spectral import compute_svd_and_variance

            t0 = time.perf_counter()
            for name, p in model.named_parameters():
                if p.ndim >= 2:
                    try:
                        result = compute_svd_and_variance(
                            p, thresholds=self.variance_thresholds
                        )
                        prefix = f"spectral/{name}"
                        for k, v in result.items():
                            if k in (
                                "singular_values",
                                "eigenvalues",
                                "cumulative_variance_ratio",
                            ):
                                continue
                            log_dict[f"{prefix}/{k}"] = v
                    except Exception as e:
                        logger.warning(
                            f"Spectral analysis failed for {name}: {e}"
                        )
            timing_log["time/spectral"] = time.perf_counter() - t0

        # 4. Timing
        log_dict.update(timing_log)
        logger.info(
            f"Analysis at step {global_step}: "
            + ", ".join(f"{k}={v:.3f}s" for k, v in timing_log.items())
        )

        if log_dict:
            try:
                import wandb

                wandb.log(log_dict, step=global_step)
            except ImportError:
                pass

    def on_train_end(self, **kwargs):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
