import time

import torch

from src.tinyllm.callbacks.base_callback import BaseCallback
from src.tinyllm.logger.logger_utils import logger


class GradientCapture:
    """Captures per-parameter gradient norms during backward via hooks."""

    def __init__(self, model):
        self.norms = {}
        self._handles = []

        for name, p in model.named_parameters():
            if p.requires_grad:
                handle = p.register_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name):
        def hook(grad):
            self.norms[name] = grad.norm().item()

        return hook

    def compute_stats(self):
        if not self.norms:
            return {}

        total = 0.0
        n_zero = 0
        n_exploded = 0
        layer_norms = {}

        for name, norm_val in self.norms.items():
            layer_norms[name] = norm_val
            total += norm_val**2
            if norm_val == 0:
                n_zero += 1
            if norm_val > 1e4:
                n_exploded += 1

        total = total**0.5
        return {
            "global/gradient_norm": total,
            "global/exploded_gradients": n_exploded,
            "global/zero_gradients": n_zero,
            "global/layers_with_grad": len(self.norms),
            "_layer_norms": layer_norms,
        }

    def clear(self):
        self.norms = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class AnalysisCallback(BaseCallback):
    def __init__(self, config: dict):
        super().__init__()
        self.every_n_steps = config.get("every_n_steps", 500)
        self.track_weights = config.get("track_weights", True)
        self.track_gradients = config.get("track_gradients", True)
        self.track_activations = config.get("track_activations", False)
        self.track_spectral = config.get("spectral", False)
        self.variance_thresholds = config.get(
            "variance_thresholds", [0.95, 0.99]
        )
        self.activation_captures = {}
        self._hooks = []
        self._grad_capture = None

    def on_train_begin(self, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        if self.track_gradients:
            self._grad_capture = GradientCapture(model)
        if self.track_activations:
            self._register_activation_hooks(model)

    def _register_activation_hooks(self, model):
        from src.tinyllm.analysis.activation_stats import ActivationCapture

        for name, module in model.named_modules():
            # Match modules named exactly "attn" (not attn_norm, attn_norm_2, etc.)
            if name.endswith(".attn") or name == "attn":
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

        # 1. Gradient norms (captured during backward via hooks)
        if self._grad_capture is not None:
            t0 = time.perf_counter()
            gstats = self._grad_capture.compute_stats()
            timing_log["time/gradient_norms"] = time.perf_counter() - t0
            for k, v in gstats.items():
                if k in ("_layer_norms",):
                    continue
                log_dict[f"gradients/{k}"] = v
            for name, norm_val in gstats.get("_layer_norms", {}).items():
                log_dict[f"gradients/layer/{name}"] = norm_val
            self._grad_capture.clear()

        # 2. Weight stats
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

        # 3. Activation stats
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

        # 4. Spectral analysis
        if self.track_spectral:
            from src.tinyllm.analysis.spectral import (
                compute_svd_and_variance,
                compute_weightwatcher_alpha,
            )

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

                        alpha = compute_weightwatcher_alpha(p)
                        log_dict[f"{prefix}/alpha"] = alpha
                    except Exception as e:
                        logger.warning(
                            f"Spectral analysis failed for {name}: {e}"
                        )
            timing_log["time/spectral"] = time.perf_counter() - t0

        # 5. Timing
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
        if self._grad_capture is not None:
            self._grad_capture.remove()
