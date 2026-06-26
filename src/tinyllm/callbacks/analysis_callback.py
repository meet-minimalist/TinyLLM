import time

import torch

from src.tinyllm.callbacks.base_callback import BaseCallback
from src.tinyllm.logger.logger_utils import logger


class GradientCapture:
    """Captures per-parameter gradient statistics during backward via hooks."""

    def __init__(self, model, num_bins: int = 32):
        self.num_bins = num_bins
        self.stats = {}  # name -> {norm, mean, std, abs_max, _hist}
        self._handles = []

        for name, p in model.named_parameters():
            if p.requires_grad:
                self._handles.append(p.register_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(grad):
            import numpy as np

            g = grad.detach().float().flatten()
            self.stats[name] = {
                "norm": g.norm().item(),
                "_hist": np.histogram(g.cpu().numpy(), bins=self.num_bins),
            }

        return hook

    def compute_stats(self):
        if not self.stats:
            return {}

        total_sq = 0.0
        n_zero = 0
        n_exploded = 0

        for s in self.stats.values():
            total_sq += s["norm"] ** 2
            if s["norm"] == 0:
                n_zero += 1
            if s["norm"] > 1e4:
                n_exploded += 1

        return {
            "global/gradient_norm": total_sq**0.5,
            "global/exploded_gradients": n_exploded,
            "global/zero_gradients": n_zero,
            "global/layers_with_grad": len(self.stats),
            "_layer_stats": self.stats,
        }

    def clear(self):
        self.stats = {}

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
        self.track_layer_cosim = config.get("track_layer_cosim", False)
        self.hist_bins = config.get("hist_bins", 32)
        self._hooks = []
        self._grad_capture = None
        self._attn_capture = None  # ForwardHookCapture for attention metadata
        self._layer_capture = None  # ForwardHookCapture for block outputs
        self._layer_cosim_history = (
            []
        )  # [(step, [cosim_per_pair])], capped at cosim_window

    def on_train_begin(self, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        if self.track_gradients:
            self._grad_capture = GradientCapture(model, num_bins=self.hist_bins)
        if self.track_activations:
            self._register_activation_hooks(model)
        if self.track_layer_cosim:
            from src.tinyllm.analysis.activation_stats import ForwardHookCapture

            self._layer_capture = ForwardHookCapture()
            for name, module in model.named_modules():
                if module.__class__.__name__ == "TransformerBlock":
                    self._hooks.append(
                        module.register_forward_hook(
                            self._layer_capture.make_hook(name)
                        )
                    )

    def _register_activation_hooks(self, model):
        from src.tinyllm.analysis.activation_stats import ForwardHookCapture

        self._attn_capture = ForwardHookCapture()
        for name, module in model.named_modules():
            if name.endswith(".attn") or name == "attn":
                self._hooks.append(
                    module.register_forward_hook(
                        self._attn_capture.make_hook(name, unpack_metadata=True)
                    )
                )

    def on_train_step_end(self, **kwargs):
        global_step = kwargs.get("global_step", 0)
        if global_step == 0 or global_step % self.every_n_steps != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        log_dict = {}
        timing_log = {}

        # 1. Gradient global stats + histograms
        if self._grad_capture is not None:
            t0 = time.perf_counter()
            gstats = self._grad_capture.compute_stats()
            timing_log["time/gradient_stats"] = time.perf_counter() - t0
            for k, v in gstats.items():
                if k != "_layer_stats":
                    log_dict[f"gradients/{k}"] = v
            self._grad_capture.clear()

        # 2. Weight histograms
        if self.track_weights:
            from src.tinyllm.analysis.weight_stats import compute_weight_stats

            t0 = time.perf_counter()
            wstats = compute_weight_stats(model, num_bins=self.hist_bins)
            timing_log["time/weight_stats"] = time.perf_counter() - t0

        # 3. Activation histograms
        activation_hists = {}
        if self.track_activations and self._attn_capture is not None:
            import numpy as np

            t0 = time.perf_counter()
            for layer_name, tensors in self._attn_capture.captured.items():
                for tensor_name, tensor in tensors.items():
                    key = f"{layer_name}/{tensor_name}"
                    activation_hists[key] = np.histogram(
                        tensor.cpu().float().flatten().numpy(),
                        bins=self.hist_bins,
                    )
            self._attn_capture.clear()
            timing_log["time/activation_stats"] = time.perf_counter() - t0

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

        # 5. Layer-wise cosine similarity profile
        if self.track_layer_cosim and self._layer_capture:
            from src.tinyllm.analysis.activation_stats import (
                compute_layer_cosine_similarities,
            )

            if not self._layer_capture.captured:
                logger.warning(
                    "layer_cosim: no TransformerBlock outputs captured — "
                    "hooks may not be firing (torch.compile can suppress hooks)."
                )
            sims = compute_layer_cosine_similarities(
                self._layer_capture.captured
            )
            self._layer_cosim_history.append((global_step, sims))
            self._layer_capture.clear()

            for i, s in enumerate(sims):
                log_dict[f"layer_cosim/layer_{i}_to_{i + 1}"] = s

        # 6. Timing
        log_dict.update(timing_log)
        logger.info(
            f"Analysis at step {global_step}: "
            + ", ".join(f"{k}={v:.3f}s" for k, v in timing_log.items())
        )

        if log_dict:
            try:
                import wandb

                if self._grad_capture is not None:
                    for name, s in gstats.get("_layer_stats", {}).items():
                        log_dict[f"gradients/hist/{name}"] = wandb.Histogram(
                            np_histogram=s["_hist"]
                        )
                if self.track_weights:
                    for name, s in wstats.items():
                        if isinstance(s, dict) and "_hist" in s:
                            log_dict[f"weights/hist/{name}"] = wandb.Histogram(
                                np_histogram=s["_hist"]
                            )
                for key, hist in activation_hists.items():
                    log_dict[f"activations/hist/{key}"] = wandb.Histogram(
                        np_histogram=hist
                    )
                if self.track_layer_cosim and self._layer_cosim_history:
                    _n = len(self._layer_cosim_history[0][1])
                    _cols = ["step"] + [
                        f"layer_{i}_to_{i+1}" for i in range(_n)
                    ]
                    _rows = [
                        [st] + list(s) for st, s in self._layer_cosim_history
                    ]
                    log_dict["layer_cosim/table"] = wandb.Table(
                        data=_rows, columns=_cols
                    )
                wandb.log(log_dict, step=global_step)
            except ImportError:
                pass

    def on_train_end(self, **kwargs):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._grad_capture is not None:
            self._grad_capture.remove()
