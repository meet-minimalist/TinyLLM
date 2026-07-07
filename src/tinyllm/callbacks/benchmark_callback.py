import time

from src.tinyllm.callbacks.base_callback import BaseCallback
from src.tinyllm.logger.logger_utils import logger


class BenchmarkCallback(BaseCallback):
    """
    Runs downstream benchmarks at per-benchmark configurable intervals.

    Each benchmark in the config can have its own `every_n_steps`. Only the
    benchmarks due at the current step are passed to the runner, so fast
    benchmarks (e.g. LAMBADA every 5K) don't trigger slower ones (every 10K).
    """

    def __init__(self, benchmark_config: dict):
        super().__init__()
        self.benchmark_config = benchmark_config

    def on_train_step_end(self, **kwargs):
        global_step = kwargs.get("global_step", 0)
        if global_step == 0:
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        device = kwargs.get("device")
        if model is None or tokenizer is None or device is None:
            return

        # Build a sub-config containing only benchmarks due this step.
        due = {
            name: cfg
            for name, cfg in self.benchmark_config.items()
            if cfg.get("enabled", False)
            and global_step % cfg.get("every_n_steps", 1000) == 0
        }
        if not due:
            return

        from src.tinyllm.benchmarks.runner import run_benchmarks

        t0 = time.perf_counter()
        results = run_benchmarks(due, model, tokenizer, device)
        elapsed = time.perf_counter() - t0

        log_dict = {}
        for name, result in results.items():
            if name == "_total_time":
                continue
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    log_dict[f"benchmark/{name}/{k}"] = v

        log_dict["time/benchmarks"] = elapsed
        logger.info(
            f"Benchmarks at step {global_step}: total_time={elapsed:.2f}s"
        )

        if log_dict:
            try:
                import wandb

                wandb.log(log_dict, step=global_step)
            except ImportError:
                pass
