import time

from src.tinyllm.callbacks.base_callback import BaseCallback
from src.tinyllm.logger.logger_utils import logger


class BenchmarkCallback(BaseCallback):
    """
    Runs downstream benchmarks at configurable intervals.
    """

    def __init__(self, benchmark_config: dict):
        super().__init__()
        self.benchmark_config = benchmark_config
        self._benchmark_intervals = {}
        for name, cfg in benchmark_config.items():
            if cfg.get("enabled", False):
                interval = cfg.get("every_n_steps", 1000)
                self._benchmark_intervals[name] = interval

    def on_train_step_end(self, **kwargs):
        global_step = kwargs.get("global_step", 0)
        if global_step == 0:
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        device = kwargs.get("device")
        if model is None or tokenizer is None or device is None:
            return

        run_any = False
        for name, interval in self._benchmark_intervals.items():
            if global_step % interval == 0:
                run_any = True
                break

        if not run_any:
            return

        from src.tinyllm.benchmarks.runner import run_benchmarks

        t0 = time.perf_counter()
        results = run_benchmarks(
            self.benchmark_config, model, tokenizer, device
        )
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
