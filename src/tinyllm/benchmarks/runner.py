import time

from src.tinyllm.benchmarks.hellaswag import HellaSwagBenchmark
from src.tinyllm.logger.logger_utils import logger

_BENCHMARK_MAP = {
    "hellaswag": HellaSwagBenchmark,
}


def run_benchmarks(benchmark_config: dict, model, tokenizer, device):
    """
    Run all enabled benchmarks.

    Args:
        benchmark_config: Dict from training config with per-benchmark settings.
        model: nn.Module.
        tokenizer: Tokenizer.
        device: torch device.

    Returns:
        Dict of {benchmark_name: {metric: value}}.
    """
    results = {}
    t0 = time.perf_counter()

    for name, cfg in benchmark_config.items():
        if not cfg.get("enabled", False):
            continue
        if name not in _BENCHMARK_MAP:
            logger.warning(f"Unknown benchmark: {name}")
            continue

        cls = _BENCHMARK_MAP[name]
        num_samples = cfg.get("num_samples", 200)
        benchmark = cls(num_samples=num_samples)

        bt0 = time.perf_counter()
        result = benchmark.run(model, tokenizer, device)
        elapsed = time.perf_counter() - bt0

        results[name] = result
        logger.info(
            f"Benchmark {name}: accuracy={result.get('accuracy', 0):.4f}, "
            f"time={elapsed:.2f}s"
        )

    results["_total_time"] = time.perf_counter() - t0
    return results
