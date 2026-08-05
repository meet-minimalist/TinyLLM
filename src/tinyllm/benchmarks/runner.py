import time

from src.tinyllm.benchmarks.arc import ARCEasyBenchmark
from src.tinyllm.benchmarks.hellaswag import HellaSwagBenchmark
from src.tinyllm.benchmarks.lambada import LambadaBenchmark
from src.tinyllm.benchmarks.piqa import PIQABenchmark
from src.tinyllm.benchmarks.winogrande import WinograndeBenchmark
from src.tinyllm.logger.logger_utils import logger

_BENCHMARK_MAP = {
    "hellaswag": HellaSwagBenchmark,
    "lambada": LambadaBenchmark,
    "winogrande": WinograndeBenchmark,
    "arc_easy": ARCEasyBenchmark,
    "piqa": PIQABenchmark,
}


def run_benchmarks(benchmark_config: dict, model, tokenizer, device):
    """
    Run all enabled benchmarks and return their results.

    The model is set to eval mode before running and restored to its original
    mode (train/eval) afterwards. Individual benchmarks must not call
    model.train() themselves.

    Returns:
        Dict of {benchmark_name: {metric: value}}.
    """
    results = {}
    t0 = time.perf_counter()

    was_training = model.training
    model.eval()

    for name, cfg in benchmark_config.items():
        if not cfg.get("enabled", False):
            continue
        if name not in _BENCHMARK_MAP:
            logger.warning(
                f"Unknown benchmark: '{name}'. Available: {list(_BENCHMARK_MAP)}"
            )
            continue

        cls = _BENCHMARK_MAP[name]
        benchmark = cls(
            num_samples=cfg.get("num_samples", 200),
            seed=cfg.get("seed", 42),
        )

        # A benchmark must never abort training — dataset outages, HF API
        # changes, etc. are logged and skipped so the run continues.
        try:
            bt0 = time.perf_counter()
            result = benchmark.run(model, tokenizer, device)
            elapsed = time.perf_counter() - bt0
        except Exception as e:
            logger.error(f"Benchmark '{name}' failed and was skipped: {e}")
            continue

        results[name] = result
        acc = result.get("accuracy", float("nan"))
        extra = (
            f", perplexity={result['perplexity']:.2f}"
            if "perplexity" in result
            else ""
        )
        logger.info(
            f"Benchmark {name}: accuracy={acc:.4f}{extra}, time={elapsed:.2f}s"
        )

    if was_training:
        model.train()

    results["_total_time"] = time.perf_counter() - t0
    return results
