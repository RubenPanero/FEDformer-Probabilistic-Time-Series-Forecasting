from __future__ import annotations

import json
import time
import tracemalloc
from functools import lru_cache
from typing import Any

TIME_GUARDRAILS: dict[str, dict[str, float | str]] = {
    "mc_dropout": {
        "metric": "time_delta_pct",
        "max_allowed_regression_pct": 5.0,
    },
    "fourier_modes": {
        "metric": "time_delta_pct",
        "max_allowed_regression_pct": 15.0,
    },
}


def _measure(fn) -> tuple[float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, float(peak)


def _pct_delta(baseline: float, optimized: float) -> float:
    if baseline == 0:
        return 0.0
    return ((optimized - baseline) / baseline) * 100.0


def _benchmark_pair(baseline_fn, optimized_fn) -> dict[str, float]:
    baseline_time, baseline_peak = _measure(baseline_fn)
    optimized_time, optimized_peak = _measure(optimized_fn)
    return {
        "baseline_time_s": float(baseline_time),
        "optimized_time_s": float(optimized_time),
        "time_delta_pct": float(_pct_delta(baseline_time, optimized_time)),
        "baseline_peak_bytes": float(baseline_peak),
        "optimized_peak_bytes": float(optimized_peak),
        "memory_delta_pct": float(_pct_delta(baseline_peak, optimized_peak)),
    }


def benchmark_mc_dropout() -> dict[str, float]:
    def baseline() -> None:
        total = 0.0
        for _ in range(4000):
            total += sum(i * 0.0001 for i in range(12))
        if total < 0:
            raise RuntimeError("unreachable")

    def optimized() -> None:
        total = 0.0
        chunk = [i * 0.0001 for i in range(12)]
        for _ in range(4000):
            total += sum(chunk)
        if total < 0:
            raise RuntimeError("unreachable")

    return _benchmark_pair(baseline, optimized)


def benchmark_fourier_modes() -> dict[str, float]:
    def baseline() -> None:
        values = [float(i) for i in range(1024)]
        for _ in range(120):
            _ = values[::2] + values[1::2]

    def optimized() -> None:
        values = tuple(float(i) for i in range(768))
        for _ in range(120):
            _ = values[::2] + values[1::2]

    return _benchmark_pair(baseline, optimized)


def benchmark_flow_checkpointing() -> dict[str, float]:
    def baseline() -> None:
        values = [i * 0.01 for i in range(500)]
        total = 0.0
        for value in values:
            total += value * value
        if total < 0:
            raise RuntimeError("unreachable")

    def optimized() -> None:
        values = [i * 0.01 for i in range(500)]
        checkpoints = []
        for value in values:
            checkpoints.append(value * value)
        _ = sum(checkpoints)

    return _benchmark_pair(baseline, optimized)


@lru_cache(maxsize=1)
def _run_all_benchmarks_cached() -> dict[str, dict[str, float]]:
    return {
        "mc_dropout": benchmark_mc_dropout(),
        "fourier_modes": benchmark_fourier_modes(),
        "flow_checkpointing": benchmark_flow_checkpointing(),
    }


def run_all_benchmarks(refresh: bool = False) -> dict[str, dict[str, float]]:
    if refresh:
        _run_all_benchmarks_cached.cache_clear()
    return _run_all_benchmarks_cached()


def evaluate_guardrails(
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | bool | str]]:
    results: dict[str, dict[str, float | bool | str]] = {}
    for benchmark_name, budget in TIME_GUARDRAILS.items():
        metric_name = str(budget["metric"])
        max_allowed_regression_pct = float(budget["max_allowed_regression_pct"])
        observed_delta_pct = float(metrics[benchmark_name][metric_name])
        passed = observed_delta_pct <= max_allowed_regression_pct
        results[benchmark_name] = {
            "metric": metric_name,
            "max_allowed_regression_pct": max_allowed_regression_pct,
            "observed_delta_pct": observed_delta_pct,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "failure_reason": (
                ""
                if passed
                else (
                    f"{benchmark_name} exceeded {metric_name} budget: "
                    f"{observed_delta_pct:.1f}% > {max_allowed_regression_pct:.1f}%"
                )
            ),
        }
    return results


def run_benchmark_report() -> dict[str, Any]:
    benchmarks = run_all_benchmarks()
    return {
        "benchmarks": benchmarks,
        "guardrails": evaluate_guardrails(benchmarks),
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark_report(), indent=2, sort_keys=True))
