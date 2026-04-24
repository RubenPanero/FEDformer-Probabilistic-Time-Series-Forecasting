from __future__ import annotations

import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_MODULE_PATH = PROJECT_ROOT / "docs" / "critical_bottlenecks_benchmark.py"

HOTSPOTS = [
    {
        "id": "mc_dropout",
        "description": "trainer stochastic evaluation path",
    },
    {
        "id": "fourier_attention",
        "description": "frequency attention path",
    },
    {
        "id": "flow_log_prob",
        "description": "normalizing-flow loss path",
    },
    {
        "id": "dataset_fit_transform",
        "description": "dataset preprocessing fit/transform path",
    },
    {
        "id": "optuna_subprocess",
        "description": "hyperparameter subprocess overhead",
    },
]


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "critical_bottlenecks_benchmark",
        BENCHMARK_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module: {BENCHMARK_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_baseline_report(label: str) -> dict:
    benchmark_module = _load_benchmark_module()
    return {
        "label": label,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "hotspots": HOTSPOTS,
        "benchmarks": benchmark_module.run_all_benchmarks(refresh=True),
    }


def write_baseline_report(output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}.json"
    payload = build_baseline_report(label=label)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path


if __name__ == "__main__":
    default_output_dir = PROJECT_ROOT / "docs" / "baselines"
    output_path = write_baseline_report(default_output_dir, "manual")
    print(output_path)
