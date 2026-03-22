#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

# Directorios de producción donde aplica pylint --errors-only
_PRODUCTION_DIRS = ("models/", "training/", "data/", "utils/", "inference/")

# Mapa de módulo editado → test relacionado (runner rápido tras edición)
_MODULE_TEST_MAP = {
    "training/trainer.py": "tests/test_trainer_scheduling.py",
    "training/forecast_output.py": "tests/test_forecast_output.py",
    "utils/probabilistic_metrics.py": "tests/test_probabilistic_metrics.py",
    "utils/io_experiment.py": "tests/test_experiment_outputs.py",
    "models/flows.py": "tests/test_flows.py",
    "data/preprocessing.py": "tests/test_preprocessing_pipeline.py",
    "data/dataset.py": "tests/test_dataset.py",
    "config.py": "tests/test_trainer_scheduling.py",
    "utils/experiment_registry.py": "tests/test_experiment_registry.py",
    "scripts/run_ablation_matrix.py": "tests/test_ablation_matrix.py",
    "scripts/run_multi_seed.py": "tests/test_seed_aggregation.py",
    "utils/calibration.py": "tests/test_calibration.py",
    "utils/visualization.py": "tests/test_visualization.py",
    "inference/loader.py": "tests/test_inference.py",
    "inference/predictor.py": "tests/test_inference.py",
    "inference/__main__.py": "tests/test_inference.py",
    "main.py": "tests/test_finetune.py",
    "tune_hyperparams.py": "tests/test_tune_hyperparams.py",
    "utils/helpers.py": "tests/test_trainer_scheduling.py",
}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = str(tool_input.get("file_path") or "")

    if not file_path.endswith(".py"):
        return 0

    commands = [
        ["ruff", "format", file_path],
        ["ruff", "check", "--fix", file_path],
    ]

    # Añadir pylint solo para módulos de producción (no tests ni scripts)
    normalized = file_path.replace("\\", "/")
    is_production = any(
        f"/{d}" in normalized or normalized.startswith(d) for d in _PRODUCTION_DIRS
    )
    if is_production:
        commands.append(["pylint", "--errors-only", file_path])

    for command in commands:
        try:
            subprocess.run(command, check=False)
        except FileNotFoundError:
            return 0

    # Ejecutar test relacionado si el módulo tiene uno mapeado y el archivo de test existe
    for module_suffix, test_file in _MODULE_TEST_MAP.items():
        if normalized.endswith(module_suffix) and Path(test_file).exists():
            try:
                subprocess.run(
                    [
                        "python3",
                        "-m",
                        "pytest",
                        test_file,
                        "-q",
                        "--tb=short",
                        "--no-header",
                        "--disable-warnings",
                    ],
                    check=False,
                )
            except FileNotFoundError:
                pass
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
