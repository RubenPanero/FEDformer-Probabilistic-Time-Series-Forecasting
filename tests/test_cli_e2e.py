# -*- coding: utf-8 -*-
"""End-to-end CLI tests for existing FEDformer workflows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from config import FEDformerConfig
from data.preprocessing import PreprocessingPipeline
from main import _build_config_dict
from models.fedformer import Flow_FEDformer
from utils.model_registry import register_specialist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "NVDA_features.csv"


def _subprocess_env() -> dict[str, str]:
    """Build a stable environment for subprocess-based CLI tests."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["MPLBACKEND"] = "Agg"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    return env


def _write_tiny_training_csv(tmp_path: Path, rows: int = 180) -> Path:
    """Write a small real fixture slice for CLI training/inference runs."""
    csv_path = tmp_path / "NVDA_features.csv"
    pd.read_csv(FIXTURE_CSV).head(rows).to_csv(csv_path, index=False)
    return csv_path


def _run_main_cli(
    tmp_path: Path, csv_path: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run main.py from an isolated working directory."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--csv",
        str(csv_path),
        "--targets",
        "Close",
        "--seq-len",
        "24",
        "--label-len",
        "12",
        "--pred-len",
        "4",
        "--batch-size",
        "8",
        "--epochs",
        "1",
        "--splits",
        "2",
        "--preset",
        "debug",
        "--compile-mode",
        "none",
        "--mc-dropout-eval-samples",
        "2",
        "--save-results",
        "--no-show",
    ]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.run(
        cmd,
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=240,
    )


def _create_registered_specialist(tmp_path: Path) -> tuple[Path, Path]:
    """Create a real canonical specialist registry for inference CLI E2E."""
    csv_path = tmp_path / "NVDA_features.csv"
    df = pd.DataFrame(
        {
            "Close": [100.0 + i * 0.25 for i in range(200)],
            "Volume": [1000.0 + i * 5.0 for i in range(200)],
        }
    )
    df.to_csv(csv_path, index=False)

    config = FEDformerConfig(
        target_features=["Close"],
        file_path=str(csv_path),
        seq_len=20,
        label_len=10,
        pred_len=4,
        batch_size=8,
    )
    model = Flow_FEDformer(config)

    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    checkpoint_src = checkpoints_dir / "best_model_fold_3.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scaler_state_dict": None,
            "epoch": 3,
            "fold": 3,
            "loss": 0.5,
            "config": _build_config_dict(config),
        },
        checkpoint_src,
    )

    preprocessor = PreprocessingPipeline.from_config(
        config,
        target_features=["Close"],
        date_column=None,
    )
    preprocessor.fit(df)
    preprocessing_dir = checkpoints_dir / "nvda_preprocessing"
    preprocessor.save_artifacts(preprocessing_dir)

    registry_path = checkpoints_dir / "model_registry.json"
    register_specialist(
        ticker="NVDA",
        checkpoint_src=checkpoint_src,
        metrics={
            "sharpe": 1.0,
            "sortino": 1.2,
            "max_drawdown": -0.2,
            "volatility": 0.1,
        },
        config_dict=_build_config_dict(config),
        data_info={
            "file": str(csv_path),
            "rows": len(df),
            "features": len(df.columns),
            "preprocessing_artifacts": str(preprocessing_dir),
        },
        registry_path=registry_path,
        canonical_dir=checkpoints_dir,
    )
    return registry_path, csv_path


def _write_main_save_canonical_wrapper(wrapper_path: Path) -> None:
    """Create a wrapper that forces a stable positive Sharpe registration path."""
    wrapper_path.write_text(
        """
from __future__ import annotations

import sys
from unittest.mock import patch

import main as fed_main


def _patched_metrics(self, strategy_returns):
    metrics = _original(self, strategy_returns)
    metrics["sharpe_ratio"] = 1.25
    metrics["sortino_ratio"] = 1.5
    return metrics


_original = fed_main.PortfolioSimulator.calculate_metrics

with patch.object(fed_main.PortfolioSimulator, "calculate_metrics", _patched_metrics):
    sys.argv[0] = "main.py"
    fed_main.main()
""".strip(),
        encoding="utf-8",
    )


def _write_tune_hyperparams_wrapper(wrapper_path: Path) -> None:
    """Create a wrapper that stabilizes the expensive training subprocess."""
    wrapper_path.write_text(
        """
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import tune_hyperparams as th


def _fake_run(cmd, **kwargs):
    del kwargs
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time() + 1))
    pd.DataFrame(
        {
            "metric": ["sharpe_ratio", "sortino_ratio", "max_drawdown", "volatility"],
            "value": [0.91, 1.11, -0.12, 0.08],
        }
    ).to_csv(results_dir / f"portfolio_metrics_{ts}.csv", index=False)
    pd.DataFrame(
        {
            "step": [0, 1],
            "target_idx": [0, 0],
            "var_95": [0.03, 0.03],
            "cvar_95": [0.05, 0.05],
        }
    ).to_csv(results_dir / f"risk_metrics_{ts}.csv", index=False)
    return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")


with patch("subprocess.run", side_effect=_fake_run):
    th.main()
""".strip(),
        encoding="utf-8",
    )


def _write_verify_cp_full_wrapper(wrapper_path: Path, workdir: Path) -> None:
    """Create a wrapper that preserves the script launch path with safe main flags."""
    wrapper_path.write_text(
        f"""
from __future__ import annotations

from pathlib import Path

import scripts.verify_cp_walkforward as cpwf

cpwf.PROJECT_DIR = Path(r"{workdir}")


def _patched_check_flag_available():
    return None


def _patched_build_training_cmd(args):
    return [
        args.python_cmd,
        r"{PROJECT_ROOT / "main.py"}",
        "--csv",
        args.csv,
        "--targets",
        args.targets,
        "--seq-len",
        str(args.seq_len),
        "--label-len",
        "12",
        "--pred-len",
        str(args.pred_len),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        "1",
        "--splits",
        str(args.splits),
        "--preset",
        "debug",
        "--return-transform",
        args.return_transform,
        "--metric-space",
        args.metric_space,
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--seed",
        str(args.seed),
        "--compile-mode",
        args.compile_mode or "none",
        "--mc-dropout-eval-samples",
        "2",
        "--cp-walkforward",
        "--save-results",
        "--no-show",
    ]


cpwf.check_flag_available = _patched_check_flag_available
cpwf.build_training_cmd = _patched_build_training_cmd
raise SystemExit(cpwf.main())
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.slow
def test_main_cli_end_to_end_generates_results_and_manifest(tmp_path: Path) -> None:
    """main.py should train on a small real fixture slice and export run artifacts."""
    csv_path = _write_tiny_training_csv(tmp_path)

    result = _run_main_cli(tmp_path, csv_path)

    assert result.returncode == 0, result.stderr
    results_dir = tmp_path / "results"
    assert list(results_dir.glob("predictions_*.csv"))
    assert list(results_dir.glob("portfolio_metrics_*.csv"))
    assert list(results_dir.glob("risk_metrics_*.csv"))

    manifest_paths = list(results_dir.glob("run_manifest_*.json"))
    assert manifest_paths

    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert "metrics" in manifest
    assert "sharpe_ratio" in manifest["metrics"]

    checkpoints_dir = tmp_path / "checkpoints"
    assert (checkpoints_dir / "best_model_fold_1.pt").exists()


@pytest.mark.slow
def test_inference_cli_end_to_end_exports_predictions(tmp_path: Path) -> None:
    """python -m inference should load a registered specialist and export quantiles."""
    registry_path, csv_path = _create_registered_specialist(tmp_path)
    output_path = tmp_path / "results" / "inference_nvda.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference",
            "--ticker",
            "NVDA",
            "--csv",
            str(csv_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--n-samples",
            "5",
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    exported = pd.read_csv(output_path)
    assert not exported.empty
    assert {"p10_Close", "p50_Close", "p90_Close", "gt_Close"}.issubset(
        exported.columns
    )


@pytest.mark.slow
def test_inference_cli_with_plot_generates_visual_artifacts(tmp_path: Path) -> None:
    """python -m inference --plot should generate fan-chart and calibration PNGs."""
    registry_path, csv_path = _create_registered_specialist(tmp_path)
    output_path = tmp_path / "results" / "inference_nvda.csv"
    output_dir = tmp_path / "plots"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference",
            "--ticker",
            "NVDA",
            "--csv",
            str(csv_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--output-dir",
            str(output_dir),
            "--n-samples",
            "5",
            "--plot",
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert (output_dir / "fan_chart_nvda.png").exists()
    assert (output_dir / "calibration_nvda.png").exists()
    assert "Fan chart:" in result.stdout
    assert "Calibration:" in result.stdout


@pytest.mark.slow
def test_verify_cp_walkforward_full_mode_runs_training_and_reports_manifest(
    tmp_path: Path,
) -> None:
    """verify_cp_walkforward.py should launch training and report on the final manifest."""
    csv_path = _write_tiny_training_csv(tmp_path)
    wrapper_path = tmp_path / "run_verify_cp_full.py"
    _write_verify_cp_full_wrapper(wrapper_path, tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(wrapper_path),
            "--csv",
            str(csv_path),
            "--seq-len",
            "24",
            "--pred-len",
            "4",
            "--batch-size",
            "8",
            "--splits",
            "2",
            "--compile-mode",
            "none",
            "--results-dir",
            str(tmp_path / "results"),
            "--python-cmd",
            sys.executable,
            "--objective-threshold",
            "1.0",
            "--skip-gpu-info",
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2
    assert "=== Run --cp-walkforward ===" in result.stdout
    assert "cp_wf_coverage_80" in result.stdout
    assert list((tmp_path / "results").glob("run_manifest_*.json"))


@pytest.mark.slow
def test_main_save_canonical_positive_path_generates_registry_and_inference_uses_it(
    tmp_path: Path,
) -> None:
    """A controlled wrapper should drive --save-canonical to a stable positive path."""
    csv_path = _write_tiny_training_csv(tmp_path)
    wrapper_path = tmp_path / "run_main_save_canonical.py"
    _write_main_save_canonical_wrapper(wrapper_path)

    train_result = subprocess.run(
        [
            sys.executable,
            str(wrapper_path),
            "--csv",
            str(csv_path),
            "--targets",
            "Close",
            "--seq-len",
            "24",
            "--label-len",
            "12",
            "--pred-len",
            "4",
            "--batch-size",
            "8",
            "--epochs",
            "1",
            "--splits",
            "2",
            "--preset",
            "debug",
            "--compile-mode",
            "none",
            "--mc-dropout-eval-samples",
            "2",
            "--save-results",
            "--save-canonical",
            "--no-show",
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert train_result.returncode == 0, train_result.stderr

    registry_path = tmp_path / "checkpoints" / "model_registry.json"
    canonical_ckpt = tmp_path / "checkpoints" / "nvda_canonical.pt"
    preprocessing_dir = tmp_path / "checkpoints" / "nvda_preprocessing"

    assert registry_path.exists()
    assert canonical_ckpt.exists()
    assert preprocessing_dir.exists()

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert "NVDA" in registry["specialists"]

    inference_output = tmp_path / "results" / "canonical_inference.csv"
    inference_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference",
            "--ticker",
            "NVDA",
            "--csv",
            str(csv_path),
            "--registry",
            str(registry_path),
            "--output",
            str(inference_output),
            "--n-samples",
            "5",
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert inference_result.returncode == 0, inference_result.stderr
    assert inference_output.exists()
    exported = pd.read_csv(inference_output)
    assert not exported.empty


@pytest.mark.slow
def test_tune_hyperparams_minimal_cli_e2e_creates_results_and_study(
    tmp_path: Path,
) -> None:
    """A wrapped minimal tune_hyperparams CLI run should create results and storage."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "MSFT_features.csv"
    pd.read_csv(FIXTURE_CSV).head(180).to_csv(csv_path, index=False)

    wrapper_path = tmp_path / "run_tune_hyperparams_e2e.py"
    _write_tune_hyperparams_wrapper(wrapper_path)

    study_path = tmp_path / "optuna_studies" / "e2e.db"
    result = subprocess.run(
        [
            sys.executable,
            str(wrapper_path),
            "--csv",
            str(csv_path),
            "--n-trials",
            "1",
            "--n-splits",
            "2",
            "--storage-path",
            str(study_path),
            "--clean-results",
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert study_path.exists()
    assert list((tmp_path / "results").glob("portfolio_metrics_*.csv"))
    assert list((tmp_path / "results").glob("risk_metrics_*.csv"))


@pytest.mark.slow
def test_verify_cp_walkforward_report_only_reports_real_manifest_status(
    tmp_path: Path,
) -> None:
    """verify_cp_walkforward.py should report the CP status from a real run manifest."""
    csv_path = _write_tiny_training_csv(tmp_path)
    train_result = _run_main_cli(tmp_path, csv_path, extra_args=["--cp-walkforward"])

    assert train_result.returncode == 0, train_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "verify_cp_walkforward.py"),
            "--report-only",
            "--results-dir",
            str(tmp_path / "results"),
            "--objective-threshold",
            "0.0",
            "--skip-gpu-info",
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "cp_wf_coverage_80" in result.stdout
    assert "BAJO OBJETIVO" in result.stdout
