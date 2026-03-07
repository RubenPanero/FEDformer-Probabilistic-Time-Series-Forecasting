# -*- coding: utf-8 -*-
"""
Tests unitarios para tune_hyperparams.py:
  - Poda de trials con seq_len < pred_len*3
  - Penalizaciones por VaR > 0.08 y Sortino < 0
  - Que --save-canonical no se incluye en el cmd de trials
  - Parsing de CSV de portafolio y riesgo
  - download_extra_tickers omite los que ya existen
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pandas as pd
import pytest

import tune_hyperparams as th


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_portfolio_csv(
    tmp_path: Path,
    ticker_stem: str,
    sharpe: float,
    sortino: float,
    include_ticker_suffix: bool = False,
) -> Path:
    """Escribe un CSV de portafolio mínimo para tests."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{ticker_stem}" if include_ticker_suffix else ""
    p = results_dir / f"portfolio_metrics_{ts}{suffix}.csv"
    pd.DataFrame(
        {
            "metric": ["sharpe_ratio", "sortino_ratio", "max_drawdown", "volatility"],
            "value": [sharpe, sortino, -0.4, 0.02],
        }
    ).to_csv(p, index=False)
    return p


def _write_risk_csv(
    tmp_path: Path,
    ticker_stem: str,
    var_95: float,
    include_ticker_suffix: bool = False,
) -> Path:
    """Escribe un CSV de riesgo mínimo para tests."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{ticker_stem}" if include_ticker_suffix else ""
    p = results_dir / f"risk_metrics_{ts}{suffix}.csv"
    pd.DataFrame(
        {
            "step": [0, 1],
            "target_idx": [0, 0],
            "var_95": [var_95, var_95],
            "cvar_95": [var_95 * 1.3, var_95 * 1.3],
        }
    ).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests de _parse_portfolio_csv y _parse_risk_csv
# ---------------------------------------------------------------------------


def test_parse_portfolio_csv_returns_correct_sharpe(tmp_path: Path) -> None:
    """_parse_portfolio_csv extrae el Sharpe correcto del formato real de main.py."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.75, sortino=1.2)

    result = th._parse_portfolio_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result["sharpe"] - 0.75) < 1e-6
    assert abs(result["sortino"] - 1.2) < 1e-6


def test_parse_portfolio_csv_accepts_legacy_ticker_suffix(tmp_path: Path) -> None:
    """_parse_portfolio_csv sigue aceptando el formato legado con ticker."""
    ts_before = time.time() - 1
    _write_portfolio_csv(
        tmp_path,
        "MSFT_features",
        sharpe=0.55,
        sortino=0.8,
        include_ticker_suffix=True,
    )

    result = th._parse_portfolio_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result["sharpe"] - 0.55) < 1e-6
    assert abs(result["sortino"] - 0.8) < 1e-6


def test_parse_portfolio_csv_returns_sentinel_when_missing(tmp_path: Path) -> None:
    """_parse_portfolio_csv retorna Sharpe=-1.0 si no hay CSV para el ticker."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = th._parse_portfolio_csv(results_dir, "UNKNOWN_features", time.time())
    assert result["sharpe"] == -1.0


def test_parse_risk_csv_returns_mean_var(tmp_path: Path) -> None:
    """_parse_risk_csv retorna el VaR medio del formato real de main.py."""
    ts_before = time.time() - 1
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.06)
    result = th._parse_risk_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result - 0.06) < 1e-6


def test_parse_risk_csv_accepts_legacy_ticker_suffix(tmp_path: Path) -> None:
    """_parse_risk_csv sigue aceptando el formato legado con ticker."""
    ts_before = time.time() - 1
    _write_risk_csv(
        tmp_path,
        "MSFT_features",
        var_95=0.04,
        include_ticker_suffix=True,
    )
    result = th._parse_risk_csv(tmp_path / "results", "MSFT_features", ts_before)
    assert abs(result - 0.04) < 1e-6


# ---------------------------------------------------------------------------
# Tests de la función objective
# ---------------------------------------------------------------------------


def _mock_trial(
    seq_len: int, pred_len: int, batch_size: int = 64, clip: float = 0.5
) -> MagicMock:
    """Construye un trial mock con suggest_categorical predefinido.

    El orden de llamadas en objective es: seq_len, pred_len, batch_size, gradient_clip_norm.
    """
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_categorical.side_effect = [seq_len, pred_len, batch_size, clip]
    trial.set_user_attr = MagicMock()
    return trial


def test_objective_prunes_when_seq_len_too_short() -> None:
    """objective lanza TrialPruned si seq_len < pred_len * 3."""
    # seq_len=48, pred_len=20 → 48 < 20*3=60 → debe podarse
    trial = _mock_trial(seq_len=48, pred_len=20)

    with pytest.raises(optuna.TrialPruned):
        th.objective(trial, "data/MSFT_features.csv", 4, Path("results"))


def test_objective_penalizes_high_var(tmp_path: Path) -> None:
    """objective aplica penalización ×0.5 cuando VaR_95 > 0.08."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.8, sortino=1.0)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.09)  # > 0.08

    trial = _mock_trial(seq_len=96, pred_len=20)
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        result = th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    # Sharpe base = 0.8; penalización VaR → 0.8 × 0.5 = 0.4
    assert abs(result - 0.4) < 1e-6


def test_objective_penalizes_negative_sortino(tmp_path: Path) -> None:
    """objective aplica penalización -0.3 cuando Sortino < 0."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.6, sortino=-0.1)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.05)  # VaR OK

    trial = _mock_trial(seq_len=96, pred_len=20)
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        result = th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    # Sharpe base = 0.6; penalización Sortino → 0.6 - 0.3 = 0.3
    assert abs(result - 0.3) < 1e-6


def test_objective_returns_minus_one_on_subprocess_failure(tmp_path: Path) -> None:
    """objective retorna -1.0 si el subproceso falla (returncode != 0)."""
    trial = _mock_trial(seq_len=96, pred_len=20)
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "error simulado"

    with patch("subprocess.run", return_value=mock_proc):
        result = th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    assert result == -1.0


def test_objective_cmd_excludes_save_canonical(tmp_path: Path) -> None:
    """Los trials nunca incluyen --save-canonical en el comando de entrenamiento."""
    ts_before = time.time() - 1
    _write_portfolio_csv(tmp_path, "MSFT_features", sharpe=0.7, sortino=1.0)
    _write_risk_csv(tmp_path, "MSFT_features", var_95=0.05)

    trial = _mock_trial(seq_len=96, pred_len=20)
    captured_cmd = []

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        captured_cmd.extend(cmd)
        m = MagicMock()
        m.returncode = 0
        return m

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("tune_hyperparams._current_time", return_value=ts_before),
    ):
        th.objective(trial, "data/MSFT_features.csv", 4, tmp_path / "results")

    assert "--save-canonical" not in captured_cmd


# ---------------------------------------------------------------------------
# Tests del resumen de trials
# ---------------------------------------------------------------------------


def test_build_completed_trials_dataframe_includes_state() -> None:
    """_build_completed_trials_dataframe filtra usando la columna state sin romper."""
    study = optuna.create_study(direction="maximize")

    def _objective(trial: optuna.Trial) -> float:
        trial.suggest_categorical("seq_len", [96])
        trial.set_user_attr("sortino", 1.1)
        return 0.42

    study.optimize(_objective, n_trials=1)

    completed_df = th._build_completed_trials_dataframe(study)

    assert not completed_df.empty
    assert completed_df.iloc[0]["state"] == "COMPLETE"
    assert completed_df.iloc[0]["value"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Tests de download_extra_tickers
# ---------------------------------------------------------------------------


def test_download_extra_tickers_skips_existing(tmp_path: Path, monkeypatch) -> None:
    """download_extra_tickers omite los tickers que ya tienen CSV en data/."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Crear CSV existente para AMD
    (data_dir / "AMD_features.csv").write_text("col\n1\n2\n")

    called_for = []

    def mock_run(cmd, **kwargs):  # noqa: ANN001
        # Detectar el ticker del comando
        ticker = None
        for i, arg in enumerate(cmd):
            if arg == "--symbol" and i + 1 < len(cmd):
                ticker = cmd[i + 1]
                called_for.append(ticker)
                break
        if ticker is not None:
            (data_dir / f"{ticker}_features.csv").write_text("date,Close\n2024-01-01,1\n")
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("subprocess.run", side_effect=mock_run):
        th.download_extra_tickers()

    # AMD ya existe → no debe llamarse
    assert "AMD" not in called_for
    # Los otros 7 sí deben haberse intentado
    for ticker in th.EXTRA_TICKERS:
        if ticker != "AMD":
            assert ticker in called_for


def test_download_extra_tickers_continues_when_csv_is_not_generated(
    tmp_path: Path, monkeypatch
) -> None:
    """download_extra_tickers no debe fallar si el builder devuelve 0 sin crear CSV."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        th.download_extra_tickers()
