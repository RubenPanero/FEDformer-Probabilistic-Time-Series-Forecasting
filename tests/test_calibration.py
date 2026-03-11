# -*- coding: utf-8 -*-
"""Tests para utils/calibration.py y _apply_cp_calibration en main.py."""

import numpy as np
import pytest

from utils.calibration import apply_conformal_interval, conformal_quantile


# ---------------------------------------------------------------------------
# Tests de conformal_quantile y apply_conformal_interval
# ---------------------------------------------------------------------------


def test_conformal_quantile_basic() -> None:
    """conformal_quantile devuelve un float no negativo para datos simples."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    q = conformal_quantile(y_true, y_pred, alpha=0.2)
    assert isinstance(q, float)
    assert q >= 0.0


def test_conformal_quantile_invalid_alpha() -> None:
    """conformal_quantile lanza ValueError para alpha fuera de (0, 1)."""
    y = np.ones(5)
    with pytest.raises(ValueError):
        conformal_quantile(y, y, alpha=0.0)
    with pytest.raises(ValueError):
        conformal_quantile(y, y, alpha=1.0)


def test_conformal_quantile_shape_mismatch() -> None:
    """conformal_quantile lanza ValueError cuando las formas no coinciden."""
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), np.ones(6), alpha=0.1)


def test_apply_conformal_interval_shape() -> None:
    """apply_conformal_interval devuelve lower y upper con el mismo shape que y_pred."""
    y_pred = np.linspace(-1.0, 1.0, 20)
    lower, upper = apply_conformal_interval(y_pred, q_hat=0.5)
    assert lower.shape == y_pred.shape
    assert upper.shape == y_pred.shape
    assert np.all(upper >= lower)


# ---------------------------------------------------------------------------
# Test de _apply_cp_calibration (Enfoque 2 — global)
# ---------------------------------------------------------------------------


def test_apply_cp_calibration_returns_expected_coverage() -> None:
    """_apply_cp_calibration devuelve q_hat > 0 y cobertura en [0, 1]."""
    from training.forecast_output import ForecastOutput

    rng = np.random.default_rng(42)
    n_windows, pred_len, n_targets = 50, 5, 1
    preds = rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32)
    gt = (
        preds
        + rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32) * 0.3
    )

    forecast = ForecastOutput(
        preds_scaled=preds,
        gt_scaled=gt,
        preds_real=preds,
        gt_real=gt,
        samples_scaled=np.zeros((n_windows, pred_len, 10, n_targets)),
        samples_real=np.zeros((n_windows, pred_len, 10, n_targets)),
        target_names=["Close"],
        metric_space="returns",
        return_transform="log_return",
    )

    from main import _apply_cp_calibration

    result = _apply_cp_calibration(forecast, alpha=0.2)
    assert result["q_hat"] > 0
    assert 0.0 <= result["cp_coverage_80"] <= 1.0
    assert result["cp_lower"].shape == (n_windows * pred_len * n_targets,)
    assert result["cp_upper"].shape == (n_windows * pred_len * n_targets,)


def test_apply_cp_calibration_coverage_near_target() -> None:
    """Con residuos pequeños la cobertura 80% debe ser razonablemente alta."""
    from training.forecast_output import ForecastOutput

    rng = np.random.default_rng(0)
    n_windows, pred_len, n_targets = 200, 10, 1
    preds = rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32)
    # Residuos pequeños → intervalo ampliado ≥ 80% cobertura
    gt = (
        preds
        + rng.standard_normal((n_windows, pred_len, n_targets)).astype(np.float32) * 0.1
    )

    forecast = ForecastOutput(
        preds_scaled=preds,
        gt_scaled=gt,
        preds_real=preds,
        gt_real=gt,
        samples_scaled=np.zeros((n_windows, pred_len, 10, n_targets)),
        samples_real=np.zeros((n_windows, pred_len, 10, n_targets)),
        target_names=["Close"],
        metric_space="returns",
        return_transform="log_return",
    )

    from main import _apply_cp_calibration

    result = _apply_cp_calibration(forecast, alpha=0.2)
    # La cobertura empírica debe ser ≥ 0.75 con residuos pequeños y suficientes muestras
    assert result["cp_coverage_80"] >= 0.75
