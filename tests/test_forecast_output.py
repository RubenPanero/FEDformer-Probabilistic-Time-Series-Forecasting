# -*- coding: utf-8 -*-
"""Tests para ForecastOutput: dual-space predictions y backward compat."""

import numpy as np
from training.forecast_output import ForecastOutput
from simulations import RiskSimulator, PortfolioSimulator


def _make_forecast(
    metric_space: str = "returns", return_transform: str = "none"
) -> ForecastOutput:
    """Factory de ForecastOutput para tests."""
    n_windows, pred_len, n_targets = 10, 5, 1
    n_samples = 50
    rng = np.random.default_rng(42)
    preds_scaled = rng.normal(0, 1, (n_windows, pred_len, n_targets))
    gt_scaled = rng.normal(0, 1, (n_windows, pred_len, n_targets))
    samples_scaled = rng.normal(0, 1, (n_samples, n_windows, pred_len, n_targets))
    # Espacio real: versión escalada x 100 (simulando inverse_transform)
    preds_real = preds_scaled * 100
    gt_real = gt_scaled * 100
    samples_real = samples_scaled * 100
    return ForecastOutput(
        preds_scaled=preds_scaled,
        gt_scaled=gt_scaled,
        samples_scaled=samples_scaled,
        preds_real=preds_real,
        gt_real=gt_real,
        samples_real=samples_real,
        metric_space=metric_space,
        return_transform=return_transform,
        target_names=["Close"],
    )


def test_forecast_output_returns_space():
    """Con metric_space='returns', *_for_metrics devuelve los arrays escalados."""
    fo = _make_forecast(metric_space="returns")
    assert np.array_equal(fo.preds_for_metrics, fo.preds_scaled)
    assert np.array_equal(fo.gt_for_metrics, fo.gt_scaled)
    assert np.array_equal(fo.samples_for_metrics, fo.samples_scaled)


def test_forecast_output_prices_space():
    """Con metric_space='prices', *_for_metrics devuelve los arrays reales."""
    fo = _make_forecast(metric_space="prices", return_transform="log_return")
    assert np.array_equal(fo.preds_for_metrics, fo.preds_real)
    assert np.array_equal(fo.gt_for_metrics, fo.gt_real)
    assert np.array_equal(fo.samples_for_metrics, fo.samples_real)


def test_risk_simulator_backward_compat():
    """RiskSimulator acepta np.ndarray directamente (backward compat)."""
    samples = np.random.randn(100, 10, 1)
    risk = RiskSimulator(samples)
    var = risk.calculate_var()
    assert var.shape == (10, 1)


def test_portfolio_simulator_backward_compat():
    """PortfolioSimulator acepta np.ndarray directamente (backward compat)."""
    preds = np.random.randn(20, 10, 1)
    gt = np.random.randn(20, 10, 1)
    ps = PortfolioSimulator(preds, gt)
    returns = ps.run_simple_strategy()
    assert returns.shape[0] == 19


def test_inverse_transform_noop_when_no_return_transform():
    """Con return_transform='none', preds_real y preds_scaled no son iguales
    (son los valores que el trainer calculó), pero el dataclass los almacena correctamente."""
    fo = _make_forecast(metric_space="returns", return_transform="none")
    # Con metric_space='returns', preds_for_metrics == preds_scaled
    assert np.array_equal(fo.preds_for_metrics, fo.preds_scaled)
    # preds_real existe y es diferente de preds_scaled (x100 en nuestro mock)
    assert not np.array_equal(fo.preds_real, fo.preds_scaled)


def test_inverse_transform_log_return_produces_positive_prices():
    """Con return_transform='log_return' y metric_space='prices', samples_for_metrics
    devuelve samples_real (que debe ser positivo si representan precios)."""
    rng = np.random.default_rng(0)
    n_windows, pred_len, n_targets = 8, 4, 1
    n_samples = 20
    # Retornos pequeños positivos (simula log-returns)
    returns = rng.normal(0.001, 0.01, (n_samples, n_windows, pred_len, n_targets))
    # Precios reconstruidos (siempre positivos si last_price > 0)
    last_price = 150.0
    prices = last_price * np.exp(np.cumsum(returns, axis=-2))

    fo = ForecastOutput(
        preds_scaled=returns[:1, :, :, :].mean(axis=0),  # (8,4,1)
        gt_scaled=returns[:1, :, :, :].mean(axis=0),
        samples_scaled=returns,
        preds_real=prices[:1, :, :, :].mean(axis=0),
        gt_real=prices[:1, :, :, :].mean(axis=0),
        samples_real=prices,
        metric_space="prices",
        return_transform="log_return",
        target_names=["Close"],
    )

    assert fo.samples_for_metrics.shape == prices.shape
    assert np.all(fo.samples_for_metrics > 0), (
        "Los precios reconstruidos deben ser positivos"
    )
