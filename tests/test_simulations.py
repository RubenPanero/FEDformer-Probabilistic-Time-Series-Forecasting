import pytest
import numpy as np
from simulations import RiskSimulator, PortfolioSimulator
from typing import Tuple


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # (n_samples, n_timesteps, n_assets)
    samples = np.random.randn(100, 50, 1)
    predictions = np.mean(samples, axis=0)
    ground_truth = predictions + np.random.randn(50, 1) * 0.1
    return samples, predictions, ground_truth


def test_risk_simulator(sample_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    samples, _, _ = sample_data
    risk_sim = RiskSimulator(samples)
    var = risk_sim.calculate_var()
    cvar = risk_sim.calculate_cvar()
    assert var.shape == (50, 1)
    assert cvar.shape == (50, 1)
    assert np.all(var >= 0)
    assert np.all(cvar >= 0)
    assert np.all(var <= cvar)


def test_portfolio_simulator(
    sample_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, predictions, ground_truth = sample_data
    portfolio_sim = PortfolioSimulator(predictions, ground_truth)
    strategy_returns = portfolio_sim.run_simple_strategy()
    metrics = portfolio_sim.calculate_metrics(strategy_returns)
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert "sortino_ratio" in metrics


def test_portfolio_uses_all_timesteps() -> None:
    """La señal debe usar el primer y último timestep predicho (no solo los 2 primeros)."""
    rng = np.random.default_rng(42)
    pred_len = 24
    n_samples = 10
    n_assets = 1

    # Predicciones con tendencia fuertemente positiva en toda la ventana
    predictions = np.linspace(1.0, 2.0, pred_len)[None, :, None].repeat(
        n_samples, axis=0
    )
    predictions += rng.normal(0, 0.01, predictions.shape)

    # Ground truth también creciente
    ground_truth = np.linspace(1.0, 2.0, pred_len)[None, :, None].repeat(
        n_samples, axis=0
    )
    ground_truth += rng.normal(0, 0.01, ground_truth.shape)

    sim = PortfolioSimulator(predictions, ground_truth)
    returns = sim.run_simple_strategy()

    # Con tendencia positiva clara, la mayoría de señales deben ser positivas
    assert returns.shape == (n_samples - 1, n_assets)
    assert returns.mean() > 0, (
        "Señal de momentum debe ser positiva con tendencia alcista"
    )


def test_portfolio_single_timestep_fallback() -> None:
    """Fallback para pred_len=1 no debe fallar."""
    rng = np.random.default_rng(0)
    predictions = rng.normal(0, 1, (20, 1, 1))
    ground_truth = rng.normal(0, 1, (20, 1, 1))
    sim = PortfolioSimulator(predictions, ground_truth)
    returns = sim.run_simple_strategy()
    assert returns.shape[0] == 19


def test_risk_simulator_accepts_forecast_output() -> None:
    """RiskSimulator acepta ForecastOutput y usa samples_for_metrics."""
    from training.forecast_output import ForecastOutput

    n_samples, n_windows, pred_len, n_targets = 100, 20, 5, 1
    rng = np.random.default_rng(7)
    samples_scaled = rng.normal(0, 1, (n_samples, n_windows, pred_len, n_targets))
    samples_real = samples_scaled * 50 + 100  # precios simulados positivos

    # metric_space="returns" → samples_for_metrics == samples_scaled
    fo_returns = ForecastOutput(
        preds_scaled=np.zeros((n_windows, pred_len, n_targets)),
        gt_scaled=np.zeros((n_windows, pred_len, n_targets)),
        samples_scaled=samples_scaled,
        preds_real=np.zeros((n_windows, pred_len, n_targets)),
        gt_real=np.zeros((n_windows, pred_len, n_targets)),
        samples_real=samples_real,
        metric_space="returns",
        return_transform="none",
        target_names=["Close"],
    )
    risk = RiskSimulator(fo_returns)
    var = risk.calculate_var()
    assert var.shape == (pred_len, n_targets)

    # metric_space="prices" → samples_for_metrics == samples_real
    fo_prices = ForecastOutput(
        preds_scaled=np.zeros((n_windows, pred_len, n_targets)),
        gt_scaled=np.zeros((n_windows, pred_len, n_targets)),
        samples_scaled=samples_scaled,
        preds_real=np.zeros((n_windows, pred_len, n_targets)),
        gt_real=np.zeros((n_windows, pred_len, n_targets)),
        samples_real=samples_real,
        metric_space="prices",
        return_transform="log_return",
        target_names=["Close"],
    )
    risk2 = RiskSimulator(fo_prices)
    var2 = risk2.calculate_var()
    assert var2.shape == (pred_len, n_targets)
    # Con precios reales, el VaR debe ser diferente al calculado sobre retornos escalados
    assert not np.allclose(risk.calculate_var(), risk2.calculate_var())


def test_portfolio_simulator_accepts_forecast_output() -> None:
    """PortfolioSimulator acepta ForecastOutput y usa preds/gt_for_metrics."""
    from training.forecast_output import ForecastOutput

    n_windows, pred_len, n_targets = 20, 10, 1
    rng = np.random.default_rng(13)
    preds_scaled = rng.normal(0, 0.01, (n_windows, pred_len, n_targets))
    gt_scaled = preds_scaled + rng.normal(0, 0.001, (n_windows, pred_len, n_targets))
    samples_scaled = rng.normal(0, 0.01, (50, n_windows, pred_len, n_targets))

    fo = ForecastOutput(
        preds_scaled=preds_scaled,
        gt_scaled=gt_scaled,
        samples_scaled=samples_scaled,
        preds_real=preds_scaled * 100 + 200,
        gt_real=gt_scaled * 100 + 200,
        samples_real=samples_scaled * 100 + 200,
        metric_space="returns",
        return_transform="none",
        target_names=["Close"],
    )

    ps = PortfolioSimulator(fo)
    strategy_returns = ps.run_simple_strategy()
    metrics = ps.calculate_metrics(strategy_returns)

    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert "sortino_ratio" in metrics
