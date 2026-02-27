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
