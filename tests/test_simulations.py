
import pytest
import numpy as np
from simulations import RiskSimulator, PortfolioSimulator


@pytest.fixture
def sample_data():
    # (n_samples, n_timesteps, n_assets)
    samples = np.random.randn(100, 50, 1)
    predictions = np.mean(samples, axis=0)
    ground_truth = predictions + np.random.randn(50, 1) * 0.1
    return samples, predictions, ground_truth


def test_risk_simulator(sample_data):
    samples, _, _ = sample_data
    risk_sim = RiskSimulator(samples)
    var = risk_sim.calculate_var()
    cvar = risk_sim.calculate_cvar()
    assert var.shape == (50, 1)
    assert cvar.shape == (50, 1)
    assert np.all(var >= 0)
    assert np.all(cvar >= 0)
    assert np.all(var <= cvar)


def test_portfolio_simulator(sample_data):
    _, predictions, ground_truth = sample_data
    portfolio_sim = PortfolioSimulator(predictions, ground_truth)
    strategy_returns = portfolio_sim.run_simple_strategy()
    metrics = portfolio_sim.calculate_metrics(strategy_returns)
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'volatility' in metrics
    assert 'sortino_ratio' in metrics
