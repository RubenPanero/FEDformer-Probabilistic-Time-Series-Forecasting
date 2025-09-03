# -*- coding: utf-8 -*-
"""
Simulador de portafolio con estrategias de trading.
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """Enhanced portfolio simulator with additional metrics"""

    def __init__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> None:
        self.predictions = predictions
        self.ground_truth = ground_truth

    def run_simple_strategy(self) -> np.ndarray:
        """Run a simple momentum strategy"""
        try:
            if self.predictions.shape[1] > 1 and self.ground_truth.shape[1] > 1:
                signals = np.sign(
                    self.predictions[:, 0, :] - self.ground_truth[:, 0, :]
                )
                actual_returns = (
                    self.ground_truth[:, 1, :] - self.ground_truth[:, 0, :]
                ) / (np.abs(self.ground_truth[:, 0, :]) + 1e-9)
                return signals[:-1] * actual_returns[1:]
            else:
                # Fallback for single timestep predictions
                signals = np.sign(np.diff(self.predictions[:, 0, :], axis=0))
                actual_returns = np.diff(self.ground_truth[:, 0, :], axis=0) / (
                    np.abs(self.ground_truth[:-1, 0, :]) + 1e-9
                )
                return signals * actual_returns
        except Exception as e:
            logger.warning(f"Strategy calculation failed: {e}")
            return np.zeros((len(self.predictions) - 1, self.predictions.shape[-1]))

    def calculate_metrics(self, strategy_returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        if strategy_returns.size == 0:
            return {
                "cumulative_returns": np.array([0]),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "sortino_ratio": 0.0,
            }

        try:
            # Handle multi-asset returns by averaging
            if strategy_returns.ndim > 1:
                strategy_returns = strategy_returns.mean(axis=1)

            cumulative_returns = np.cumprod(1 + strategy_returns) - 1

            # Sharpe ratio
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)
            sharpe_ratio = mean_return / (std_return + 1e-9) * np.sqrt(252)

            # Maximum drawdown
            equity_curve = np.concatenate(([1], 1 + cumulative_returns))
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / (peak + 1e-9)
            max_drawdown = np.min(drawdown)

            # Volatility
            volatility = std_return * np.sqrt(252)

            # Sortino ratio
            negative_returns = strategy_returns[strategy_returns < 0]
            downside_std = (
                np.std(negative_returns) if len(negative_returns) > 0 else 1e-9
            )
            sortino_ratio = mean_return / (downside_std + 1e-9) * np.sqrt(252)

            return {
                "cumulative_returns": cumulative_returns,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "sortino_ratio": sortino_ratio,
            }
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {
                "cumulative_returns": np.array([0]),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "sortino_ratio": 0.0,
            }
