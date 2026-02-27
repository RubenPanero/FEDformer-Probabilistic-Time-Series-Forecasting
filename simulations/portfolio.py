# -*- coding: utf-8 -*-
"""
Simulador de portafolio y backtesting con estrategias de trading temporales.
Refactorizado con tipado nativo Python 3.10+ y español estricto para registros estadísticos.
"""

from __future__ import annotations

import logging

import numpy as np

from training.forecast_output import ForecastOutput

logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """Monitor avanzado de simulación de portafolio con métricas analíticas."""

    def __init__(
        self,
        forecast: ForecastOutput | np.ndarray,
        ground_truth: np.ndarray | None = None,
    ) -> None:
        # Acepta ForecastOutput o np.ndarray para compatibilidad hacia atrás
        if isinstance(forecast, ForecastOutput):
            self.predictions = forecast.preds_for_metrics
            self.ground_truth = forecast.gt_for_metrics
        else:
            self.predictions = forecast
            self.ground_truth = ground_truth  # type: ignore[assignment]

    def run_simple_strategy(self) -> np.ndarray:
        """Opera una estrategia base de impulso temporal (momentum)."""
        try:
            if self.predictions.shape[1] > 1 and self.ground_truth.shape[1] > 1:
                # Señal: pendiente promedio predicha a través de todos los pasos temporales
                predicted_trend = (
                    self.predictions[:, -1, :] - self.predictions[:, 0, :]
                ) / (np.abs(self.predictions[:, 0, :]) + 1e-9)
                signals = np.sign(predicted_trend)

                # Retornos reales: promedio de retornos a través de todos los pasos
                step_returns = np.diff(self.ground_truth, axis=1) / (
                    np.abs(self.ground_truth[:, :-1, :]) + 1e-9
                )
                avg_actual_returns = step_returns.mean(axis=1)
                return signals[:-1] * avg_actual_returns[1:]

            # Recesión tolerante (Fallback) para inferencias predictivas de 1 intervalo
            signals = np.sign(np.diff(self.predictions[:, 0, :], axis=0))
            actual_returns = np.diff(self.ground_truth[:, 0, :], axis=0) / (
                np.abs(self.ground_truth[:-1, 0, :]) + 1e-9
            )
            return signals * actual_returns
        except Exception as exc:
            logger.warning("Fallo en modelado de estrategia iterativa: %s", exc)
            return np.zeros((len(self.predictions) - 1, self.predictions.shape[-1]))

    def calculate_metrics(
        self, strategy_returns: np.ndarray
    ) -> dict[str, float | np.ndarray]:
        """Calcula el ratio abarcativo de analíticas para el portafolio expuesto."""
        if strategy_returns.size == 0:
            return {
                "cumulative_returns": np.array([0.0]),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "sortino_ratio": 0.0,
            }

        try:
            # Gestiona retornos multi-activo mediante un suavizado promedio
            if strategy_returns.ndim > 1:
                strategy_returns = strategy_returns.mean(axis=1)

            cumulative_returns = np.cumprod(1 + strategy_returns) - 1

            # Sharpe ratio
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)
            sharpe_ratio = float(mean_return / (std_return + 1e-9) * np.sqrt(252))

            # Maximum drawdown (Caída máxima sostenida)
            equity_curve = np.concatenate(([1], 1 + cumulative_returns))
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / (peak + 1e-9)
            max_drawdown = float(np.min(drawdown))

            # Volatilidad estructural
            volatility = float(std_return * np.sqrt(252))

            # Sortino ratio (Penalización a los retornos divergentes negativos)
            negative_returns = strategy_returns[strategy_returns < 0]
            downside_std = (
                np.std(negative_returns) if len(negative_returns) > 0 else 1e-9
            )
            sortino_ratio = float(mean_return / (downside_std + 1e-9) * np.sqrt(252))

            return {
                "cumulative_returns": cumulative_returns,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "sortino_ratio": sortino_ratio,
            }
        except Exception as exc:
            logger.error("Error crítico consolidando métricas analíticas: %s", exc)
            return {
                "cumulative_returns": np.array([0.0]),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "sortino_ratio": 0.0,
            }
