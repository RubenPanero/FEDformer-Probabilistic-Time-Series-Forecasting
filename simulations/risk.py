# -*- coding: utf-8 -*-
"""
Simulador de riesgo para métricas financieras (VaR y CVaR).
Refactorizado con tipado nativo Python 3.10+ y diseño robusto para series.
"""

from __future__ import annotations

import numpy as np

from training.forecast_output import ForecastOutput


class RiskSimulator:
    """Monitor de simulación predictiva con mitigantes de riegos bursátiles."""

    def __init__(
        self,
        forecast: ForecastOutput | np.ndarray,
        confidence_level: float = 0.95,
    ) -> None:
        # Acepta ForecastOutput o np.ndarray para compatibilidad hacia atrás
        if isinstance(forecast, ForecastOutput):
            raw = forecast.samples_for_metrics
        else:
            raw = forecast
        # Si las muestras son 4D (n_samples, n_windows, pred_len, n_targets),
        # se aplana el eje de ventanas en el eje de muestras para obtener
        # (n_samples * n_windows, pred_len, n_targets) y así el cuantil
        # se calcula sobre toda la distribución de escenarios.
        if raw.ndim == 4:
            n_samples, n_windows, pred_len, n_targets = raw.shape
            self.samples = raw.reshape(n_samples * n_windows, pred_len, n_targets)
        else:
            self.samples = raw
        self.confidence_level = confidence_level

    def calculate_var(self) -> np.ndarray:
        """Cálculo estocástico del VaR (Valor en Riesgo) a umbral fijo."""
        # El VaR estima retornos negativos que causen severidad temporal.
        losses = -self.samples
        return np.quantile(losses, self.confidence_level, axis=0)

    def calculate_cvar(self) -> np.ndarray:
        """Cálculo estocástico del Conditional Value at Risk (VaR Condicional)."""
        losses = -self.samples
        var = self.calculate_var()
        # Expande var para que sea compatible con losses independientemente del ndim
        tail_mask = losses >= var[np.newaxis, ...]
        tail_losses = np.where(tail_mask, losses, np.nan)

        with np.errstate(invalid="ignore"):
            cvar_result = np.nanmean(tail_losses, axis=0)

        return np.where(np.isnan(cvar_result), var, cvar_result)

    def calculate_expected_shortfall(self) -> np.ndarray:
        """Alias financiero: Expected Shortfall (Idéntico resolutivamente al CVaR)."""
        return self.calculate_cvar()
