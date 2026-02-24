# -*- coding: utf-8 -*-
"""
Simulador de riesgo para métricas financieras (VaR y CVaR).
Refactorizado con tipado nativo Python 3.10+ y diseño robusto para series.
"""

import numpy as np


class RiskSimulator:
    """Monitor de simulación predictiva con mitigantes de riegos bursátiles."""

    def __init__(self, samples: np.ndarray, confidence_level: float = 0.95) -> None:
        self.samples = samples
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
        tail_mask = losses >= var[None, :, :]
        tail_losses = np.where(tail_mask, losses, np.nan)

        with np.errstate(invalid="ignore"):
            cvar_result = np.nanmean(tail_losses, axis=0)

        return np.where(np.isnan(cvar_result), var, cvar_result)

    def calculate_expected_shortfall(self) -> np.ndarray:
        """Alias financiero: Expected Shortfall (Idéntico resolutivamente al CVaR)."""
        return self.calculate_cvar()
