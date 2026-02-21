# -*- coding: utf-8 -*-
"""
Simulador de riesgo para métricas financieras.
"""

import numpy as np


class RiskSimulator:
    """Enhanced risk simulator with additional metrics"""

    def __init__(self, samples: np.ndarray, confidence_level: float = 0.95) -> None:
        self.samples = samples
        self.confidence_level = confidence_level

    def calculate_var(self) -> np.ndarray:
        """Value at Risk calculation"""
        # VaR is a measure of loss, so we look at the negative of the samples
        losses = -self.samples
        return np.quantile(losses, self.confidence_level, axis=0)

    def calculate_cvar(self) -> np.ndarray:
        """Conditional Value at Risk calculation"""
        losses = -self.samples
        var = self.calculate_var()
        tail_mask = losses >= var[None, :, :]
        tail_losses = np.where(tail_mask, losses, np.nan)
        with np.errstate(invalid="ignore"):
            cvar_result = np.nanmean(tail_losses, axis=0)
        return np.where(np.isnan(cvar_result), var, cvar_result)

    def calculate_expected_shortfall(self) -> np.ndarray:
        """Expected Shortfall (same as CVaR)"""
        return self.calculate_cvar()
