# -*- coding: utf-8 -*-
"""
Simulador de riesgo para métricas financieras.
"""

import numpy as np


class RiskSimulator:
    """Enhanced risk simulator with additional metrics"""
    def __init__(self, samples: np.ndarray, confidence_level=0.95):
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
        cvar_result = np.zeros_like(var)
        
        for t in range(self.samples.shape[1]):
            for f in range(self.samples.shape[2]):
                tail_samples = losses[losses[:, t, f] >= var[t, f], t, f]
                if len(tail_samples) > 0:
                    cvar_result[t, f] = tail_samples.mean()
                else:
                    cvar_result[t, f] = var[t, f]
        return cvar_result

    def calculate_expected_shortfall(self) -> np.ndarray:
        """Expected Shortfall (same as CVaR)"""
        return self.calculate_cvar()

