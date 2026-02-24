# -*- coding: utf-8 -*-
"""
Asistentes de calibración conforme (Conformal Calibration) para predicciones estocásticas.
Refactorizado a Python 3.10+ para garantizar typing nativo purificado, eficiencia y PEP 8.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1
) -> float:
    """Computa el cuantil residual dual para particiones analíticas."""
    if not 0 < alpha < 1:
        raise ValueError(f"El parámetro Alpha debe encontrarse en (0, 1), recibido {alpha}")
        
    if y_true.shape != y_pred.shape:
        raise ValueError("Las matrices y_true e y_pred deben estructurarse asimétricamente idénticas")
        
    residuals = np.abs(y_true - y_pred).reshape(-1)
    if residuals.size == 0:
        raise ValueError("Las matrices de calibración estocásticas carecen de vectores empíricos")
        
    n = residuals.size
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def apply_conformal_interval(
    y_pred: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye un canal paramétrico [Low, High] envolviendo predicciones directas."""
    if q_hat < 0:
        raise ValueError(f"El estimador hat de Cuantil debe ser positivo, pero se computó: {q_hat}")
        
    lower = y_pred - q_hat
    upper = y_pred + q_hat
    return lower, upper
