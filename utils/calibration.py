# -*- coding: utf-8 -*-
"""
Conformal calibration helpers for probabilistic forecasting.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def conformal_quantile(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1
) -> float:
    """Compute split-conformal residual quantile for two-sided intervals."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    residuals = np.abs(y_true - y_pred).reshape(-1)
    if residuals.size == 0:
        raise ValueError("Calibration arrays must be non-empty")
    n = residuals.size
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def apply_conformal_interval(
    y_pred: np.ndarray,
    q_hat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build calibrated lower/upper intervals from point predictions."""
    if q_hat < 0:
        raise ValueError(f"q_hat must be non-negative, got {q_hat}")
    lower = y_pred - q_hat
    upper = y_pred + q_hat
    return lower, upper
