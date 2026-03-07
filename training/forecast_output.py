# -*- coding: utf-8 -*-
"""
ForecastOutput: contenedor dual-space para predicciones escaladas y reales.
Permite transportar predicciones en espacio escalado (raw del modelo) y en
espacio real (precios o retornos no escalados) a través del pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ForecastOutput:
    """Contenedor de predicciones en espacio escalado y real."""

    # Espacio escalado (raw del modelo)
    preds_scaled: np.ndarray  # (n_windows, pred_len, n_targets)
    gt_scaled: np.ndarray  # (n_windows, pred_len, n_targets)
    samples_scaled: np.ndarray  # (n_samples, n_windows, pred_len, n_targets) o similar

    # Espacio real (precios o retornos no escalados)
    preds_real: np.ndarray
    gt_real: np.ndarray
    samples_real: np.ndarray
    quantiles_scaled: np.ndarray | None = (
        None  # (n_quantiles, n_windows, pred_len, n_targets)
    )
    quantiles_real: np.ndarray | None = None
    quantile_levels: np.ndarray | None = None  # (n_quantiles,)

    # Metadatos
    metric_space: str = "returns"  # "returns" | "prices"
    return_transform: str = "none"  # "none" | "log_return" | "simple_return"
    target_names: list[str] | None = None
    # Índice de fold por ventana — shape (n_windows,), dtype int32; None si no disponible
    window_fold_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.target_names is None:
            self.target_names = []

    @property
    def preds_for_metrics(self) -> np.ndarray:
        """Predicciones desescaladas para métricas financieras.

        Siempre devuelve preds_real: el espacio interpretable (retornos o precios
        dependiendo de return_transform). metric_space controla qué contiene
        preds_real (vía _inverse_transform_all), no qué array se selecciona aquí.
        """
        return self.preds_real

    @property
    def gt_for_metrics(self) -> np.ndarray:
        """Ground truth desescalado para métricas financieras."""
        return self.gt_real

    @property
    def samples_for_metrics(self) -> np.ndarray:
        """Muestras desescaladas para métricas financieras."""
        return self.samples_real

    @property
    def quantiles_for_metrics(self) -> np.ndarray | None:
        """Cuantiles desescalados para métricas financieras, si están disponibles."""
        return self.quantiles_real
