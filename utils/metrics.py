# -*- coding: utf-8 -*-
"""
Sistema de seguimiento analítico para el ciclo de vida del entrenamiento.
Refactorizado a Python 3.10+ PEP 8.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Monitor temporal in-memory local de logs algorítmicos."""

    def __init__(self) -> None:
        self.metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Adiciona métricas para su correlación en base al bloque (step) en ejecución."""
        for key, value in metrics.items():
            self.metrics[key].append((step, value))
            logger.info("Iteración Computada %s - [%s]: %.4f", step, key, value)

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Devuelve un objeto paramétrico purgado con la consolidación global histórica."""
        summary: dict[str, dict[str, float]] = {}
        for key, values in self.metrics.items():
            vals = [float(v[1]) for v in values]
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return summary
