# -*- coding: utf-8 -*-
"""
Sistema de seguimiento de métricas.
"""

import logging
from collections import defaultdict
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and log training metrics"""

    def __init__(self) -> None:
        self.metrics = defaultdict(list)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Append metrics for a given training step and emit a log message."""
        for key, value in metrics.items():
            self.metrics[key].append((step, value))
            logger.info("Step %s - %s: %.4f", step, key, value)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Return aggregate statistics for each recorded metric."""
        summary = {}
        for key, values in self.metrics.items():
            vals = [v[1] for v in values]
            summary[key] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "max": np.max(vals),
            }
        return summary
