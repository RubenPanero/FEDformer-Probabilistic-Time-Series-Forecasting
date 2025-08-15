# -*- coding: utf-8 -*-
"""
Sistema de seguimiento de métricas.
"""

import logging
import numpy as np
from collections import defaultdict
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        for key, value in metrics.items():
            self.metrics[key].append((step, value))
            logger.info(f"Step {step} - {key}: {value:.4f}")
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for key, values in self.metrics.items():
            vals = [v[1] for v in values]
            summary[key] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals)
            }
        return summary

