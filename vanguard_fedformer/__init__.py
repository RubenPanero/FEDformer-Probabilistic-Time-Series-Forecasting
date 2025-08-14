"""
Vanguard-FEDformer: Advanced Probabilistic Time Series Forecasting

A modular implementation of FEDformer with normalizing flows,
regime detection, and advanced probabilistic forecasting capabilities.
"""

__version__ = "0.1.0"
__author__ = "Vanguard Team"
__description__ = "Advanced probabilistic time series forecasting with FEDformer"

from .core.models.fedformer import VanguardFEDformer
from .core.models.flows import NormalizingFlow
from .core.data.dataset import TimeSeriesDataset
from .core.training.trainer import VanguardTrainer

__all__ = [
    "VanguardFEDformer",
    "NormalizingFlow", 
    "TimeSeriesDataset",
    "VanguardTrainer"
]