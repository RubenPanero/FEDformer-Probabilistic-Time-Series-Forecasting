"""
Data handling modules for Vanguard-FEDformer.

Contains dataset classes, preprocessing utilities,
and regime detection algorithms.
"""

from .dataset import TimeSeriesDataset
from .preprocessing import DataPreprocessor
from .regime_detection import RegimeDetector

__all__ = [
    "TimeSeriesDataset",
    "DataPreprocessor", 
    "RegimeDetector"
]