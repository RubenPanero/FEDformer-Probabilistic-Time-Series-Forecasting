"""
Módulos de entrenamiento y backtesting.
"""

from .trainer import WalkForwardTrainer
from .utils import mc_dropout_inference

__all__ = ["WalkForwardTrainer", "mc_dropout_inference"]
