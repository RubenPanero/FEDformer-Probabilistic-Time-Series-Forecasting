"""
Utilidades generales para el sistema FEDformer.
"""

from .metrics import MetricsTracker
from .helpers import _select_amp_dtype, setup_cuda_optimizations, get_device
from .calibration import conformal_quantile, apply_conformal_interval
from .model_registry import (
    load_registry,
    save_registry,
    register_specialist,
    get_specialist,
    list_specialists,
)

__all__ = [
    "MetricsTracker",
    "_select_amp_dtype",
    "setup_cuda_optimizations",
    "get_device",
    "conformal_quantile",
    "apply_conformal_interval",
    "load_registry",
    "save_registry",
    "register_specialist",
    "get_specialist",
    "list_specialists",
]
