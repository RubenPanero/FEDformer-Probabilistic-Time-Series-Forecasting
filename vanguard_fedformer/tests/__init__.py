"""
Test suite for Vanguard-FEDformer.

Contains unit tests for all major components
including models, data handling, and training.
"""

from . import test_models
from . import test_data
from . import test_training

__all__ = ["test_models", "test_data", "test_training"]