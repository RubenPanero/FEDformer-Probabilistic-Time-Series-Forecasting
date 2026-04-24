"""Tests for reproducibility utilities used across the project."""

import random
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import Subset, TensorDataset
from torch.testing import assert_close

from config import FEDformerConfig
from training.trainer import _SeedWorker, WalkForwardTrainer
from utils.helpers import set_seed

FIXTURE_CSV = "tests/fixtures/NVDA_features.csv"


def _make_config(**kwargs: object) -> FEDformerConfig:
    return FEDformerConfig(
        target_features=["Close"],
        file_path=FIXTURE_CSV,
        batch_size=4,
        **kwargs,
    )


@pytest.mark.parametrize("seed", [1, 42, 123])
def test_set_seed_yields_reproducible_sequences(seed: int) -> None:
    """set_seed should synchronise RNGs for Python, NumPy and PyTorch."""
    set_seed(seed)
    python_value = random.random()
    numpy_value = float(np.random.rand())
    torch_value = torch.rand(2, 3)

    set_seed(seed)
    assert random.random() == pytest.approx(python_value)
    assert float(np.random.rand()) == pytest.approx(numpy_value)
    assert_close(torch.rand(2, 3), torch_value)


def test_seed_worker_replays_numpy_and_torch_sequences_per_worker_id() -> None:
    """_SeedWorker debe derivar una secuencia estable por worker_id."""
    worker = _SeedWorker(base_seed=123)

    worker(worker_id=2)
    numpy_first = float(np.random.rand())
    torch_first = torch.rand(3)

    worker(worker_id=2)
    assert float(np.random.rand()) == pytest.approx(numpy_first)
    assert_close(torch.rand(3), torch_first)


def test_prepare_data_loaders_use_seeded_generator_and_worker_init_fn() -> None:
    """Los loaders train/test deben compartir seed base y worker_init_fn picklable."""
    config = _make_config(seed=77, num_workers=0)
    trainer = WalkForwardTrainer(config, MagicMock())

    dummy_x = torch.zeros(6, config.seq_len, config.enc_in)
    subset = Subset(TensorDataset(dummy_x), list(range(6)))

    train_loader, test_loader = trainer._prepare_data_loaders(subset, subset)

    assert train_loader.generator is not None
    assert test_loader.generator is not None
    assert train_loader.generator.initial_seed() == 77
    assert test_loader.generator.initial_seed() == 77
    assert isinstance(train_loader.worker_init_fn, _SeedWorker)
    assert isinstance(test_loader.worker_init_fn, _SeedWorker)
    assert train_loader.worker_init_fn.base_seed == 77
    assert test_loader.worker_init_fn.base_seed == 77


def test_make_loader_uses_seeded_generator_and_disables_worker_init_without_workers() -> (
    None
):
    """_make_loader debe mantener generator seeded y omitir worker_init_fn con num_workers=0."""
    config = _make_config(seed=91, num_workers=0)
    trainer = WalkForwardTrainer(config, MagicMock())

    dummy_x = torch.zeros(5, config.seq_len, config.enc_in)
    subset = Subset(TensorDataset(dummy_x), list(range(5)))

    loader = trainer._make_loader(subset, shuffle=True)

    assert loader.generator is not None
    assert loader.generator.initial_seed() == 91
    assert loader.worker_init_fn is None
