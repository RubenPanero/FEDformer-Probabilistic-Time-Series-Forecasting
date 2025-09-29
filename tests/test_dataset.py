import pytest
from config import FEDformerConfig
from data import TimeSeriesDataset
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    data = {
        "date": pd.to_datetime(pd.date_range(start="2023-01-01", periods=100)),
        "Close": np.random.rand(100) * 100,
        "Volume": np.random.rand(100) * 1000,
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "sample_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_dataset_loading(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    assert len(dataset) > 0


def test_dataset_scaling(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    # The scaler should be fitted
    assert dataset.scaler is not None
    # The scaled data should have a mean close to 0 and std close to 1
    assert np.allclose(dataset.data_x.mean(), 0, atol=1e-1)
    assert np.allclose(dataset.data_x.std(), 1, atol=1e-1)


def test_dataset_inverse_scaling(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        date_column="date",
        seq_len=10,
        pred_len=4,
        label_len=5,
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    original_data = dataset.scaler.inverse_transform(dataset.data_x)
    assert np.allclose(
        original_data.mean(),
        pd.read_csv(sample_csv).drop("date", axis=1).mean().mean(),
        atol=100,
    )


def test_dataset_item(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        seq_len=10,
        pred_len=4,
        label_len=5,
        date_column="date",
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    item = dataset[0]
    assert "x_enc" in item
    assert "x_dec" in item
    assert "y_true" in item
    assert "x_regime" in item
    assert item["x_enc"].shape == (10, 2)  # seq_len, n_features
    assert item["y_true"].shape == (4, 1)  # pred_len, n_targets


def test_dataset_sequence_creation(sample_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=sample_csv,
        seq_len=10,
        pred_len=4,
        label_len=5,
        date_column="date",
    )
    dataset = TimeSeriesDataset(config=config, flag="train")
    item = dataset[0]
    # The last `label_len` values of the encoder input should be the same as the first `label_len` values of the decoder input
    assert np.allclose(
        item["x_enc"][-config.label_len :], item["x_dec"][: config.label_len]
    )
