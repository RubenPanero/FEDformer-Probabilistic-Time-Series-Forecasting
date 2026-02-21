from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import FEDformerConfig
from data import PreprocessingPipeline, TimeSeriesDataset


@pytest.fixture
def mixed_csv(tmp_path: Path) -> str:
    rng = np.random.default_rng(123)
    rows = 120
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "Close": rng.normal(100, 5, rows),
            "Volume": rng.normal(1_000, 50, rows),
            "sector": np.where(np.arange(rows) % 2 == 0, "tech", "energy"),
        }
    )
    df.loc[5, "Close"] = np.nan
    df.loc[15, "Volume"] = np.nan
    df.loc[20, "Close"] = 10_000.0
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_non_numeric_target_rejected(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["sector"],
        file_path=mixed_csv,
        date_column="date",
    )
    with pytest.raises(ValueError):
        TimeSeriesDataset(config=config, flag="train")


def test_missing_impute_is_deterministic(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        missing_policy="impute_median",
        scaling_strategy="robust",
    )
    pipeline = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipeline.fit(raw, fit_end_idx=80)
    first = pipeline.transform(raw).values
    second = pipeline.transform(raw).values
    assert np.allclose(first, second, atol=1e-12)


def test_outlier_winsorize_clips_extremes(mixed_csv: str) -> None:
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    base_cfg = dict(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
        missing_policy="impute_median",
        scaling_strategy="none",
    )
    cfg_none = FEDformerConfig(**base_cfg, outlier_policy="none")
    cfg_win = FEDformerConfig(**base_cfg, outlier_policy="winsorize")

    pipe_none = PreprocessingPipeline.from_config(
        cfg_none, target_features=cfg_none.target_features, date_column=cfg_none.date_column
    )
    pipe_none.fit(raw, fit_end_idx=80)
    none_vals = pipe_none.transform(raw)["Close"].to_numpy()

    pipe_win = PreprocessingPipeline.from_config(
        cfg_win, target_features=cfg_win.target_features, date_column=cfg_win.date_column
    )
    pipe_win.fit(raw, fit_end_idx=80)
    win_vals = pipe_win.transform(raw)["Close"].to_numpy()

    assert np.max(np.abs(win_vals)) < np.max(np.abs(none_vals))


def test_persistence_roundtrip_equivalence(mixed_csv: str, tmp_path: Path) -> None:
    config = FEDformerConfig(
        target_features=["Close", "Volume"],
        file_path=mixed_csv,
        date_column="date",
        missing_policy="impute_median",
        scaling_strategy="standard",
        categorical_encoding="onehot",
        drift_checks={"enabled": False},
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw, fit_end_idx=90)
    ref = pipe.transform(raw)

    artifact_dir = tmp_path / "prep"
    pipe.save_artifacts(artifact_dir)

    loaded = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    loaded.load_artifacts(artifact_dir)
    out = loaded.transform(raw)
    assert np.allclose(ref.values, out.values, atol=1e-8)


def test_drift_check_catches_missing_columns(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=mixed_csv,
        date_column="date",
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw, fit_end_idx=80)
    with pytest.raises(ValueError):
        pipe.validate_input_schema(raw.drop(columns=["Volume"]))


def test_inverse_transform_multitarget(mixed_csv: str) -> None:
    config = FEDformerConfig(
        target_features=["Close", "Volume"],
        file_path=mixed_csv,
        date_column="date",
        scaling_strategy="standard",
        missing_policy="impute_median",
        outlier_policy="none",
        drift_checks={"enabled": False},
    )
    raw = pd.read_csv(mixed_csv, parse_dates=["date"])
    pipe = PreprocessingPipeline.from_config(
        config, target_features=config.target_features, date_column=config.date_column
    )
    pipe.fit(raw, fit_end_idx=100)
    transformed = pipe.transform(raw)

    y_scaled = transformed[["Close", "Volume"]].to_numpy().reshape(-1, 1, 2)
    y_inv = pipe.inverse_transform_targets(y_scaled, ["Close", "Volume"]).reshape(-1, 2)

    close_fill = pipe.fill_values.get("Close", float(raw["Close"].median()))
    volume_fill = pipe.fill_values.get("Volume", float(raw["Volume"].median()))
    close = raw["Close"].fillna(close_fill).to_numpy()
    volume = raw["Volume"].fillna(volume_fill).to_numpy()
    assert np.allclose(y_inv[:, 0], close, atol=1e-5)
    assert np.allclose(y_inv[:, 1], volume, atol=1e-5)
