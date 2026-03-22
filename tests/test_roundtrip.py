"""Tests de roundtrip save→load para el ciclo canónico completo."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from config import FEDformerConfig
from data.preprocessing import PreprocessingPipeline
from inference.loader import _build_config
from models.fedformer import Flow_FEDformer

# ── Dimensiones tiny para tests rápidos ──────────────────────────────
TINY = dict(
    d_model=32,
    n_heads=2,
    d_ff=64,
    e_layers=1,
    d_layers=1,
    modes=8,
    dropout=0.0,
    n_flow_layers=2,
    flow_hidden_dim=16,
    seq_len=16,
    pred_len=4,
    label_len=8,
    batch_size=2,
    return_transform="log_return",
    metric_space="returns",
    gradient_clip_norm=0.5,
    seed=7,
)
N_FEATURES = 5  # sin contar date
FEATURE_NAMES = ["Close", "High", "Low", "Open", "Volume"]
TARGET = ["Close"]


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """CSV sintético con N_FEATURES columnas + date."""
    n_rows = 60  # suficiente para seq_len=16 + pred_len=4
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    rng = np.random.default_rng(0)
    data = {name: rng.standard_normal(n_rows).cumsum() + 100 for name in FEATURE_NAMES}
    df = pd.DataFrame(data)
    df.insert(0, "date", dates)
    csv_path = tmp_path / "TEST_features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tiny_config(synthetic_csv: Path) -> FEDformerConfig:
    """FEDformerConfig con dimensiones mínimas apuntando al CSV sintético."""
    cfg = FEDformerConfig(
        target_features=TARGET,
        file_path=str(synthetic_csv),
        **TINY,
    )
    # Forzar enc_in/dec_in al nº real de features (no depender de __post_init__)
    cfg.enc_in = N_FEATURES
    cfg.dec_in = N_FEATURES
    return cfg


def _make_config_dict(cfg: FEDformerConfig) -> dict:
    """Construye config_dict igual que _save_canonical_specialist."""
    return {
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "label_len": cfg.label_len,
        "return_transform": cfg.return_transform,
        "metric_space": cfg.metric_space,
        "gradient_clip_norm": cfg.gradient_clip_norm,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "target_features": list(cfg.target_features),
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "e_layers": cfg.e_layers,
        "d_layers": cfg.d_layers,
        "modes": cfg.modes,
        "dropout": cfg.dropout,
        "n_flow_layers": cfg.n_flow_layers,
        "flow_hidden_dim": cfg.flow_hidden_dim,
        "enc_in": cfg.enc_in,
        "dec_in": cfg.dec_in,
    }


# ── T1: Config roundtrip ─────────────────────────────────────────────


def test_config_roundtrip(tiny_config: FEDformerConfig, synthetic_csv: Path) -> None:
    """Config sobrevive save→registry→_build_config sin perder arch params."""
    config_dict = _make_config_dict(tiny_config)
    entry = {
        "config": config_dict,
        "data": {"file": str(synthetic_csv)},
    }

    rebuilt = _build_config(entry)

    for key in config_dict:
        original = config_dict[key]
        loaded = getattr(rebuilt, key)
        if isinstance(original, list):
            loaded = list(loaded)
        assert loaded == original, f"{key}: {loaded} != {original}"


# ── T2: Preprocessor artifacts roundtrip ─────────────────────────────


@pytest.fixture
def fitted_preprocessor(
    tiny_config: FEDformerConfig, synthetic_csv: Path
) -> PreprocessingPipeline:
    """PreprocessingPipeline fitteado con el CSV sintético."""
    df = pd.read_csv(synthetic_csv)
    pp = PreprocessingPipeline.from_config(
        tiny_config,
        target_features=TARGET,
        date_column="date",
    )
    pp.fit(df)
    return pp


def test_preprocessor_artifacts_roundtrip(
    tiny_config: FEDformerConfig,
    fitted_preprocessor: PreprocessingPipeline,
    tmp_path: Path,
) -> None:
    """Preprocessor artifacts sobreviven save→load con transformaciones idénticas."""
    artifacts_dir = tmp_path / "preprocessing"
    fitted_preprocessor.save_artifacts(artifacts_dir)

    # Cargar en un pipeline nuevo
    loaded = PreprocessingPipeline(
        config=tiny_config,
        target_features=TARGET,
    )
    loaded.load_artifacts(artifacts_dir)

    # Verificar campos de schema
    assert loaded.feature_columns == fitted_preprocessor.feature_columns
    assert loaded.target_indices == fitted_preprocessor.target_indices
    assert loaded.numeric_columns == fitted_preprocessor.numeric_columns
    assert loaded.target_features == fitted_preprocessor.target_features

    # Verificar campos de metadata
    assert loaded.return_transform == fitted_preprocessor.return_transform
    assert loaded.last_prices == fitted_preprocessor.last_prices
    assert loaded.outlier_bounds == fitted_preprocessor.outlier_bounds
    assert loaded.fill_values == fitted_preprocessor.fill_values
    assert loaded.fitted is True

    # Verificar que el scaler produce la misma transformación
    rng = np.random.default_rng(1)
    test_data = rng.standard_normal((10, N_FEATURES)).astype(np.float32)
    original_scaled = fitted_preprocessor.scaler.transform(test_data)
    loaded_scaled = loaded.scaler.transform(test_data)
    np.testing.assert_array_equal(original_scaled, loaded_scaled)


# ── T3: Model state_dict roundtrip ───────────────────────────────────


@pytest.fixture
def tiny_model(tiny_config: FEDformerConfig) -> Flow_FEDformer:
    """Flow_FEDformer tiny en eval mode con seed determinista."""
    torch.manual_seed(42)
    model = Flow_FEDformer(tiny_config)
    model.eval()
    return model


def _make_input(cfg: FEDformerConfig, batch: int = 2) -> tuple:
    """Crea input determinista para forward pass."""
    torch.manual_seed(99)
    x_enc = torch.randn(batch, cfg.seq_len, cfg.enc_in)
    x_dec = torch.randn(batch, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_regime = torch.zeros(batch, 1, 1, dtype=torch.long)
    return x_enc, x_dec, x_regime


def test_model_state_dict_roundtrip(
    tiny_config: FEDformerConfig,
    tiny_model: Flow_FEDformer,
    tmp_path: Path,
) -> None:
    """Model state_dict sobrevive save→load con outputs bitwise idénticos."""
    # Forward pass original
    x_enc, x_dec, x_regime = _make_input(tiny_config)
    with torch.no_grad():
        dist_original = tiny_model(x_enc, x_dec, x_regime)
        mean_original = dist_original.mean

    # Guardar checkpoint (formato idéntico a trainer.save_checkpoint)
    checkpoint_path = tmp_path / "best_model_fold_3.pt"
    checkpoint = {
        "model_state_dict": tiny_model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": None,
        "epoch": 5,
        "fold": 3,
        "loss": 0.123,
        "config": asdict(tiny_config),
    }
    torch.save(checkpoint, checkpoint_path)

    # Cargar en modelo nuevo
    model_loaded = Flow_FEDformer(tiny_config)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_loaded.load_state_dict(saved["model_state_dict"])
    model_loaded.eval()

    # Forward pass con modelo cargado
    x_enc2, x_dec2, x_regime2 = _make_input(tiny_config)  # misma seed → mismo input
    with torch.no_grad():
        dist_loaded = model_loaded(x_enc2, x_dec2, x_regime2)
        mean_loaded = dist_loaded.mean

    torch.testing.assert_close(mean_loaded, mean_original, atol=0, rtol=0)
