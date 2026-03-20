# -*- coding: utf-8 -*-
"""Tests para el paquete inference."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def mock_registry(tmp_path):
    """Crea un registry self-contained en tmp_path con un especialista NVDA.

    IMPORTANTE: Crea un CSV sintético en tmp_path para que FEDformerConfig
    pueda leerlo al detectar enc_in/dec_in. El modelo y los datos usan
    las mismas 2 columnas (Close, Volume) para coherencia de dimensiones.
    """
    # 1. Crear CSV sintético — necesario para que FEDformerConfig.__init__ funcione
    rng = np.random.default_rng(42)
    n_rows = 200
    csv_path = tmp_path / "NVDA_features.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    # 2. Crear config y modelo con el CSV real → enc_in/dec_in = 2
    from config import FEDformerConfig

    config = FEDformerConfig(
        target_features=["Close"],
        file_path=str(csv_path),
        seq_len=20,  # Corto para tests rápidos
        label_len=10,
        pred_len=4,  # Par (requisito affine coupling)
        batch_size=8,
    )
    from models.fedformer import Flow_FEDformer

    model = Flow_FEDformer(config)

    # 3. Crear checkpoint
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": None,
        "epoch": 10,
        "fold": 3,
        "loss": 0.5,
        "config": {
            "seq_len": 20,
            "label_len": 10,
            "pred_len": 4,
            "n_splits": 4,
            "return_transform": "none",
            "metric_space": "returns",
            "gradient_clip_norm": 0.5,
            "batch_size": 8,
            "seed": 7,
            "target_features": ["Close"],
            # Parámetros de arquitectura (reflejan defaults con seq_len=20)
            "d_model": 512,
            "n_heads": 8,
            "d_ff": 2048,
            "e_layers": 2,
            "d_layers": 1,
            "modes": 10,  # clamped: max(1, seq_len//2) = 10
            "dropout": 0.1,
            "n_flow_layers": 4,
            "flow_hidden_dim": 64,
            "enc_in": 2,
            "dec_in": 2,
        },
    }
    torch.save(checkpoint, checkpoints / "nvda_canonical.pt")

    # 4. Crear preprocessing artifacts
    from sklearn.preprocessing import RobustScaler

    preproc_dir = checkpoints / "nvda_preprocessing"
    preproc_dir.mkdir()
    (preproc_dir / "schema.json").write_text(
        json.dumps(
            {
                "source_columns": ["Close", "Volume"],
                "feature_columns": ["Close", "Volume"],
                "target_features": ["Close"],
                "target_indices": [0],
                "numeric_columns": ["Close", "Volume"],
                "categorical_columns": [],
                "time_feature_columns": [],
                "onehot_columns": [],
                "category_mappings": {},
            }
        )
    )
    (preproc_dir / "metadata.json").write_text(
        json.dumps(
            {
                "fit_end_idx": 100,
                "fill_values": {},
                "outlier_bounds": {},
                "fit_stats": {},
                "return_transform": "none",
                "last_prices": {"Close": 100.0},
                "settings": {
                    "feature_roles": {},
                    "scaling_strategy": "robust",
                    "missing_policy": "impute_median",
                    "outlier_policy": "winsorize",
                    "fit_scope": "fold_train_only",
                    "persist_artifacts": False,
                    "drift_checks": {"enabled": False},
                    "strict_mode": False,
                    "categorical_encoding": "none",
                    "time_features": [],
                    "artifact_dir": "reports/preprocessing",
                    "return_transform": "none",
                },
            }
        )
    )
    scaler = RobustScaler()
    scaler.fit(rng.standard_normal((50, 2)))
    with (preproc_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    # 5. Crear registry JSON — file apunta al CSV en tmp_path
    registry = {
        "version": "1.0",
        "specialists": {
            "NVDA": {
                "checkpoint": str(checkpoints / "nvda_canonical.pt"),
                "config": checkpoint["config"],
                "data": {
                    "file": str(csv_path),
                    "rows": n_rows,
                    "features": 2,
                    "preprocessing_artifacts": str(preproc_dir),
                },
                "metrics": {"sharpe": 1.06},
            }
        },
    }
    registry_path = checkpoints / "model_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))

    return registry_path


def _rewrite_registry_ticker_key(registry_path, new_key: str) -> None:
    """Reescribe la clave del especialista para probar compatibilidad por casing."""
    registry = json.loads(registry_path.read_text())
    specialist = registry["specialists"].pop("NVDA")
    registry["specialists"][new_key] = specialist
    registry_path.write_text(json.dumps(registry, indent=2))


def test_predict_returns_forecast_output(mock_registry):
    """predict() retorna ForecastOutput con shapes correctos."""
    from inference.loader import load_specialist
    from inference.predictor import predict
    from training.forecast_output import ForecastOutput

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    # CSV sintético con suficientes filas (seq_len=20, pred_len=4)
    rng = np.random.default_rng(99)
    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "test_data.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    forecast = predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
        n_samples=3,  # mínimo para tests rápidos
    )

    assert isinstance(forecast, ForecastOutput)
    assert forecast.preds_real.size > 0
    assert forecast.quantiles_real is not None
    assert forecast.quantile_levels is not None
    assert forecast.preds_real.shape[1] == config.pred_len
    assert forecast.preds_real.shape[2] == len(config.target_features)


def test_predict_preserves_label_len(mock_registry):
    """predict() usa label_len del modelo entrenado, no el default del config."""
    from inference.loader import load_specialist
    from inference.predictor import _make_inference_config

    _, config, _ = load_specialist("NVDA", registry_path=mock_registry)
    # El mock usa label_len=10, seq_len=20 (seq_len//2 == label_len, latente)
    # Forzamos una discrepancia modificando label_len en config
    config.label_len = 7  # != seq_len//2 = 10

    csv_path = (
        mock_registry.parent.parent / "NVDA_features.csv"
    )  # tmp_path/NVDA_features.csv
    inference_cfg = _make_inference_config(config, str(csv_path))

    assert inference_cfg.label_len == 7, (
        f"label_len esperado=7, obtenido={inference_cfg.label_len}"
    )


def test_predict_does_not_refit_preprocessor(mock_registry, monkeypatch):
    """predict() no re-fittea el preprocessor — usa artefactos de entrenamiento."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)
    assert preprocessor.fitted, "El preprocessor debe estar fitted tras load_specialist"

    refit_calls = []
    original_fit = preprocessor.fit

    def tracking_fit(*args, **kwargs):
        refit_calls.append(1)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(preprocessor, "fit", tracking_fit)

    rng = np.random.default_rng(77)
    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "test_data2.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(rng.standard_normal(n_rows)) + 100,
            "Volume": rng.integers(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
        n_samples=3,
    )

    assert refit_calls == [], (
        f"preprocessor.fit fue llamado {len(refit_calls)} vez(ces) — no debe re-fitear en inferencia"
    )


def test_load_specialist_returns_model_config_preprocessor(mock_registry):
    """load_specialist retorna tupla (model, config, preprocessor)."""
    from inference.loader import load_specialist

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    assert model is not None
    # Verificar que config tiene los valores del registry
    assert config.seq_len == 20
    assert config.label_len == 10
    assert config.pred_len == 4
    assert config.target_features == ["Close"]
    # Verificar que preprocessor está fitted
    assert preprocessor.fitted is True
    # Verificar que el modelo está en eval mode
    assert not model.training


def test_load_specialist_accepts_lowercase_registry_key(mock_registry):
    """load_specialist resuelve claves legacy en minúsculas."""
    from inference.loader import load_specialist

    _rewrite_registry_ticker_key(mock_registry, "nvda")

    model, config, preprocessor = load_specialist("nvda", registry_path=mock_registry)

    assert model is not None
    assert config.label_len == 10
    assert preprocessor.fitted is True


def test_load_specialist_accepts_uppercase_query_for_lowercase_registry_key(
    mock_registry,
):
    """load_specialist debe resolver queries uppercase contra claves lowercase."""
    from inference.loader import load_specialist

    _rewrite_registry_ticker_key(mock_registry, "nvda")

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    assert model is not None
    assert config.target_features == ["Close"]
    assert preprocessor.fitted is True


def test_load_specialist_accepts_mixed_case_registry_key(mock_registry):
    """load_specialist resuelve claves mixed case sin exigir normalización previa."""
    from inference.loader import load_specialist

    _rewrite_registry_ticker_key(mock_registry, "NvDa")

    model, config, preprocessor = load_specialist("nvda", registry_path=mock_registry)

    assert model is not None
    assert config.pred_len == 4
    assert preprocessor.fitted is True
