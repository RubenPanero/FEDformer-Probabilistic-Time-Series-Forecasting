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
    n_rows = 200
    csv_path = tmp_path / "NVDA_features.csv"
    pd.DataFrame(
        {
            "Close": np.cumsum(np.random.randn(n_rows)) + 100,
            "Volume": np.random.randint(1000, 10000, n_rows).astype(float),
        }
    ).to_csv(csv_path, index=False)

    # 2. Crear config y modelo con el CSV real → enc_in/dec_in = 2
    from config import FEDformerConfig

    config = FEDformerConfig(
        target_features=["Close"],
        file_path=str(csv_path),
        seq_len=20,  # Corto para tests rápidos
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
            "pred_len": 4,
            "n_splits": 4,
            "return_transform": "none",
            "metric_space": "returns",
            "gradient_clip_norm": 0.5,
            "batch_size": 8,
            "seed": 7,
            "target_features": ["Close"],
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
    scaler.fit(np.random.randn(50, 2))
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


def test_load_specialist_returns_model_config_preprocessor(mock_registry):
    """load_specialist retorna tupla (model, config, preprocessor)."""
    from inference.loader import load_specialist

    model, config, preprocessor = load_specialist("NVDA", registry_path=mock_registry)

    assert model is not None
    # Verificar que config tiene los valores del registry
    assert config.seq_len == 20
    assert config.pred_len == 4
    assert config.target_features == ["Close"]
    # Verificar que preprocessor está fitted
    assert preprocessor.fitted is True
    # Verificar que el modelo está en eval mode
    assert not model.training
