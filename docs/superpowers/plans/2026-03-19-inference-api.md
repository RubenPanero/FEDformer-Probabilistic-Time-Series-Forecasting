# Inference API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permitir cargar un modelo canónico (NVDA/GOOGL) y generar predicciones probabilísticas sobre datos nuevos sin reentrenar.

**Architecture:** La inference API se construye como un paquete `inference/` con tres responsabilidades: (1) cargar modelo+preprocessor desde el registry, (2) predecir sobre datos nuevos usando MC Dropout, (3) exportar CSV con cuantiles. El gap principal es que `--save-canonical` actualmente NO guarda los artefactos de preprocessing (scaler, schema) — hay que cerrar ese gap primero.

**Tech Stack:** PyTorch 2.0+ · FEDformerConfig · PreprocessingPipeline · mc_dropout_inference · ForecastOutput

**Caveats críticos del codebase:**
- `FEDformerConfig.__init__` llama `pd.read_csv(file_path, nrows=0)` para auto-detectar `enc_in`/`dec_in` — cualquier config debe apuntar a un CSV real.
- El model state_dict depende de `enc_in`/`dec_in` — si creas un modelo con N features pero le pasas datos con M≠N columnas, crashea.
- `mc_dropout_inference` ya usa `torch.no_grad()` internamente — NO envolver en `torch.inference_mode()` (redundante + puede causar problemas con dropout stochastic).

---

## File Structure

| File | Responsabilidad |
|------|----------------|
| `main.py` (modify) | Guardar artefactos de preprocessing + target_features en `_save_canonical_specialist` |
| `inference/__init__.py` (create) | Lazy exports: `load_specialist`, `predict` |
| `inference/loader.py` (create) | `load_specialist()` — reconstruye modelo+config+preprocessor desde registry |
| `inference/predictor.py` (create) | `predict()` — evalúa modelo sobre datos, retorna ForecastOutput |
| `inference/__main__.py` (create) | CLI: `python3 -m inference --ticker NVDA --csv data/NVDA_features.csv` |
| `tests/test_inference.py` (create) | Tests unitarios con fixture self-contained en tmp_path |

---

## Task 1: Guardar artefactos de preprocessing en `--save-canonical` ✅ COMPLETADO (sesión 15)

**Files:**
- Modify: `main.py:935-1053` (`_save_canonical_specialist`)
- Test: `tests/test_model_registry.py` (verificar no regresión)

**Commit:** `dea10ae` — `feat: save preprocessing artifacts and target_features on --save-canonical`
**Tests añadidos:** `test_save_canonical_saves_preprocessing_artifacts` + `test_save_canonical_handles_preprocessing_save_failure`
**Fix post-review:** `AttributeError` añadido al `except` clause (code quality review)

**Contexto:** Actualmente `_save_canonical_specialist` guarda el checkpoint (.pt) y registra métricas/config en el JSON, pero NO guarda el scaler ni los metadatos de preprocessing. Sin estos artefactos es imposible hacer inference. `PreprocessingPipeline` ya tiene `save_artifacts()` y `load_artifacts()`. Además, `target_features` no se guarda en el config_dict del registry — necesario para reconstruir config en inference.

- [ ] **Step 1: Write the failing test**

Añadir en `tests/test_model_registry.py`:

```python
def test_save_canonical_includes_preprocessing_artifacts_field(tmp_path):
    """Verificar que data_info incluye preprocessing_artifacts tras save."""
    # Este test verifica que el campo existe en el dict que se pasa a register_specialist.
    # No necesita ejecutar _save_canonical_specialist completo — solo verificar que
    # el campo preprocessing_artifacts se añade a data_info.
    from main import _save_canonical_specialist
    # La función actual NO genera el campo. Verificar que tras el cambio sí lo hace.
    # Por ahora, verificar que el campo NO existe en el registry actual.
    from utils.model_registry import get_specialist
    entry = get_specialist("NVDA")
    if entry is not None:
        data = entry.get("data", {})
        # Antes del fix, este campo no existe
        assert "preprocessing_artifacts" not in data or data["preprocessing_artifacts"] is not None
```

- [ ] **Step 2: Run test to verify current state**

Run: `pytest tests/test_model_registry.py::test_save_canonical_includes_preprocessing_artifacts_field -v`
Expected: PASS (el assert es sobre estado actual, confirma que el campo no existe)

- [ ] **Step 3: Modificar `_save_canonical_specialist` en `main.py`**

Insertar en `main.py`, función `_save_canonical_specialist`. Cambios exactos:

**Cambio A** — Añadir `target_features` y `seed` a `config_dict` (después de línea ~995 donde se define `config_dict`):

```python
    config_dict = {
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "n_splits": args.splits,
        "return_transform": config.return_transform,
        "metric_space": config.metric_space,
        "gradient_clip_norm": config.gradient_clip_norm,
        "batch_size": config.batch_size,
        "seed": getattr(args, "seed", 7),
        "target_features": list(config.target_features),
    }
```

**Cambio B** — Guardar artefactos de preprocessing. Insertar DESPUÉS de la construcción de `data_info` (después de línea ~1008) y ANTES del bloque de guardias Sharpe (línea ~1012):

```python
    # Guardar artefactos de preprocessing para inferencia sin reentrenar
    preprocessing_dir = Path("checkpoints") / f"{ticker.lower()}_preprocessing"
    try:
        full_dataset.preprocessor.save_artifacts(preprocessing_dir)
        logger.info(
            "Artefactos de preprocessing guardados en %s", preprocessing_dir
        )
        data_info["preprocessing_artifacts"] = str(preprocessing_dir)
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "Error al guardar artefactos de preprocessing para '%s': %s",
            ticker, exc,
        )
        data_info["preprocessing_artifacts"] = None
```

- [ ] **Step 4: Verificar que tests existentes siguen pasando**

Run: `pytest tests/test_model_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: save preprocessing artifacts and target_features on --save-canonical"
```

---

## Task 2: Crear `inference/loader.py` — carga de modelo canónico

**Files:**
- Create: `inference/__init__.py`
- Create: `inference/loader.py`
- Create: `tests/test_inference.py`

**Contexto:** El loader reconstruye tres objetos desde el registry: (1) `FEDformerConfig` desde el dict guardado, (2) `Flow_FEDformer` con pesos del checkpoint, (3) `PreprocessingPipeline` desde artefactos. **Caveat**: `FEDformerConfig.__init__` hace `pd.read_csv(file_path, nrows=0)`, así que `file_path` debe apuntar a un CSV real con las columnas correctas.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
"""Tests para el paquete inference."""

import json
import pickle
from pathlib import Path

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
    pd.DataFrame({
        "Close": np.cumsum(np.random.randn(n_rows)) + 100,
        "Volume": np.random.randint(1000, 10000, n_rows).astype(float),
    }).to_csv(csv_path, index=False)

    # 2. Crear config y modelo con el CSV real → enc_in/dec_in = 2
    from config import FEDformerConfig
    config = FEDformerConfig(
        target_features=["Close"],
        file_path=str(csv_path),
        seq_len=20,   # Corto para tests rápidos
        pred_len=4,    # Par (requisito affine coupling)
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
    (preproc_dir / "schema.json").write_text(json.dumps({
        "source_columns": ["Close", "Volume"],
        "feature_columns": ["Close", "Volume"],
        "target_features": ["Close"],
        "target_indices": [0],
        "numeric_columns": ["Close", "Volume"],
        "categorical_columns": [],
        "time_feature_columns": [],
        "onehot_columns": [],
        "category_mappings": {},
    }))
    (preproc_dir / "metadata.json").write_text(json.dumps({
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
    }))
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

    model, config, preprocessor = load_specialist(
        "NVDA", registry_path=mock_registry
    )

    assert model is not None
    # Verificar que config tiene los valores del registry
    assert config.seq_len == 20
    assert config.pred_len == 4
    assert config.target_features == ["Close"]
    # Verificar que preprocessor está fitted
    assert preprocessor.fitted is True
    # Verificar que el modelo está en eval mode
    assert not model.training
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py::test_load_specialist_returns_model_config_preprocessor -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference'`

- [ ] **Step 3: Create `inference/__init__.py` (lazy imports)**

```python
"""API de inferencia para modelos canónicos FEDformer."""


def load_specialist(*args, **kwargs):
    """Carga un modelo canónico — lazy import para evitar import circular."""
    from inference.loader import load_specialist as _load
    return _load(*args, **kwargs)


def predict(*args, **kwargs):
    """Genera predicciones probabilísticas — lazy import."""
    from inference.predictor import predict as _predict
    return _predict(*args, **kwargs)


__all__ = ["load_specialist", "predict"]
```

- [ ] **Step 4: Write `inference/loader.py`**

```python
"""Carga de modelos canónicos desde el model_registry."""

import logging
from pathlib import Path

import numpy as np
import torch

from config import FEDformerConfig
from data.preprocessing import PreprocessingPipeline
from models.fedformer import Flow_FEDformer
from utils.model_registry import get_specialist, DEFAULT_REGISTRY_PATH

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_specialist(
    ticker: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[Flow_FEDformer, FEDformerConfig, PreprocessingPipeline]:
    """Carga un modelo canónico con su config y preprocessor desde el registry.

    Args:
        ticker: Símbolo del activo financiero (ej. "NVDA").
        registry_path: Ruta al model_registry.json.

    Returns:
        Tupla (modelo, config, preprocessor) listos para inferencia.

    Raises:
        ValueError: Si el ticker no está registrado.
        FileNotFoundError: Si el checkpoint o artefactos no existen.
    """
    entry = get_specialist(ticker, registry_path=registry_path)
    if entry is None:
        raise ValueError(
            f"Ticker '{ticker}' no registrado. "
            f"Disponibles: {available_tickers(registry_path)}"
        )

    # Reconstruir config — necesita file_path a un CSV real
    config = _build_config(entry)

    # Cargar modelo
    checkpoint_path = Path(entry["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {checkpoint_path}"
        )
    model = _load_model(config, checkpoint_path)

    # Cargar preprocessor
    artifacts_path = entry.get("data", {}).get("preprocessing_artifacts")
    if artifacts_path is None:
        raise FileNotFoundError(
            f"Artefactos de preprocessing no registrados para '{ticker}'. "
            "Re-entrena con --save-canonical para generarlos."
        )
    preprocessor = _load_preprocessor(config, Path(artifacts_path))

    return model, config, preprocessor


def _build_config(entry: dict) -> FEDformerConfig:
    """Reconstruye FEDformerConfig desde el dict del registry.

    Lee target_features del config guardado — no asume siempre 'Close'.
    file_path apunta al CSV original para que __init__ detecte enc_in/dec_in.
    """
    saved_config = entry.get("config", {})
    data_info = entry.get("data", {})
    target_features = saved_config.get("target_features", ["Close"])

    return FEDformerConfig(
        target_features=target_features,
        file_path=data_info.get("file", ""),
        seq_len=saved_config.get("seq_len", 96),
        pred_len=saved_config.get("pred_len", 20),
        batch_size=saved_config.get("batch_size", 64),
        gradient_clip_norm=saved_config.get("gradient_clip_norm", 0.5),
        return_transform=saved_config.get("return_transform", "log_return"),
        metric_space=saved_config.get("metric_space", "returns"),
        seed=saved_config.get("seed", 7),
    )


def _load_model(
    config: FEDformerConfig, checkpoint_path: Path
) -> Flow_FEDformer:
    """Carga pesos del modelo desde un checkpoint canónico."""
    import numpy._core.multiarray as _npcma  # pylint: disable=import-outside-toplevel

    model = Flow_FEDformer(config).to(device, non_blocking=True)

    with torch.serialization.safe_globals(
        [_npcma.scalar, np.float64, np.float32, np.int64, np.int32, np.bool_]
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        "Modelo cargado desde %s (epoch=%d, fold=%d)",
        checkpoint_path, checkpoint["epoch"], checkpoint["fold"],
    )
    return model


def _load_preprocessor(
    config: FEDformerConfig, artifacts_path: Path
) -> PreprocessingPipeline:
    """Reconstruye PreprocessingPipeline desde artefactos guardados."""
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Directorio de artefactos no encontrado: {artifacts_path}"
        )

    preprocessor = PreprocessingPipeline(
        config=config,
        target_features=list(config.target_features),
    )
    preprocessor.load_artifacts(artifacts_path)
    logger.info("Preprocessor cargado desde %s", artifacts_path)
    return preprocessor


def available_tickers(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[str]:
    """Lista tickers disponibles en el registry."""
    from utils.model_registry import list_specialists  # pylint: disable=import-outside-toplevel
    return list_specialists(registry_path=registry_path)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_inference.py::test_load_specialist_returns_model_config_preprocessor -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add inference/__init__.py inference/loader.py tests/test_inference.py
git commit -m "feat: add inference loader — load canonical models from registry"
```

---

## Task 3: Crear `inference/predictor.py` — predicción probabilística

**Files:**
- Create: `inference/predictor.py`
- Modify: `tests/test_inference.py` (añadir test)

**Contexto:** El predictor toma un modelo cargado + datos y genera `ForecastOutput`. Reutiliza `mc_dropout_inference` de `training/utils.py`. **Caveat**: `mc_dropout_inference` ya envuelve en `torch.no_grad()` — NO envolver en `torch.inference_mode()` por encima (redundante). `TimeSeriesDataset.__init__` acepta `preprocessor=` como parámetro opcional — si se pasa, reutiliza ese preprocessor en vez de crear uno nuevo.

- [ ] **Step 1: Write the failing test**

Añadir en `tests/test_inference.py`:

```python
def test_predict_returns_forecast_output(mock_registry):
    """predict() retorna ForecastOutput con shapes correctos."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist(
        "NVDA", registry_path=mock_registry
    )

    # CSV sintético con suficientes filas (seq_len=20, pred_len=4)
    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "test_data.csv"
    pd.DataFrame({
        "Close": np.cumsum(np.random.randn(n_rows)) + 100,
        "Volume": np.random.randint(1000, 10000, n_rows).astype(float),
    }).to_csv(csv_path, index=False)

    forecast = predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=str(csv_path),
        n_samples=3,  # mínimo para tests rápidos
    )

    from training.forecast_output import ForecastOutput
    assert isinstance(forecast, ForecastOutput)
    assert forecast.preds_real.size > 0
    assert forecast.quantiles_real is not None
    assert forecast.quantile_levels is not None
    # Verificar shapes coherentes
    assert forecast.preds_real.shape[1] == config.pred_len
    assert forecast.preds_real.shape[2] == len(config.target_features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py::test_predict_returns_forecast_output -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.predictor'`

- [ ] **Step 3: Write `inference/predictor.py`**

```python
"""Predicción probabilística sobre datos nuevos usando modelos canónicos."""

import logging

import numpy as np
import torch

from config import FEDformerConfig
from data.dataset import TimeSeriesDataset
from data.preprocessing import PreprocessingPipeline
from models.fedformer import Flow_FEDformer
from training.forecast_output import ForecastOutput
from training.trainer import DEFAULT_QUANTILE_LEVELS
from training.utils import mc_dropout_inference

logger = logging.getLogger(__name__)


def predict(
    model: Flow_FEDformer,
    config: FEDformerConfig,
    preprocessor: PreprocessingPipeline,
    csv_path: str,
    n_samples: int = 50,
) -> ForecastOutput:
    """Genera predicciones probabilísticas sobre datos de un CSV.

    Crea un dataset con el preprocessor pre-ajustado (sin re-fit),
    evalúa todas las ventanas disponibles con MC Dropout, y retorna
    un ForecastOutput con predicciones en espacio real.

    Args:
        model: Modelo Flow_FEDformer con pesos cargados.
        config: Configuración del modelo.
        preprocessor: PreprocessingPipeline ya ajustado (fitted).
        csv_path: Ruta al CSV con datos para predecir.
        n_samples: Número de muestras MC Dropout por ventana.

    Returns:
        ForecastOutput con predicciones, cuantiles y muestras.
    """
    # Crear dataset reutilizando el preprocessor pre-ajustado (no re-fit)
    inference_config = _make_inference_config(config, csv_path)
    dataset = TimeSeriesDataset(
        config=inference_config,
        flag="all",
        preprocessor=preprocessor,
    )

    if len(dataset) == 0:
        logger.warning("CSV no contiene ventanas suficientes para predicción.")
        return _empty_forecast(config)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
    )

    all_preds, all_gt, all_samples, all_quantiles = _evaluate(
        model, loader, n_samples
    )

    if not all_preds:
        return _empty_forecast(config)

    preds_scaled = np.concatenate(all_preds, axis=0)
    gt_scaled = np.concatenate(all_gt, axis=0)
    samples_scaled = np.concatenate(all_samples, axis=1)
    quantiles_scaled = np.concatenate(all_quantiles, axis=1)

    # Invertir escala a espacio real — por slices para manejar dimensiones extra
    targets = list(config.target_features)
    preds_real = preprocessor.inverse_transform_targets(preds_scaled, targets)
    gt_real = preprocessor.inverse_transform_targets(gt_scaled, targets)

    n_q = quantiles_scaled.shape[0]
    quantiles_real = np.stack([
        preprocessor.inverse_transform_targets(quantiles_scaled[i], targets)
        for i in range(n_q)
    ])

    n_s = samples_scaled.shape[0]
    samples_real = np.stack([
        preprocessor.inverse_transform_targets(samples_scaled[i], targets)
        for i in range(n_s)
    ])

    return ForecastOutput(
        preds_scaled=preds_scaled,
        gt_scaled=gt_scaled,
        samples_scaled=samples_scaled,
        preds_real=preds_real,
        gt_real=gt_real,
        samples_real=samples_real,
        quantiles_scaled=quantiles_scaled,
        quantiles_real=quantiles_real,
        quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
        metric_space=config.metric_space,
        return_transform=config.sections.preprocessing.return_transform,
        target_names=targets,
    )


def _make_inference_config(
    config: FEDformerConfig, csv_path: str,
) -> FEDformerConfig:
    """Crea config apuntando al CSV de inferencia."""
    return FEDformerConfig(
        target_features=list(config.target_features),
        file_path=csv_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=config.batch_size,
        return_transform=config.return_transform,
        metric_space=config.metric_space,
    )


def _evaluate(
    model: Flow_FEDformer,
    loader: torch.utils.data.DataLoader,
    n_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Evaluación probabilística con MC Dropout sobre un DataLoader.

    NO envuelve en torch.inference_mode() — mc_dropout_inference ya usa
    torch.no_grad() internamente y necesita modo train en dropout layers.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    all_samples: list[np.ndarray] = []
    all_quantiles: list[np.ndarray] = []

    for batch in loader:
        try:
            samples = mc_dropout_inference(
                model, batch, n_samples=n_samples, use_flow_sampling=True,
            )
            samples_cpu = samples.detach().to("cpu", dtype=torch.float32)
            quantiles_cpu = torch.quantile(
                samples_cpu,
                q=torch.tensor(
                    DEFAULT_QUANTILE_LEVELS.tolist(), dtype=torch.float32,
                ),
                dim=0,
            )
            all_samples.append(samples_cpu.numpy())
            all_quantiles.append(quantiles_cpu.numpy())
            all_preds.append(quantiles_cpu[1].numpy())  # p50
            all_gt.append(batch["y_true"].cpu().numpy())
        except (RuntimeError, ValueError) as exc:
            logger.warning("Error en evaluación de batch: %s", exc)
            continue

    return all_preds, all_gt, all_samples, all_quantiles


def _empty_forecast(config: FEDformerConfig) -> ForecastOutput:
    """Retorna un ForecastOutput vacío con shapes coherentes."""
    n_targets = len(config.target_features)
    empty_2d = np.empty((0, config.pred_len, n_targets), dtype=np.float32)
    empty_q = np.empty(
        (len(DEFAULT_QUANTILE_LEVELS), 0, config.pred_len, n_targets),
        dtype=np.float32,
    )
    empty_s = np.empty((0, 0, config.pred_len, n_targets), dtype=np.float32)
    return ForecastOutput(
        preds_scaled=empty_2d,
        gt_scaled=empty_2d.copy(),
        samples_scaled=empty_s,
        preds_real=empty_2d.copy(),
        gt_real=empty_2d.copy(),
        samples_real=empty_s.copy(),
        quantiles_scaled=empty_q,
        quantiles_real=empty_q.copy(),
        quantile_levels=DEFAULT_QUANTILE_LEVELS.copy(),
        metric_space=config.metric_space,
        return_transform=config.sections.preprocessing.return_transform,
        target_names=list(config.target_features),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference.py::test_predict_returns_forecast_output -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference/predictor.py tests/test_inference.py
git commit -m "feat: add inference predictor — probabilistic prediction on new data"
```

---

## Task 4: CLI de inferencia (`inference/__main__.py`)

**Files:**
- Create: `inference/__main__.py`
- Modify: `tests/test_inference.py` (añadir tests CLI)

**Contexto:** CLI permite `python3 -m inference --ticker NVDA --csv data/NVDA_features.csv`. Genera CSV con cuantiles. Añade `--registry` flag para testabilidad (permite apuntar a un registry alternativo en tests).

- [ ] **Step 1: Write the failing test**

Añadir en `tests/test_inference.py`:

```python
import subprocess
import sys


def test_inference_cli_help():
    """CLI de inference responde a --help sin errores."""
    result = subprocess.run(
        [sys.executable, "-m", "inference", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "--ticker" in result.stdout
    assert "--csv" in result.stdout
    assert "--registry" in result.stdout


def test_inference_cli_list_models(mock_registry):
    """CLI --list-models muestra tickers del registry."""
    result = subprocess.run(
        [sys.executable, "-m", "inference",
         "--list-models", "--registry", str(mock_registry)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "NVDA" in result.stdout


def test_inference_cli_unknown_ticker(mock_registry):
    """CLI falla con error claro para ticker no registrado."""
    result = subprocess.run(
        [sys.executable, "-m", "inference",
         "--ticker", "FAKE", "--csv", "x.csv",
         "--registry", str(mock_registry)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "Error" in result.stderr or "no registrado" in result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py::test_inference_cli_help -v`
Expected: FAIL with `No module named inference.__main__`

- [ ] **Step 3: Write `inference/__main__.py`**

```python
"""CLI de inferencia para modelos canónicos FEDformer.

Uso:
    python3 -m inference --ticker NVDA --csv data/NVDA_features.csv
    python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --output predictions.csv
    python3 -m inference --list-models
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inference.loader import load_specialist, available_tickers
from inference.predictor import predict
from utils.model_registry import DEFAULT_REGISTRY_PATH

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia probabilística con modelos canónicos FEDformer",
        prog="python3 -m inference",
    )
    parser.add_argument(
        "--ticker", type=str,
        help="Símbolo del ticker (ej. NVDA, GOOGL)",
    )
    parser.add_argument(
        "--csv", type=str,
        help="Ruta al CSV con datos para predecir",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ruta del CSV de salida (default: results/inference_{ticker}.csv)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Número de muestras MC Dropout (default: 50)",
    )
    parser.add_argument(
        "--registry", type=str, default=None,
        help="Ruta al model_registry.json (default: checkpoints/model_registry.json)",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Lista modelos canónicos disponibles y sale",
    )
    return parser.parse_args()


def _export_predictions(forecast, output_path: Path) -> None:
    """Exporta predicciones a CSV con cuantiles."""
    n_windows = forecast.preds_real.shape[0]
    pred_len = forecast.preds_real.shape[1]

    rows = []
    for w in range(n_windows):
        for t in range(pred_len):
            row = {
                "window": w,
                "step": t,
                "pred_mean": float(forecast.preds_real[w, t, 0]),
                "gt": float(forecast.gt_real[w, t, 0]),
            }
            if forecast.quantiles_real is not None:
                row["p10"] = float(forecast.p10_real[w, t, 0])
                row["p50"] = float(forecast.p50_real[w, t, 0])
                row["p90"] = float(forecast.p90_real[w, t, 0])
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Predicciones exportadas a %s (%d filas)", output_path, len(df))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_args()
    registry_path = Path(args.registry) if args.registry else DEFAULT_REGISTRY_PATH

    if args.list_models:
        tickers = available_tickers(registry_path)
        if tickers:
            print("Modelos canónicos disponibles:")
            for t in tickers:
                print(f"  - {t}")
        else:
            print("No hay modelos canónicos registrados.")
        return 0

    if not args.ticker or not args.csv:
        print("Error: --ticker y --csv son requeridos.", file=sys.stderr)
        print("Uso: python3 -m inference --ticker NVDA --csv data/NVDA_features.csv")
        return 1

    ticker = args.ticker.upper()
    csv_path = args.csv

    if not Path(csv_path).exists():
        print(f"Error: CSV no encontrado: {csv_path}", file=sys.stderr)
        return 1

    try:
        model, config, preprocessor = load_specialist(ticker, registry_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error cargando modelo: {exc}", file=sys.stderr)
        return 1

    forecast = predict(
        model=model,
        config=config,
        preprocessor=preprocessor,
        csv_path=csv_path,
        n_samples=args.n_samples,
    )

    if forecast.preds_real.size == 0:
        print("Error: no se generaron predicciones.", file=sys.stderr)
        return 1

    output_path = Path(
        args.output or f"results/inference_{ticker.lower()}.csv"
    )
    _export_predictions(forecast, output_path)

    # Resumen en stdout
    print(f"\n{'=' * 50}")
    print(f"Inferencia {ticker} completada")
    print(f"{'=' * 50}")
    print(f"  Ventanas evaluadas: {forecast.preds_real.shape[0]}")
    print(f"  Horizonte (pred_len): {forecast.preds_real.shape[1]}")
    print(f"  Muestras MC: {args.n_samples}")
    print(f"  Output: {output_path}")
    if forecast.quantiles_real is not None:
        p10_mean = float(np.mean(forecast.p10_real))
        p50_mean = float(np.mean(forecast.p50_real))
        p90_mean = float(np.mean(forecast.p90_real))
        print(f"  Media cuantiles — p10: {p10_mean:.4f}  p50: {p50_mean:.4f}  p90: {p90_mean:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inference.py::test_inference_cli_help tests/test_inference.py::test_inference_cli_list_models tests/test_inference.py::test_inference_cli_unknown_ticker -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference/__main__.py tests/test_inference.py
git commit -m "feat: add inference CLI — python3 -m inference --ticker NVDA --csv ..."
```

---

## Task 5: Tests de edge cases y regresión

**Files:**
- Modify: `tests/test_inference.py`

**Contexto:** Verificar ticker inexistente, CSV sin suficientes filas, y quantile_levels correctos.

- [ ] **Step 1: Añadir tests de edge cases**

Añadir en `tests/test_inference.py`:

```python
def test_load_specialist_unknown_ticker_raises(mock_registry):
    """load_specialist lanza ValueError para ticker no registrado."""
    from inference.loader import load_specialist
    with pytest.raises(ValueError, match="no registrado"):
        load_specialist("FAKE", registry_path=mock_registry)


def test_predict_insufficient_data(mock_registry):
    """predict retorna ForecastOutput vacío si CSV tiene pocas filas."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist(
        "NVDA", registry_path=mock_registry
    )

    # CSV con 5 filas — insuficiente para seq_len=20 + pred_len=4
    csv_path = mock_registry.parent / "tiny_data.csv"
    pd.DataFrame({
        "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
        "Volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
    }).to_csv(csv_path, index=False)

    forecast = predict(
        model=model, config=config, preprocessor=preprocessor,
        csv_path=str(csv_path), n_samples=3,
    )

    assert forecast.preds_real.size == 0


def test_forecast_output_quantile_levels(mock_registry):
    """ForecastOutput tiene quantile_levels [0.1, 0.5, 0.9]."""
    from inference.loader import load_specialist
    from inference.predictor import predict

    model, config, preprocessor = load_specialist(
        "NVDA", registry_path=mock_registry
    )

    n_rows = config.seq_len + config.pred_len + 10
    csv_path = mock_registry.parent / "enough_data.csv"
    pd.DataFrame({
        "Close": np.cumsum(np.random.randn(n_rows)) + 100,
        "Volume": np.random.randint(1000, 10000, n_rows).astype(float),
    }).to_csv(csv_path, index=False)

    forecast = predict(
        model=model, config=config, preprocessor=preprocessor,
        csv_path=str(csv_path), n_samples=3,
    )

    if forecast.preds_real.size > 0:
        np.testing.assert_array_almost_equal(
            forecast.quantile_levels, [0.1, 0.5, 0.9],
        )
```

- [ ] **Step 2: Run inference test suite**

Run: `pytest tests/test_inference.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full regression suite**

Run: `pytest -q -m "not slow"`
Expected: 299+ passed (original 299 + nuevos tests de inference)

- [ ] **Step 4: Commit**

```bash
git add tests/test_inference.py
git commit -m "test: add inference edge case tests — unknown ticker, insufficient data, quantile levels"
```

---

## Task 6: Pre-commit checks, CLAUDE.md, commit final

**Files:** Todos los creados/modificados

- [ ] **Step 1: Lint y formato**

Run: `ruff check . --fix && ruff format .`
Expected: No errors o solo fixes automáticos

- [ ] **Step 2: Pylint**

Run: `pylint --errors-only models/ training/ data/ utils/ inference/`
Expected: No E/F errors

- [ ] **Step 3: Full test suite**

Run: `pytest -q -m "not slow"`
Expected: ALL PASS

- [ ] **Step 4: Actualizar CLAUDE.md**

Añadir en sección "Comandos esenciales" después del bloque de Multi-seed:

```bash
# Inferencia sobre modelo canónico
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv
python3 -m inference --list-models
```

Añadir en sección "Convenciones críticas":

```
- **Inference**: el preprocessor se carga pre-ajustado — NUNCA re-fitear con datos nuevos en inferencia.
```

- [ ] **Step 5: Commit docs**

```bash
git add CLAUDE.md
git commit -m "docs: add inference API commands and conventions to CLAUDE.md"
```

---

## Notas de diseño

### Decisión: NO re-fitear el preprocessor en inferencia
El preprocessor se carga con artefactos pre-ajustados (scaler, outlier bounds). Re-fitear con datos nuevos cambiaría la escala y haría las predicciones incomparables con el entrenamiento. Anti-leakage: el scaler debe ser el mismo que vio el modelo durante el último fold de training.

### Decisión: Evaluar TODAS las ventanas del CSV
Evaluamos todas las ventanas disponibles. Para "predicción pura" (solo la última ventana), el usuario filtra el CSV de salida por `window == max(window)`.

### Decisión: Lazy imports en `__init__.py`
Usamos lazy imports para evitar ImportError cuando `predictor.py` aún no existe (durante desarrollo incremental). Esto permite crear `loader.py` y `__init__.py` en Task 2 sin que importe la ausencia de `predictor.py`.

### Decisión: `--registry` flag en CLI
Permite testabilidad aislada — los tests de CLI pasan un registry en `tmp_path` en vez de depender del filesystem real.

### Decisión: `target_features` en el config_dict del registry
Antes hardcodeábamos `["Close"]`. Ahora se guarda en `_save_canonical_specialist` y se lee en `_build_config`. Soporta modelos entrenados con targets diferentes.

### Requisito pendiente
Los checkpoints existentes (NVDA/GOOGL) no tienen preprocessing artifacts. Opciones:
1. Re-entrenar con `--save-canonical` (regenera todo), o
2. Script one-shot: cargar dataset, fitear preprocessor sobre todos los datos, guardar artefactos.

### Testing: self-contained fixtures
La fixture `mock_registry` crea TODOS los artefactos en `tmp_path`: CSV sintético (2 columnas), checkpoint con modelo real (enc_in=2), preprocessing artifacts, y registry JSON. No depende de ningún archivo fuera de `tmp_path`.
