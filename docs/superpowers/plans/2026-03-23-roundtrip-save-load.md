# Roundtrip Save-Load Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that the full cycle `save_checkpoint → save_artifacts → register_specialist → load_specialist → predict` produces deterministic, identical outputs.

**Architecture:** Single test file with shared fixtures. A tiny model (d_model=32, 1 layer) keeps runtime <3s. Tests go from unit (config, preprocessor, model) to full integration.

**Tech Stack:** pytest, torch, pandas, numpy. No mocks of external services. No GPU required.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `tests/test_roundtrip.py` | All 5 roundtrip tests + fixtures |

No other files modified. The new test file is auto-discovered by pytest.

---

### Task 1: Fixtures and Config Roundtrip Test

**Files:**
- Create: `tests/test_roundtrip.py`

**Context:** `_build_config` lives in `inference/loader.py:102`. It reconstructs `FEDformerConfig` from a registry entry dict. The critical fields are the 18 arch params listed in `_save_canonical_specialist` (main.py:935). The `enc_in`/`dec_in` override at line 139-141 is the regression fix from session 19.

- [ ] **Step 1: Write test_config_roundtrip (RED)**

```python
"""Tests de roundtrip save→load para el ciclo canónico completo."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from config import FEDformerConfig
from data.preprocessing import PreprocessingPipeline
from inference.loader import _build_config, load_specialist
from models.fedformer import Flow_FEDformer
from utils.model_registry import register_specialist

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
    data = {name: np.random.randn(n_rows).cumsum() + 100 for name in FEATURE_NAMES}
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


# ── T1 ───────────────────────────────────────────────────────────────


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_roundtrip.py::test_config_roundtrip -v`
Expected: PASS (this test only exercises existing code, no new implementation needed). If it passes, good — it's our baseline assertion that the current code works.

- [ ] **Step 3: Commit**

```bash
git add tests/test_roundtrip.py
git commit -m "test: add config roundtrip test (T1)"
```

---

### Task 2: Preprocessor Artifacts Roundtrip Test

**Files:**
- Modify: `tests/test_roundtrip.py`

**Context:** `PreprocessingPipeline.save_artifacts` (data/preprocessing.py:542) saves `schema.json`, `metadata.json`, `scaler.pkl`. `load_artifacts` (line 596) restores them. Critical fields: `target_indices`, `feature_columns`, `scaler` transform behavior, `return_transform`, `last_prices`, `outlier_bounds`.

- [ ] **Step 1: Write test_preprocessor_artifacts_roundtrip (RED)**

Add to `tests/test_roundtrip.py`:

```python
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


# ── T2 ───────────────────────────────────────────────────────────────


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
    test_data = np.random.randn(10, N_FEATURES).astype(np.float32)
    original_scaled = fitted_preprocessor.scaler.transform(test_data)
    loaded_scaled = loaded.scaler.transform(test_data)
    np.testing.assert_array_equal(original_scaled, loaded_scaled)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_roundtrip.py::test_preprocessor_artifacts_roundtrip -v`
Expected: PASS (exercises existing save/load code).

- [ ] **Step 3: Commit**

```bash
git add tests/test_roundtrip.py
git commit -m "test: add preprocessor artifacts roundtrip test (T2)"
```

---

### Task 3: Model State Dict Roundtrip Test

**Files:**
- Modify: `tests/test_roundtrip.py`

**Context:** `save_checkpoint` (training/trainer.py:501) saves `model_state_dict` in a dict with keys `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `epoch`, `fold`, `loss`, `config`. `_load_model` (inference/loader.py:145) loads with `weights_only=True` and uses `safe_globals` for numpy scalars.

- [ ] **Step 1: Write test_model_state_dict_roundtrip (RED)**

Add to `tests/test_roundtrip.py`:

```python
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


# ── T3 ───────────────────────────────────────────────────────────────


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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_roundtrip.py::test_model_state_dict_roundtrip -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_roundtrip.py
git commit -m "test: add model state_dict roundtrip test (T3)"
```

---

### Task 4: Full Integration Roundtrip Test

**Files:**
- Modify: `tests/test_roundtrip.py`

**Context:** This is the crown jewel — exercises the real `register_specialist` + `load_specialist` chain. `load_specialist` (inference/loader.py:22) reads the registry, calls `_build_config`, `_load_model`, `_load_preprocessor`. The test must set up the full filesystem layout that `load_specialist` expects: registry JSON, canonical checkpoint, preprocessing artifacts directory.

Key gotcha: `_validated_file_path` (loader.py:91) validates that the CSV exists, so the synthetic CSV must be at the path recorded in `data_info["file"]`.

- [ ] **Step 1: Write test_full_roundtrip_save_load_predict (RED)**

Add to `tests/test_roundtrip.py`:

```python
# ── T4 ───────────────────────────────────────────────────────────────


def test_full_roundtrip_save_load_predict(
    tiny_config: FEDformerConfig,
    tiny_model: Flow_FEDformer,
    fitted_preprocessor: PreprocessingPipeline,
    synthetic_csv: Path,
    tmp_path: Path,
) -> None:
    """Ciclo completo: save checkpoint+artifacts+registry → load_specialist → predict idéntico."""
    # 1. Capturar output original
    x_enc, x_dec, x_regime = _make_input(tiny_config)
    with torch.no_grad():
        mean_original = tiny_model(x_enc, x_dec, x_regime).mean

    # 2. Guardar checkpoint (simula trainer.save_checkpoint para fold 3)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_src = ckpt_dir / "best_model_fold_3.pt"
    torch.save(
        {
            "model_state_dict": tiny_model.state_dict(),
            "optimizer_state_dict": {},
            "scaler_state_dict": None,
            "epoch": 5,
            "fold": 3,
            "loss": 0.123,
            "config": asdict(tiny_config),
        },
        ckpt_src,
    )

    # 3. Guardar preprocessing artifacts
    prep_dir = ckpt_dir / "test_preprocessing"
    fitted_preprocessor.save_artifacts(prep_dir)

    # 4. Registrar en registry (usa register_specialist real)
    config_dict = _make_config_dict(tiny_config)
    registry_path = tmp_path / "model_registry.json"
    data_info = {
        "file": str(synthetic_csv),
        "rows": 60,
        "features": N_FEATURES,
        "preprocessing_artifacts": str(prep_dir),
    }
    register_specialist(
        ticker="TEST",
        checkpoint_src=ckpt_src,
        metrics={"sharpe": 1.0, "sortino": 1.5, "max_drawdown": -0.2, "volatility": 0.1},
        config_dict=config_dict,
        data_info=data_info,
        registry_path=registry_path,
        canonical_dir=ckpt_dir,
    )

    # 5. Cargar con load_specialist (la función real de inferencia)
    model_loaded, config_loaded, preprocessor_loaded = load_specialist(
        "TEST", registry_path=registry_path
    )

    # 6. Verificar config
    for key in config_dict:
        original = config_dict[key]
        loaded = getattr(config_loaded, key)
        if isinstance(original, list):
            loaded = list(loaded)
        assert loaded == original, f"Config mismatch {key}: {loaded} != {original}"

    # 7. Verificar preprocessor
    assert preprocessor_loaded.fitted is True
    assert preprocessor_loaded.target_indices == fitted_preprocessor.target_indices
    assert preprocessor_loaded.feature_columns == fitted_preprocessor.feature_columns
    test_vec = np.random.randn(5, N_FEATURES).astype(np.float32)
    np.testing.assert_array_equal(
        preprocessor_loaded.scaler.transform(test_vec),
        fitted_preprocessor.scaler.transform(test_vec),
    )

    # 8. Verificar output del modelo
    model_loaded.eval()
    x_enc2, x_dec2, x_regime2 = _make_input(tiny_config)
    with torch.no_grad():
        mean_loaded = model_loaded(x_enc2, x_dec2, x_regime2).mean

    torch.testing.assert_close(mean_loaded, mean_original, atol=0, rtol=0)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_roundtrip.py::test_full_roundtrip_save_load_predict -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_roundtrip.py
git commit -m "test: add full integration roundtrip test (T4)"
```

---

### Task 5: enc_in/dec_in Regression Test

**Files:**
- Modify: `tests/test_roundtrip.py`

**Context:** Session 19 bug (commit `c4331c3`): `FEDformerConfig.__post_init__` reads the CSV and counts columns to set `enc_in`/`dec_in`. If the CSV has a `date` column and `date_column=None`, it counts `date` as a feature, inflating `enc_in` by 1. The fix in `_build_config` (loader.py:139-141) overwrites `enc_in`/`dec_in` with registry values after construction. This test ensures the fix holds.

- [ ] **Step 1: Write test_enc_in_dec_in_survives_roundtrip (RED)**

Add to `tests/test_roundtrip.py`:

```python
# ── T5 ───────────────────────────────────────────────────────────────


def test_enc_in_dec_in_survives_roundtrip(
    synthetic_csv: Path,
) -> None:
    """Regresión sesión 19: enc_in/dec_in del registry prevalecen sobre __post_init__."""
    # enc_in=5 en el registry, pero el CSV tiene 6 columnas (5 features + date).
    # Sin el fix, __post_init__ leería 6 columnas y pondría enc_in=6.
    config_dict = {**TINY, "target_features": TARGET, "enc_in": N_FEATURES, "dec_in": N_FEATURES}
    entry = {
        "config": config_dict,
        "data": {"file": str(synthetic_csv)},
    }

    rebuilt = _build_config(entry)

    assert rebuilt.enc_in == N_FEATURES, (
        f"enc_in corrupted by __post_init__: {rebuilt.enc_in} != {N_FEATURES}"
    )
    assert rebuilt.dec_in == N_FEATURES, (
        f"dec_in corrupted by __post_init__: {rebuilt.dec_in} != {N_FEATURES}"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_roundtrip.py::test_enc_in_dec_in_survives_roundtrip -v`
Expected: PASS (the fix already exists in `_build_config`).

- [ ] **Step 3: Run full suite**

Run: `pytest tests/test_roundtrip.py -v`
Expected: 5 passed.

Run: `pytest -q -m "not slow"`
Expected: 335 passed, 7 deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/test_roundtrip.py
git commit -m "test: add enc_in/dec_in regression test (T5)"
```

---

### Task 6: Final Verification

- [ ] **Step 1: Pre-commit pipeline**

```bash
ruff check . --fix && ruff format . && pylint --errors-only models/ training/ data/ utils/ inference/ && pytest -q -m "not slow"
```

Expected: all clean, 335 passed.

- [ ] **Step 2: Update hookmap (if needed)**

Check `.claude/hooks/post_python_quality.py` for `_MODULE_TEST_MAP`. Add entry if the hook requires it:

```python
"tests/test_roundtrip.py": "tests/test_roundtrip.py",
```

Since `test_roundtrip.py` doesn't map to a source module (it tests the integration of multiple modules), this entry may not be needed. Only add if the hook fails without it.

- [ ] **Step 3: Commit all remaining changes**

```bash
git add tests/test_roundtrip.py
git commit -m "chore: roundtrip save-load tests complete (#9)"
```
