# -*- coding: utf-8 -*-
"""
Tests unitarios para utils/model_registry.py:
  - load_registry, save_registry, register_specialist, get_specialist, list_specialists
"""

from pathlib import Path

from utils.model_registry import (
    get_specialist,
    list_specialists,
    load_registry,
    register_specialist,
    save_registry,
)

# ---------------------------------------------------------------------------
# Fixtures de apoyo
# ---------------------------------------------------------------------------

DUMMY_METRICS = {
    "sharpe": 0.65,
    "sortino": 1.05,
    "max_drawdown": -0.74,
    "var_95": 0.05,
}

DUMMY_CONFIG = {
    "seq_len": 96,
    "pred_len": 20,
    "n_splits": 4,
    "return_transform": "log_return",
    "metric_space": "returns",
    "gradient_clip_norm": 0.5,
    "batch_size": 64,
}

DUMMY_DATA_INFO = {
    "file": "data/NVDA_features.csv",
    "rows": 1725,
    "features": 11,
}


# ---------------------------------------------------------------------------
# Grupo 1: load_registry y save_registry
# ---------------------------------------------------------------------------


def test_load_registry_returns_empty_when_missing(tmp_path: Path) -> None:
    """load_registry retorna dict vacío si el archivo no existe."""
    registry_path = tmp_path / "nonexistent.json"
    result = load_registry(registry_path)
    assert result == {}


def test_save_and_load_registry_is_idempotent(tmp_path: Path) -> None:
    """save_registry + load_registry preserva los datos exactamente."""
    registry_path = tmp_path / "registry.json"
    original = {
        "version": "1.0",
        "last_updated": "2026-03-05",
        "specialists": {
            "NVDA": {
                "checkpoint": "checkpoints/nvda_canonical.pt",
                "trained_at": "2026-03-05",
                "config": DUMMY_CONFIG,
                "metrics": DUMMY_METRICS,
                "data": DUMMY_DATA_INFO,
                "training_command": "python3 main.py ...",
                "notes": "Test entry",
            }
        },
    }

    save_registry(original, registry_path)
    loaded = load_registry(registry_path)

    assert loaded == original


def test_save_registry_creates_parent_directory(tmp_path: Path) -> None:
    """save_registry crea el directorio padre si no existe."""
    nested_path = tmp_path / "subdir" / "deep" / "registry.json"
    registry = {"version": "1.0", "specialists": {}}
    save_registry(registry, nested_path)
    assert nested_path.exists()


# ---------------------------------------------------------------------------
# Grupo 2: register_specialist
# ---------------------------------------------------------------------------


def test_register_specialist_creates_canonical_checkpoint(tmp_path: Path) -> None:
    """register_specialist copia el checkpoint al nombre canónico."""
    # Crear dummy checkpoint fuente
    src_ckpt = tmp_path / "model.pt"
    src_ckpt.write_bytes(b"dummy_checkpoint_data")

    registry_path = tmp_path / "registry.json"
    canonical_dir = tmp_path / "checkpoints"

    result = register_specialist(
        ticker="NVDA",
        checkpoint_src=src_ckpt,
        metrics=DUMMY_METRICS,
        config_dict=DUMMY_CONFIG,
        data_info=DUMMY_DATA_INFO,
        training_command="python3 main.py --csv data/NVDA_features.csv",
        notes="Test specialist",
        registry_path=registry_path,
        canonical_dir=canonical_dir,
    )

    # Verificar que el checkpoint canónico existe con el nombre correcto
    expected_canonical = canonical_dir / "nvda_canonical.pt"
    assert expected_canonical.exists(), (
        f"Checkpoint canónico no encontrado en {expected_canonical}"
    )
    assert result == expected_canonical

    # Verificar que el contenido es el correcto (copia fiel)
    assert expected_canonical.read_bytes() == b"dummy_checkpoint_data"

    # Verificar que el ticker está registrado en el registry
    registry = load_registry(registry_path)
    assert "NVDA" in registry["specialists"]


def test_register_specialist_updates_existing_ticker(tmp_path: Path) -> None:
    """Re-registrar un ticker actualiza su entrada sin duplicar."""
    src_ckpt = tmp_path / "model.pt"
    src_ckpt.write_bytes(b"v1")

    registry_path = tmp_path / "registry.json"
    canonical_dir = tmp_path / "checkpoints"

    # Primer registro
    register_specialist(
        ticker="NVDA",
        checkpoint_src=src_ckpt,
        metrics={"sharpe": 0.5},
        config_dict=DUMMY_CONFIG,
        data_info=DUMMY_DATA_INFO,
        registry_path=registry_path,
        canonical_dir=canonical_dir,
    )

    # Segundo registro con métricas actualizadas
    src_ckpt.write_bytes(b"v2")
    register_specialist(
        ticker="NVDA",
        checkpoint_src=src_ckpt,
        metrics={"sharpe": 0.75},
        config_dict=DUMMY_CONFIG,
        data_info=DUMMY_DATA_INFO,
        registry_path=registry_path,
        canonical_dir=canonical_dir,
    )

    registry = load_registry(registry_path)

    # Solo debe haber un entry para NVDA
    assert list(registry["specialists"].keys()).count("NVDA") == 1
    # Las métricas deben ser las del último registro
    assert registry["specialists"]["NVDA"]["metrics"]["sharpe"] == 0.75


def test_register_specialist_multiple_tickers(tmp_path: Path) -> None:
    """register_specialist puede registrar múltiples tickers sin colisión."""
    registry_path = tmp_path / "registry.json"
    canonical_dir = tmp_path / "checkpoints"

    for ticker in ["NVDA", "GOOGL", "MSFT"]:
        src_ckpt = tmp_path / f"{ticker.lower()}.pt"
        src_ckpt.write_bytes(f"weights_{ticker}".encode())
        register_specialist(
            ticker=ticker,
            checkpoint_src=src_ckpt,
            metrics=DUMMY_METRICS,
            config_dict=DUMMY_CONFIG,
            data_info=DUMMY_DATA_INFO,
            registry_path=registry_path,
            canonical_dir=canonical_dir,
        )

    registry = load_registry(registry_path)
    assert set(registry["specialists"].keys()) == {"NVDA", "GOOGL", "MSFT"}

    # Verificar que cada checkpoint canónico existe y tiene el contenido correcto
    for ticker in ["NVDA", "GOOGL", "MSFT"]:
        canonical = canonical_dir / f"{ticker.lower()}_canonical.pt"
        assert canonical.exists()
        assert canonical.read_bytes() == f"weights_{ticker}".encode()


def test_register_specialist_updates_last_updated(tmp_path: Path) -> None:
    """register_specialist actualiza registry['last_updated'] con la fecha de hoy."""
    from datetime import date

    src_ckpt = tmp_path / "model.pt"
    src_ckpt.write_bytes(b"data")

    registry_path = tmp_path / "registry.json"
    canonical_dir = tmp_path / "checkpoints"

    register_specialist(
        ticker="AAPL",
        checkpoint_src=src_ckpt,
        metrics=DUMMY_METRICS,
        config_dict=DUMMY_CONFIG,
        data_info=DUMMY_DATA_INFO,
        registry_path=registry_path,
        canonical_dir=canonical_dir,
    )

    registry = load_registry(registry_path)
    assert registry.get("last_updated") == date.today().isoformat()


# ---------------------------------------------------------------------------
# Grupo 3: get_specialist y list_specialists
# ---------------------------------------------------------------------------


def test_get_specialist_returns_none_for_unknown(tmp_path: Path) -> None:
    """get_specialist retorna None para un ticker no registrado."""
    registry_path = tmp_path / "registry.json"
    result = get_specialist("UNKNOWN", registry_path)
    assert result is None


def test_get_specialist_returns_entry_for_registered(tmp_path: Path) -> None:
    """get_specialist retorna la entrada correcta para un ticker registrado."""
    src_ckpt = tmp_path / "model.pt"
    src_ckpt.write_bytes(b"data")

    registry_path = tmp_path / "registry.json"
    canonical_dir = tmp_path / "checkpoints"

    register_specialist(
        ticker="NVDA",
        checkpoint_src=src_ckpt,
        metrics=DUMMY_METRICS,
        config_dict=DUMMY_CONFIG,
        data_info=DUMMY_DATA_INFO,
        notes="Especialista NVDA",
        registry_path=registry_path,
        canonical_dir=canonical_dir,
    )

    entry = get_specialist("NVDA", registry_path)

    assert entry is not None
    assert entry["metrics"] == DUMMY_METRICS
    assert entry["config"] == DUMMY_CONFIG
    assert entry["notes"] == "Especialista NVDA"


def test_list_specialists_returns_registered_tickers(tmp_path: Path) -> None:
    """list_specialists retorna los tickers registrados."""
    registry_path = tmp_path / "registry.json"
    canonical_dir = tmp_path / "checkpoints"

    # Sin registros: lista vacía
    assert list_specialists(registry_path) == []

    # Registrar dos tickers
    for ticker in ["NVDA", "GOOGL"]:
        src_ckpt = tmp_path / f"{ticker.lower()}.pt"
        src_ckpt.write_bytes(b"data")
        register_specialist(
            ticker=ticker,
            checkpoint_src=src_ckpt,
            metrics=DUMMY_METRICS,
            config_dict=DUMMY_CONFIG,
            data_info=DUMMY_DATA_INFO,
            registry_path=registry_path,
            canonical_dir=canonical_dir,
        )

    tickers = list_specialists(registry_path)
    assert set(tickers) == {"NVDA", "GOOGL"}


def test_list_specialists_empty_when_registry_missing(tmp_path: Path) -> None:
    """list_specialists retorna lista vacía si el registry no existe."""
    registry_path = tmp_path / "nonexistent_registry.json"
    result = list_specialists(registry_path)
    assert result == []
