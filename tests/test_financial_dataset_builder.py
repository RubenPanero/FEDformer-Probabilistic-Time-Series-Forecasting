"""Tests para financial_dataset_builder — 11 features mínimas y validate_dataset."""

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from data.financial_dataset_builder import (
    _fetch_ohlcv,
    _normalize_ohlcv_frame,
    build_financial_dataset,
    validate_dataset,
)

EXPECTED_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VIX_Close",
    "RSI_14",
    "ATRr_14",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
]

REMOVED_FEATURES = [
    "SMA_20",
    "EMA_50",
    "STOCHk_14_3_3",
    "STOCHd_14_3_3",
    "BBL_20_2.0_2.0",
    "BBU_20_2.0_2.0",
    "OBV",
    "VWMA_20",
    "Sentiment_Score",
]


def _make_ohlcv_df(n_days: int = 1900) -> pd.DataFrame:
    """Genera OHLCV sintético con n_days días hábiles acabando hoy."""
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
    close = np.maximum(close, 10.0)  # mantener siempre positivo
    spread = np.abs(rng.normal(1.0, 0.5, n_days)) + 0.5
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, n_days),
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n_days).astype(float),
        },
        index=dates,
    )


def _make_multiindex_ohlcv_df(n_days: int = 60) -> pd.DataFrame:
    """Genera una respuesta estilo yfinance con MultiIndex en columnas."""
    single = _make_ohlcv_df(n_days)
    tuples = [(col, "NVDA") for col in single.columns]
    multi = pd.DataFrame(single.to_numpy(), index=single.index.copy())
    multi.columns = pd.MultiIndex.from_tuples(tuples)
    return multi


def _install_yfinance_stub(download_result):
    """Instala un stub de yfinance en sys.modules para pruebas unitarias."""
    original = sys.modules.get("yfinance")

    class _YFinanceStub:
        @staticmethod
        def download(*args, **kwargs):
            return download_result(*args, **kwargs) if callable(download_result) else download_result

    sys.modules["yfinance"] = _YFinanceStub()
    return original


def _restore_yfinance_stub(original):
    """Restaura el módulo yfinance original tras cada prueba."""
    if original is None:
        sys.modules.pop("yfinance", None)
    else:
        sys.modules["yfinance"] = original


@pytest.fixture
def nvda_df(tmp_path):
    """Construye dataset NVDA con mocks de red (yfinance y VIX)."""
    ohlcv_mock = _make_ohlcv_df()
    vix_mock = pd.DataFrame(
        {"VIX_Close": np.random.default_rng(0).uniform(10, 40, len(ohlcv_mock))},
        index=ohlcv_mock.index,
    )
    with (
        patch("yfinance.download", return_value=ohlcv_mock),
        patch(
            "data.vix_data.VixDataFetcher.get_vix_data",
            return_value=vix_mock,
        ),
    ):
        output_path = build_financial_dataset("NVDA", str(tmp_path))
    return pd.read_csv(output_path, index_col="date", parse_dates=True)


def test_fetch_ohlcv_uses_yfinance_with_or_without_legacy_flag():
    """_fetch_ohlcv debe usar yfinance sin flags de proveedor heredadas."""
    ohlcv_mock = _make_ohlcv_df(80)
    calls = []

    def _download(symbol, period, progress):
        calls.append((symbol, period, progress))
        return ohlcv_mock

    original = _install_yfinance_stub(_download)
    try:
        normalized = _fetch_ohlcv("NVDA")
    finally:
        _restore_yfinance_stub(original)

    assert calls == [("NVDA", "7y", False)]
    pd.testing.assert_frame_equal(normalized, ohlcv_mock)


def test_fetch_ohlcv_normalizes_multiindex_and_timezone():
    """_fetch_ohlcv debe aplanar MultiIndex y eliminar timezone del índice."""
    df_multi = _make_multiindex_ohlcv_df()
    df_multi.index = df_multi.index.tz_localize("UTC")

    original = _install_yfinance_stub(df_multi)
    try:
        normalized = _fetch_ohlcv("NVDA")
    finally:
        _restore_yfinance_stub(original)

    assert list(normalized.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert normalized.index.tz is None
    assert normalized.index.is_monotonic_increasing


def test_normalize_ohlcv_frame_drops_extra_columns_and_keeps_contract():
    """La normalización debe devolver solo Open/High/Low/Close/Volume."""
    raw = _make_ohlcv_df(40).assign(Adj_Close=lambda df: df["Close"] * 1.01)

    normalized = _normalize_ohlcv_frame(raw, "NVDA")

    assert list(normalized.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert "Adj_Close" not in normalized.columns


def test_normalize_ohlcv_frame_sorts_unsorted_index():
    """La normalización debe ordenar el índice ascendentemente."""
    raw = _make_ohlcv_df(20).sort_index(ascending=False)

    normalized = _normalize_ohlcv_frame(raw, "NVDA")

    assert normalized.index.is_monotonic_increasing


def test_normalize_ohlcv_frame_rejects_nans_in_required_ohlcv():
    """NaN en columnas OHLCV requeridas deben rechazarse en el contrato final."""
    raw = _make_ohlcv_df(20)
    raw.iloc[0, raw.columns.get_loc("Open")] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        _normalize_ohlcv_frame(raw, "NVDA")


def test_fetch_ohlcv_rejects_empty_dataset():
    """_fetch_ohlcv debe fallar con mensaje claro si yfinance retorna vacío."""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    original = _install_yfinance_stub(empty)
    try:
        with pytest.raises(ValueError, match="No se obtuvieron datos OHLCV"):
            _fetch_ohlcv("NVDA")
    finally:
        _restore_yfinance_stub(original)


def test_fetch_ohlcv_rejects_missing_ohlcv_columns():
    """_fetch_ohlcv debe validar las columnas OHLCV mínimas."""
    incomplete = _make_ohlcv_df(40).drop(columns=["Volume"])
    original = _install_yfinance_stub(incomplete)
    try:
        with pytest.raises(ValueError, match="columnas OHLCV requeridas"):
            _fetch_ohlcv("NVDA")
    finally:
        _restore_yfinance_stub(original)


def test_build_financial_dataset_rejects_legacy_flag_argument(tmp_path):
    """La interfaz final no debe aceptar el alias heredado use_mock."""
    with pytest.raises(TypeError):
        build_financial_dataset("NVDA", str(tmp_path), use_mock=True)


def test_build_financial_dataset_uses_final_public_interface(tmp_path):
    """El builder debe funcionar con la firma final explícita."""
    ohlcv_mock = _make_ohlcv_df()
    vix_mock = pd.DataFrame(
        {"VIX_Close": np.random.default_rng(123).uniform(10, 40, len(ohlcv_mock))},
        index=ohlcv_mock.index,
    )
    with (
        patch("yfinance.download", return_value=ohlcv_mock),
        patch("data.vix_data.VixDataFetcher.get_vix_data", return_value=vix_mock),
        patch.dict("os.environ", {}, clear=True),
    ):
        output = build_financial_dataset("NVDA", str(tmp_path))

    df = pd.read_csv(output, index_col="date", parse_dates=True)
    assert not df.empty
    assert list(df.columns) == EXPECTED_FEATURES


def test_exactly_11_features(nvda_df):
    """El dataset debe tener exactamente 11 features (sin columna date)."""
    assert nvda_df.shape[1] == 11, (
        f"Esperado 11 columnas, encontrado {nvda_df.shape[1]}: {list(nvda_df.columns)}"
    )


def test_all_expected_features_present(nvda_df):
    """Las 11 features objetivo deben estar presentes."""
    missing = [col for col in EXPECTED_FEATURES if col not in nvda_df.columns]
    assert missing == [], f"Features faltantes: {missing}"


def test_no_removed_features_present(nvda_df):
    """Indicadores eliminados no deben aparecer en el dataset."""
    present = [col for col in REMOVED_FEATURES if col in nvda_df.columns]
    assert present == [], f"Features que debían eliminarse siguen presentes: {present}"


def test_no_nulls_in_indicator_columns(nvda_df):
    """No debe haber NaN tras el dropna del builder."""
    null_counts = nvda_df.isnull().sum()
    assert null_counts.sum() == 0, f"NaN encontrados:\n{null_counts[null_counts > 0]}"


def test_ohlcv_consistency(nvda_df):
    """High debe ser siempre >= Low."""
    assert (nvda_df["High"] >= nvda_df["Low"]).all(), "High < Low en alguna fila"


def test_rsi_range(nvda_df):
    """RSI_14 debe estar en [0, 100]."""
    assert nvda_df["RSI_14"].between(0, 100).all(), "RSI fuera de rango [0, 100]"


def test_seven_years_of_data(nvda_df):
    """El dataset mock debe tener al menos 5 años de datos (yfinance period='7y')."""
    date_range_years = (nvda_df.index.max() - nvda_df.index.min()).days / 365
    assert date_range_years >= 5, (
        f"Dataset demasiado corto: {date_range_years:.1f} años"
    )


def test_validate_dataset_returns_dict(nvda_df):
    """validate_dataset debe retornar un dict con claves de integridad."""
    report = validate_dataset(nvda_df, "NVDA")
    assert isinstance(report, dict)
    for key in ("shape", "date_range", "max_date_gap_days", "price_inconsistencies"):
        assert key in report, f"Clave faltante en reporte: {key}"


def test_validate_dataset_detects_price_inconsistency():
    """validate_dataset debe contar filas con High < Low."""
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 99.0, 106.0],  # fila 1 es inválida
            "Low": [98.0, 103.0, 100.0],
            "Close": [103.0, 100.0, 104.0],
            "Volume": [1000, 2000, 1500],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )
    report = validate_dataset(df, "TEST")
    assert report["price_inconsistencies"] == 1


def test_alpha_vantage_dependency_removed_from_requirements() -> None:
    """La unificación a yfinance debe retirar alpha_vantage de requirements.txt."""
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "alpha_vantage" not in requirements


def test_alpha_vantage_client_module_removed() -> None:
    """El cliente legado de Alpha Vantage no debe seguir en el árbol activo."""
    assert not Path("data/alpha_vantage_client.py").exists()


def test_build_financial_dataset_signature_has_no_legacy_provider_flag() -> None:
    """La firma pública final no debe incluir use_mock."""
    params = inspect.signature(build_financial_dataset).parameters
    assert list(params) == ["symbol", "output_dir"]


def test_builder_cli_source_has_no_legacy_flag() -> None:
    """El CLI del builder no debe anunciar --use_mock."""
    source = Path("data/financial_dataset_builder.py").read_text(encoding="utf-8")
    assert "--use_mock" not in source
