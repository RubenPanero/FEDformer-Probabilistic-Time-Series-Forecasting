# -*- coding: utf-8 -*-
"""
Registro consolidado de corridas experimentales.

Permite cargar manifiestos JSON y CSVs de métricas generados por io_experiment.py,
construir una tabla consolidada y ordenar corridas por métrica de interés.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_run_manifests(results_dir: Path) -> list[dict]:
    """Carga todos los run_manifest_*.json de results_dir.

    Args:
        results_dir: Directorio donde buscar archivos run_manifest_*.json.

    Returns:
        Lista de dicts con el contenido de cada manifiesto. Lista vacía si no hay ninguno.
    """
    if not results_dir.exists():
        return []

    manifests: list[dict] = []
    for path in sorted(results_dir.glob("run_manifest_*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                manifests.append(json.load(fh))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("No se pudo cargar el manifiesto %s: %s", path, exc)

    return manifests


def load_probabilistic_metrics(results_dir: Path) -> pd.DataFrame:
    """Carga y concatena todos los probabilistic_metrics_*.csv de results_dir.

    Columnas esperadas: fold, metric, value, space, aggregation
    (+ timestamp, ticker si están presentes en el archivo).

    Args:
        results_dir: Directorio donde buscar archivos probabilistic_metrics_*.csv.

    Returns:
        DataFrame concatenado. DataFrame vacío si no hay archivos.
    """
    if not results_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(results_dir.glob("probabilistic_metrics_*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("No se pudo cargar %s: %s", path, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_portfolio_metrics(results_dir: Path) -> pd.DataFrame:
    """Carga y concatena todos los portfolio_metrics_*.csv de results_dir.

    Args:
        results_dir: Directorio donde buscar archivos portfolio_metrics_*.csv.

    Returns:
        DataFrame concatenado. DataFrame vacío si no hay archivos.
    """
    if not results_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(results_dir.glob("portfolio_metrics_*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("No se pudo cargar %s: %s", path, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _flatten_metrics_for_run(
    prob_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    timestamp: str,
    ticker: str,
) -> dict:
    """Aplana las métricas de una corrida a columnas clave agregadas por la media.

    Filtra por timestamp y ticker si las columnas correspondientes existen.
    Devuelve dict vacío si no hay datos para la combinación dada.

    Args:
        prob_metrics: DataFrame de métricas probabilísticas (puede estar vacío).
        portfolio_metrics: DataFrame de métricas de portfolio (puede estar vacío).
        timestamp: Marca temporal del run a filtrar.
        ticker: Ticker del run a filtrar.

    Returns:
        Dict con métricas aplanadas {nombre: valor}.
    """
    result: dict = {}

    # --- Métricas probabilísticas ---
    if not prob_metrics.empty:
        pm = prob_metrics.copy()
        # Filtrar por timestamp y ticker si las columnas están disponibles
        if "timestamp" in pm.columns:
            pm = pm[pm["timestamp"] == timestamp]
        if "ticker" in pm.columns:
            pm = pm[pm["ticker"] == ticker]

        if not pm.empty and "metric" in pm.columns and "value" in pm.columns:
            # Media por métrica a través de folds
            agg = pm.groupby("metric")["value"].mean()
            for metric, val in agg.items():
                result[f"prob_{metric}"] = float(val)

    # --- Métricas de portfolio ---
    if not portfolio_metrics.empty:
        pf = portfolio_metrics.copy()
        if "timestamp" in pf.columns:
            pf = pf[pf["timestamp"] == timestamp]
        if "ticker" in pf.columns:
            pf = pf[pf["ticker"] == ticker]

        if not pf.empty:
            # Para formato ancho: promediar columnas numéricas (excluyendo fold)
            num_cols = pf.select_dtypes(include="number").columns.tolist()
            num_cols = [c for c in num_cols if c != "fold"]
            if num_cols:
                for col in num_cols:
                    result[f"portfolio_{col}"] = float(pf[col].mean())

    return result


def build_experiment_table(
    manifests: list[dict],
    prob_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Construye tabla consolidada con una fila por corrida.

    Columnas principales: timestamp, ticker, seed, monitor_metric,
    más métricas clave aplanadas de los DataFrames de entrada.
    Si algún DataFrame está vacío, rellena con NaN.

    Args:
        manifests: Lista de dicts cargados por load_run_manifests.
        prob_metrics: DataFrame de métricas probabilísticas (puede estar vacío).
        portfolio_metrics: DataFrame de métricas de portfolio (puede estar vacío).

    Returns:
        DataFrame con una fila por corrida y columnas consolidadas.
        DataFrame vacío si manifests está vacío.
    """
    if not manifests:
        return pd.DataFrame()

    rows: list[dict] = []
    for manifest in manifests:
        timestamp = manifest.get("timestamp", "")
        ticker = manifest.get("ticker", "")

        row: dict = {
            "timestamp": timestamp,
            "ticker": ticker,
            "seed": manifest.get("seed"),
            "monitor_metric": manifest.get("monitor_metric"),
        }

        # Métricas del manifiesto (nivel superior)
        for key, val in manifest.get("metrics", {}).items():
            row[f"metric_{key}"] = val

        # Métricas aplanadas de los DataFrames
        flat = _flatten_metrics_for_run(
            prob_metrics, portfolio_metrics, timestamp, ticker
        )
        row.update(flat)

        rows.append(row)

    return pd.DataFrame(rows)


def rank_runs(
    df: pd.DataFrame, score_column: str, ascending: bool = False
) -> pd.DataFrame:
    """Ordena df por score_column.

    Args:
        df: DataFrame de corridas, típicamente generado por build_experiment_table.
        score_column: Nombre de la columna por la que ordenar.
        ascending: Si True, ordena de menor a mayor (útil para métricas de error).
            Por defecto False (mayor Sharpe primero).

    Returns:
        DataFrame ordenado por score_column (índice reseteado).

    Raises:
        ValueError: Si score_column no está en las columnas de df.
    """
    if score_column not in df.columns:
        raise ValueError(
            f"Columna '{score_column}' no encontrada en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    return df.sort_values(score_column, ascending=ascending).reset_index(drop=True)
