# -*- coding: utf-8 -*-
"""
Búsqueda de hiperparámetros con Optuna para un ticker específico.

Cada trial ejecuta main.py como subproceso independiente (seed 42 limpio,
sin contaminación cross-ticker) y extrae el Sharpe del CSV de portafolio.

El estudio es persistente (SQLite): puede interrumpirse y retomarse con
los mismos flags. Los trials fallidos/expirados retornan Sharpe=-1.0.
Los trials donde seq_len < pred_len*3 se podan sin entrenar.

Uso:
    # Búsqueda básica (20 trials in-memory)
    python3 tune_hyperparams.py --csv data/MSFT_features.csv

    # Con persistencia SQLite (reanudable)
    python3 tune_hyperparams.py --csv data/MSFT_features.csv \\
        --n-trials 20 --storage-path optuna_studies/msft.db

    # Buscar Y registrar el mejor resultado en model_registry
    python3 tune_hyperparams.py --csv data/NVDA_features.csv \\
        --n-trials 16 --best-save-canonical

    # Descargar 8 tickers adicionales y lanzar búsqueda para los 12
    python3 tune_hyperparams.py --download-extra-tickers
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------

SEQ_LENS = [48, 64, 96, 128]
PRED_LENS = [4, 6, 8, 10, 20]  # todos pares (requisito del affine coupling)
BATCH_SIZES = [32, 64]
CLIP_NORMS = [0.3, 0.5]

# Tickers adicionales para alcanzar 12 activos — cobertura sectorial diversa:
#   Semiconductores:  AMD, INTC  (peers directos de NVDA)
#   Big tech:         AMZN, META (peers de MSFT/GOOGL/AAPL)
#   Alta volatilidad: TSLA, NFLX (NF se beneficia de colas gruesas)
#   Financiero:       JPM        (dinámica contra-cíclica al tech)
#   Enterprise SaaS:  CRM        (correlación tech pero ciclo propio)
EXTRA_TICKERS = ["AMD", "INTC", "AMZN", "META", "TSLA", "NFLX", "JPM", "CRM"]


# ---------------------------------------------------------------------------
# Parsing de resultados
# ---------------------------------------------------------------------------


def _current_time() -> float:
    """Aísla la fuente de tiempo para evitar side effects al parchear tests."""
    return time.time()


def _find_recent_result_file(
    results_dir: Path,
    prefix: str,
    ticker_stem: str,
    after_ts: float,
) -> Path | None:
    """Retorna el CSV de resultados más reciente para el trial actual.

    Acepta tanto el formato histórico con ticker en el nombre como el formato
    actual de main.py, que solo incluye el timestamp.
    """
    patterns = [
        f"{prefix}_*_{ticker_stem}.csv",
        f"{prefix}_*.csv",
    ]
    candidates: dict[Path, Path] = {}
    for pattern in patterns:
        for path in results_dir.glob(pattern):
            candidates[path.resolve()] = path

    recent = sorted(
        (path for path in candidates.values() if path.stat().st_mtime >= after_ts),
        key=lambda path: path.stat().st_mtime,
    )
    return recent[-1] if recent else None


def _parse_portfolio_csv(results_dir: Path, ticker_stem: str, after_ts: float) -> dict:
    """Retorna métricas del CSV de portafolio más reciente generado después de after_ts.

    Args:
        results_dir: Directorio donde se guardan los CSVs de resultados.
        ticker_stem: Stem del archivo CSV del ticker (ej. "MSFT_features").
        after_ts: Timestamp Unix mínimo del archivo (inicio del trial).

    Returns:
        Dict con sharpe, sortino, max_drawdown, volatility; o valores centinela −1.0.
    """
    recent = _find_recent_result_file(
        results_dir,
        prefix="portfolio_metrics",
        ticker_stem=ticker_stem,
        after_ts=after_ts,
    )
    if recent is None:
        logger.warning(
            "No se encontró CSV de portafolio para %s tras el trial.", ticker_stem
        )
        return {"sharpe": -1.0, "sortino": -1.0, "max_drawdown": 0.0, "volatility": 0.0}

    df = pd.read_csv(recent)
    metrics = df.set_index("metric")["value"].to_dict()
    return {
        "sharpe": float(metrics.get("sharpe_ratio", -1.0)),
        "sortino": float(metrics.get("sortino_ratio", -1.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "volatility": float(metrics.get("volatility", 0.0)),
    }


def _parse_risk_csv(results_dir: Path, ticker_stem: str, after_ts: float) -> float:
    """Retorna el VaR 95% medio del CSV de riesgo más reciente."""
    recent = _find_recent_result_file(
        results_dir,
        prefix="risk_metrics",
        ticker_stem=ticker_stem,
        after_ts=after_ts,
    )
    if recent is None:
        return 1.0  # valor centinela alto = penalización
    df = pd.read_csv(recent)
    return float(df["var_95"].mean()) if "var_95" in df.columns else 1.0


# ---------------------------------------------------------------------------
# Función objetivo
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial,
    csv_path: str,
    n_splits: int,
    results_dir: Path,
) -> float:
    """Entrena Flow_FEDformer con los hiperparámetros sugeridos y retorna Sharpe.

    Penalización compuesta:
        - Si VaR_95 > 0.08: Sharpe × 0.5 (riesgo excesivo)
        - Si Sortino < 0: Sharpe − 0.3 (sin señal asimétrica útil)

    Args:
        trial: Trial de Optuna.
        csv_path: Ruta al CSV del ticker.
        n_splits: Número de folds walk-forward.
        results_dir: Directorio de resultados para parsear CSVs.

    Returns:
        Sharpe penalizado, o -1.0 si el trial falló.
    """
    seq_len = trial.suggest_categorical("seq_len", SEQ_LENS)
    pred_len = trial.suggest_categorical("pred_len", PRED_LENS)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZES)
    clip_norm = trial.suggest_categorical("gradient_clip_norm", CLIP_NORMS)

    # Restricción estructural: el contexto debe ser al menos 3× la predicción
    # para que el encoder tenga suficiente señal temporal
    if seq_len < pred_len * 3:
        raise optuna.TrialPruned()

    ticker_stem = Path(csv_path).stem

    cmd = [
        sys.executable,
        "main.py",
        "--csv",
        csv_path,
        "--targets",
        "Close",
        "--seq-len",
        str(seq_len),
        "--pred-len",
        str(pred_len),
        "--batch-size",
        str(batch_size),
        "--splits",
        str(n_splits),
        "--return-transform",
        "log_return",
        "--metric-space",
        "returns",
        "--gradient-clip-norm",
        str(clip_norm),
        "--save-results",
        "--no-show",
        # Sin --save-canonical: los trials no tocan el model_registry
    ]

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    ts_before = _current_time()
    logger.info(
        "Trial %d | seq=%d pred=%d batch=%d clip=%.1f",
        trial.number,
        seq_len,
        pred_len,
        batch_size,
        clip_norm,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=900,  # 15 min máximo por trial
        )
        if proc.returncode != 0:
            logger.warning(
                "Trial %d falló (exit %d). stderr: %s",
                trial.number,
                proc.returncode,
                proc.stderr[-300:],
            )
            return -1.0
    except subprocess.TimeoutExpired:
        logger.warning("Trial %d expiró (>15 min).", trial.number)
        return -1.0

    portfolio = _parse_portfolio_csv(results_dir, ticker_stem, ts_before)
    var_95 = _parse_risk_csv(results_dir, ticker_stem, ts_before)

    sharpe = portfolio["sharpe"]
    sortino = portfolio["sortino"]

    logger.info(
        "Trial %d → Sharpe=%.4f Sortino=%.4f VaR=%.4f",
        trial.number,
        sharpe,
        sortino,
        var_95,
    )

    # Registrar métricas auxiliares para análisis post-hoc en el dashboard
    trial.set_user_attr("sortino", sortino)
    trial.set_user_attr("var_95", var_95)
    trial.set_user_attr("max_drawdown", portfolio["max_drawdown"])

    # Penalizaciones
    if var_95 > 0.08:
        sharpe *= 0.5
    if sortino < 0:
        sharpe -= 0.3

    return sharpe


# ---------------------------------------------------------------------------
# Descarga de tickers adicionales
# ---------------------------------------------------------------------------


def download_extra_tickers() -> None:
    """Descarga los 8 tickers adicionales para alcanzar los 12 activos.

    Usa financial_dataset_builder.py con --use_mock (yfinance real, 7 años).
    Omite los que ya existen en data/.
    """
    for ticker in EXTRA_TICKERS:
        dest = Path("data") / f"{ticker}_features.csv"
        if dest.exists():
            logger.info(
                "%-6s ya existe (%d filas) — omitiendo.", ticker, len(pd.read_csv(dest))
            )
            continue
        logger.info("Descargando %s...", ticker)
        proc = subprocess.run(
            [
                sys.executable,
                "data/financial_dataset_builder.py",
                "--symbol",
                ticker,
                "--use_mock",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "."},
        )
        if proc.returncode == 0:
            if not dest.exists():
                logger.error("%-6s finalizó sin generar %s", ticker, dest.as_posix())
                continue
            df = pd.read_csv(dest)
            logger.info("%-6s → %d filas × %d cols", ticker, len(df), df.shape[1])
        else:
            logger.error("%-6s falló: %s", ticker, proc.stderr[-200:])


# ---------------------------------------------------------------------------
# Entrenamiento final con los mejores parámetros
# ---------------------------------------------------------------------------


def _run_best_params(
    csv_path: str,
    best_params: dict,
    n_splits: int,
    save_canonical: bool,
) -> None:
    """Re-entrena con los mejores hiperparámetros encontrados por Optuna.

    Args:
        csv_path: Ruta al CSV del ticker.
        best_params: Dict con seq_len, pred_len, batch_size, gradient_clip_norm.
        n_splits: Número de folds walk-forward.
        save_canonical: Si True, registra en model_registry con --save-canonical.
    """
    cmd = [
        sys.executable,
        "main.py",
        "--csv",
        csv_path,
        "--targets",
        "Close",
        "--seq-len",
        str(best_params["seq_len"]),
        "--pred-len",
        str(best_params["pred_len"]),
        "--batch-size",
        str(best_params["batch_size"]),
        "--splits",
        str(n_splits),
        "--return-transform",
        "log_return",
        "--metric-space",
        "returns",
        "--gradient-clip-norm",
        str(best_params["gradient_clip_norm"]),
        "--save-results",
        "--no-show",
    ]
    if save_canonical:
        cmd.append("--save-canonical")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    logger.info("Ejecutando run final con mejores parámetros...")
    subprocess.run(cmd, env=env)


def _build_completed_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Construye el DataFrame de trials completados para reporting."""
    df_trials = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs")
    )
    if "state" not in df_trials.columns:
        return pd.DataFrame(columns=df_trials.columns)
    return df_trials[df_trials["state"] == "COMPLETE"].sort_values(
        "value", ascending=False
    )


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada del optimizador de hiperparámetros."""
    parser = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros con Optuna (por ticker, sin seed cruzado)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        help="Ruta al CSV del ticker a optimizar (ej. data/MSFT_features.csv).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Número de trials Optuna a ejecutar.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Walk-forward splits para cada trial.",
    )
    parser.add_argument(
        "--storage-path",
        default=None,
        help=(
            "Ruta SQLite para persistir el estudio y permitir reanudación. "
            "Ej: optuna_studies/msft.db. Si no se especifica, el estudio es in-memory."
        ),
    )
    parser.add_argument(
        "--best-save-canonical",
        action="store_true",
        help=(
            "Tras la búsqueda, re-entrenar con los mejores parámetros y registrar "
            "en model_registry con --save-canonical."
        ),
    )
    parser.add_argument(
        "--download-extra-tickers",
        action="store_true",
        help=f"Descarga los 8 tickers adicionales: {', '.join(EXTRA_TICKERS)}.",
    )
    args = parser.parse_args()

    # Modo descarga
    if args.download_extra_tickers:
        download_extra_tickers()
        return

    if not args.csv:
        parser.error("--csv es obligatorio salvo con --download-extra-tickers.")

    csv_path = args.csv
    ticker_stem = Path(csv_path).stem
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    study_name = f"tune_{ticker_stem}"

    # Configurar storage SQLite (reanudable) o in-memory
    storage = None
    if args.storage_path:
        storage_file = Path(args.storage_path)
        storage_file.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_file}"
        logger.info("Persistiendo estudio en: %s", storage_file)

    # TPE sampler: 5 trials exploratorios aleatorios antes de usar el modelo probabilístico
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,  # reanuda si el estudio SQLite ya existe
    )

    logger.info(
        "Iniciando búsqueda para %s — %d trials, espacio: %d combinaciones posibles.",
        ticker_stem,
        args.n_trials,
        len(SEQ_LENS) * len(PRED_LENS) * len(BATCH_SIZES) * len(CLIP_NORMS),
    )

    study.optimize(
        lambda trial: objective(trial, csv_path, args.n_splits, results_dir),
        n_trials=args.n_trials,
        show_progress_bar=False,  # interfiere con el logging estándar
    )

    # ---------------------------------------------------------------------------
    # Resumen de resultados
    # ---------------------------------------------------------------------------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    if not completed:
        logger.error("Ningún trial completado correctamente.")
        return

    best = study.best_trial

    logger.info("=" * 60)
    logger.info("RESULTADOS — %s", ticker_stem.upper())
    logger.info("  Trials completados: %d | Podados: %d", len(completed), len(pruned))
    logger.info("  Mejor Sharpe:       %.4f", best.value)
    logger.info("  Mejores parámetros:")
    for k, v in best.params.items():
        logger.info("    %-25s %s", k + ":", v)
    logger.info("  Métricas auxiliares del mejor trial:")
    logger.info("    Sortino:   %.4f", best.user_attrs.get("sortino", float("nan")))
    logger.info("    VaR_95:    %.4f", best.user_attrs.get("var_95", float("nan")))
    logger.info(
        "    MaxDD:     %.4f", best.user_attrs.get("max_drawdown", float("nan"))
    )
    logger.info("=" * 60)

    # Top-5 trials en tabla
    completed_df = _build_completed_trials_dataframe(study)
    cols_show = [
        c
        for c in completed_df.columns
        if "number" in c or "value" in c or "params_" in c or "user_attrs_sortino" in c
    ]
    print("\nTop 5 trials:")
    print(completed_df[cols_show].head(5).to_string(index=False))
    print()

    # Guardar resumen completo
    summary_dir = Path("optuna_studies")
    summary_dir.mkdir(exist_ok=True)
    summary_path = summary_dir / f"{ticker_stem}_trials.csv"
    completed_df.to_csv(summary_path, index=False)
    logger.info("Resumen de trials guardado en: %s", summary_path)

    # ---------------------------------------------------------------------------
    # Run final con los mejores parámetros (opcional)
    # ---------------------------------------------------------------------------
    if args.best_save_canonical:
        _run_best_params(
            csv_path=csv_path,
            best_params=best.params,
            n_splits=args.n_splits,
            save_canonical=True,
        )


if __name__ == "__main__":
    main()
