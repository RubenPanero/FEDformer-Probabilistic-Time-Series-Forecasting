"""Validación probabilística del modelo FEDformer sobre el CSV de inferencia.

Métricas calculadas:
  - Cobertura empírica p10-p90   (nominal: 80%)
  - Pinball loss en p10, p50, p90
  - MAE sobre p50
  - Interval Score 80% (Winkler)
  - Exactitud direccional en step=1

Uso:
    python3 validate_forecast.py \\
        --pred  results/predictions_NVDA.csv \\
        --features data/NVDA_features.csv \\
        --seq-len 96
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


# ── funciones de métricas ──────────────────────────────────────────────────────


def compute_coverage(gt: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Fracción de valores GT que caen dentro del intervalo [lower, upper]."""
    return float(((gt >= lower) & (gt <= upper)).mean())


def pinball_loss(gt: np.ndarray, q_hat: np.ndarray, q: float) -> float:
    """Pinball loss (quantile loss) en el cuantil q."""
    gt = np.asarray(gt, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)
    e = gt - q_hat
    return float(np.where(e >= 0, q * e, (q - 1) * e).mean())


def mae(gt: np.ndarray, p50: np.ndarray) -> float:
    """MAE sobre la predicción mediana p50."""
    return float(
        np.abs(np.asarray(gt, dtype=float) - np.asarray(p50, dtype=float)).mean()
    )


def interval_score(
    gt: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler Interval Score para un intervalo (1-alpha)*100 %.

    IS = (upper-lower) + (2/alpha)*max(lower-gt, 0) + (2/alpha)*max(gt-upper, 0)
    """
    gt = np.asarray(gt, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    penalty_low = (2 / alpha) * np.maximum(lower - gt, 0.0)
    penalty_high = (2 / alpha) * np.maximum(gt - upper, 0.0)
    return float((width + penalty_low + penalty_high).mean())


def directional_accuracy(
    gt_next: np.ndarray,
    p50_next: np.ndarray,
    last_price: np.ndarray,
) -> float:
    """Fracción de veces que el signo de (p50-last) coincide con el de (gt-last)."""
    gt_dir = np.sign(
        np.asarray(gt_next, dtype=float) - np.asarray(last_price, dtype=float)
    )
    pred_dir = np.sign(
        np.asarray(p50_next, dtype=float) - np.asarray(last_price, dtype=float)
    )
    return float((gt_dir == pred_dir).mean())


# ── función principal de cálculo ───────────────────────────────────────────────


def compute_all_metrics(
    pred_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target: str = "Close",
    seq_len: int = 96,
) -> dict[str, float]:
    """Calcula todas las métricas de validación.

    Args:
        pred_df:     DataFrame del CSV de inferencia exportado por la Inference API.
        features_df: DataFrame de features original (columna 'date' + target).
        target:      Nombre del target (default 'Close').
        seq_len:     Longitud de la secuencia de entrada del modelo.

    Returns:
        Diccionario con todas las métricas.
    """
    gt = pred_df[f"gt_{target}"]
    p10 = pred_df[f"p10_{target}"]
    p50 = pred_df[f"p50_{target}"]
    p90 = pred_df[f"p90_{target}"]

    # Exactitud direccional: primer step de cada ventana (puede ser 0 o 1)
    first_step = int(pred_df["step"].min())
    step1 = pred_df[pred_df["step"] == first_step].reset_index(drop=True)

    # En espacio de returns, la referencia para la dirección es 0 (return neutro).
    # En espacio de precios, es el último precio conocido de la secuencia de entrada.
    predictions_are_returns = gt.abs().max() < 5.0  # heurística: returns << precios
    if predictions_are_returns:
        last_prices = np.zeros(len(step1))
    else:
        windows_step1 = step1["window"].values
        last_prices = features_df[target].iloc[windows_step1 + seq_len - 1].values

    return {
        "coverage_p10_p90": compute_coverage(gt, p10, p90),
        "pinball_p10": pinball_loss(gt.values, p10.values, 0.10),
        "pinball_p50": pinball_loss(gt.values, p50.values, 0.50),
        "pinball_p90": pinball_loss(gt.values, p90.values, 0.90),
        "mae_p50": mae(gt.values, p50.values),
        "interval_score_80": interval_score(
            gt.values, p10.values, p90.values, alpha=0.20
        ),
        "directional_acc_step1": directional_accuracy(
            step1[f"gt_{target}"].values,
            step1[f"p50_{target}"].values,
            last_prices,
        ),
        "n_windows": int(pred_df["window"].nunique()),
        "n_obs": len(pred_df),
    }


# ── reporte ────────────────────────────────────────────────────────────────────


def print_report(metrics: dict[str, float], ticker: str, target: str) -> None:
    """Imprime el reporte de validación en formato legible."""
    cov = metrics["coverage_p10_p90"]
    print(f"\n{'=' * 55}")
    print(f"  Validación probabilística — {ticker} / {target}")
    print(f"{'=' * 55}")
    print(f"  Ventanas evaluadas : {metrics['n_windows']:,}")
    print(f"  Observaciones      : {metrics['n_obs']:,}")
    print(f"{'-' * 55}")
    print(f"  Cobertura p10-p90  : {cov:.1%}  (nominal: 80%)")
    gap = cov - 0.80
    tag = (
        "✓ calibrado"
        if abs(gap) < 0.05
        else ("▲ sobreestimado" if gap > 0 else "▼ subestimado")
    )
    print(f"    → gap vs nominal : {gap:+.1%}  {tag}")
    print(f"{'-' * 55}")
    print(f"  Pinball p10        : {metrics['pinball_p10']:.4f}")
    print(f"  Pinball p50        : {metrics['pinball_p50']:.4f}")
    print(f"  Pinball p90        : {metrics['pinball_p90']:.4f}")
    print(f"  MAE p50            : {metrics['mae_p50']:.4f}")
    print(f"  Interval Score 80% : {metrics['interval_score_80']:.4f}  (menor = mejor)")
    print(f"{'-' * 55}")
    print(f"  Exactitud direccional (step 1): {metrics['directional_acc_step1']:.1%}")
    print(f"{'=' * 55}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validación probabilística FEDformer")
    parser.add_argument(
        "--pred", required=True, help="CSV de predicciones (inference output)"
    )
    parser.add_argument("--features", required=True, help="CSV de features original")
    parser.add_argument(
        "--target", default="Close", help="Columna target (default: Close)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=96, help="seq_len del modelo (default: 96)"
    )
    parser.add_argument("--ticker", default="NVDA", help="Ticker para el reporte")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    pred_df = pd.read_csv(args.pred)
    features_df = pd.read_csv(args.features)

    metrics = compute_all_metrics(pred_df, features_df, args.target, args.seq_len)
    print_report(metrics, args.ticker, args.target)
