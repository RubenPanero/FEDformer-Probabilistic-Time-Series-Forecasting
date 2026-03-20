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

from inference.loader import available_tickers, load_specialist
from inference.predictor import predict
from utils.model_registry import DEFAULT_REGISTRY_PATH

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia probabilística con modelos canónicos FEDformer",
        prog="python3 -m inference",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Símbolo del ticker (ej. NVDA, GOOGL)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Ruta al CSV con datos para predecir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del CSV de salida (default: results/inference_{ticker}.csv)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Número de muestras MC Dropout (default: 50)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Ruta al model_registry.json (default: checkpoints/model_registry.json)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
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

    output_path = Path(args.output or f"results/inference_{ticker.lower()}.csv")
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
        print(
            f"  Media cuantiles — p10: {p10_mean:.4f}  p50: {p50_mean:.4f}  p90: {p90_mean:.4f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
