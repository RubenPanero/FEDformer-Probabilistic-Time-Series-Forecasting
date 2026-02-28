import argparse
import logging
import os
from pathlib import Path

from config import FEDformerConfig
from data.dataset import TimeSeriesDataset
from training.trainer import WalkForwardTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def finetune_sequence(
    symbols: list[str],
    base_checkpoint: str,
    output_dir: str = "checkpoints/finetuned",
    n_splits: int = 3,
):
    current_checkpoint = base_checkpoint

    if not os.path.exists(current_checkpoint):
        raise FileNotFoundError(f"El checkpoint base no existe: {current_checkpoint}")

    for symbol in symbols:
        logger.info(
            f"\n{'=' * 50}\nIniciando Fine-Tuning Secuencial para: {symbol}\n{'=' * 50}"
        )

        # 1. Preparar Dataset
        data_path = f"data/{symbol}_features.csv"
        if not os.path.exists(data_path):
            logger.info(f"Generando dataset financiero para {symbol}...")
            # Importación tardía para evitar dependencia de pandas_ta en módulos que no la necesitan
            from data.financial_dataset_builder import build_financial_dataset  # noqa: PLC0415

            build_financial_dataset(symbol, "data", use_mock=True)

        # 2. Configurar Entrenamiento
        # Se usará un LR más bajo para preservar el conocimiento (Continuous Learning)
        config = FEDformerConfig(
            target_features=["Close"],
            file_path=data_path,
            seq_len=96,
            label_len=48,
            pred_len=24,
            date_column="date",
            n_epochs_per_fold=3,  # Menos épocas (Fine-tuning rápido)
            batch_size=32,
            finetune_from=current_checkpoint,
            finetune_lr=1e-5,  # LR más bajo
            freeze_backbone=False,  # Se ajusta toda la red suavemente
            n_regimes=3,
            scaling_strategy="robust",  # Crítico para adaptar varianzas por acción
        )

        # 3. Datos y Entrenador
        dataset = TimeSeriesDataset(config, flag="all")
        trainer = WalkForwardTrainer(config, full_dataset=dataset)

        # Guardaremos el modelo temporalmente aislándolo por acción
        run_ckpt_dir = Path(output_dir) / symbol
        run_ckpt_dir.mkdir(exist_ok=True, parents=True)
        trainer.checkpoint_dir = run_ckpt_dir

        # 4. Iniciar Backtest / Fine-Tuning
        try:
            forecast = trainer.run_backtest(n_splits=n_splits)
            logger.info(
                f"Entrenamiento para {symbol} finalizado. Predicciones: {forecast.preds.shape}"
            )
        except Exception as e:
            logger.error(f"Fallo durante el finetuning de {symbol}: {e}")
            continue  # Intentar con el siguiente ticker

        # 5. Descubrir el nuevo mejor checkpoint guardado para propagarlo a la siguiente acción
        # WalkForwardTrainer guarda archivos del tipo "best_model_fold_X.pt".
        # Usaremos el último fold (índice dinámico basado en n_splits)
        last_fold_idx = n_splits - 1
        last_ckpt = trainer.checkpoint_dir / f"best_model_fold_{last_fold_idx}.pt"
        if last_ckpt.exists():
            current_checkpoint = str(last_ckpt)
            logger.info(
                f"Siguiente checkpoint base actualizado a: {current_checkpoint}"
            )
        else:
            fallback = sorted(
                trainer.checkpoint_dir.glob("*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            if fallback:
                current_checkpoint = str(fallback[-1])
                logger.info(f"Usando checkpoint alternativo: {current_checkpoint}")
            else:
                logger.warning(
                    "No se generó ningún checkpoint en este paso. Se reusará el anterior."
                )

    logger.info(
        f"Proceso de escalado secuencial completado. Checkpoint final maestro: {current_checkpoint}"
    )
    return current_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Escalar a S&P 500 secuencialmente")
    parser.add_argument(
        "--base_ckpt",
        type=str,
        required=True,
        help="Ruta al checkpoint de GOOGL o anterior",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["AAPL", "MSFT", "AMZN", "NVDA"],
        help="Símbolos a reentrenar",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints/finetuned",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Número de splits para walk-forward validation",
    )
    args = parser.parse_args()

    finetune_sequence(args.symbols, args.base_ckpt, args.out_dir, args.n_splits)
