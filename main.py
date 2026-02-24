# -*- coding: utf-8 -*-
"""
Vanguard FEDformer: Sistema Predictivo Estocástico Listo para Producción (Refactorizado).

Orquestador principal del proyecto que enlaza todos los sub-módulos y ejecuta
el ciclo completo de entrenamiento, evaluación y visualización empírica.

Uso:
1. Asegurar entorno activado (.venv)
2. Ejecutar: python main.py --csv <filepath> --targets <targets_separados> [opcionales]
"""

import argparse
import contextlib
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

from config import FEDformerConfig
from data import TimeSeriesDataset
from simulations import PortfolioSimulator, RiskSimulator
from training import WalkForwardTrainer
from utils import get_device, setup_cuda_optimizations
from utils.helpers import set_seed

# Consolidación inicial de determinismo e inicializaciones del clúster físico
set_seed(42, deterministic=False)
setup_cuda_optimizations()
device = get_device()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationData:
    """Contenedor puente de artefactos expuestos con la visualización matricial final."""

    predictions: np.ndarray
    ground_truth: np.ndarray
    samples: np.ndarray
    dataset: TimeSeriesDataset


def _parse_arguments() -> argparse.Namespace:
    """Consolida la inyección de comandos desde CLI de manera fuertemente tipada."""
    parser = argparse.ArgumentParser(
        description="Monitor Vanguard: Motor Algorítmico Flow FEDformer"
    )
    parser.add_argument("--csv", required=True, help="Ruta de acceso directa al CSV")
    parser.add_argument(
        "--targets",
        required=True,
        help="Lista csv concatenada por una coma representando features analíticas",
    )
    parser.add_argument(
        "--date-col",
        default=None,
        help="Índice de serie temporal a mitigar como feature",
    )
    parser.add_argument(
        "--wandb-project",
        default="vanguard-fedformer-flow",
        help="Cámara de telemetría W&B",
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="Organización receptora de métricas (W&B)"
    )
    parser.add_argument(
        "--pred-len", type=int, default=24, help="Ventana predictiva máxima inyectada"
    )
    parser.add_argument(
        "--seq-len", type=int, default=96, help="Tamaño de memoria de tensores empírica"
    )
    parser.add_argument(
        "--label-len", type=int, default=48, help="Token histórico decoder overlap"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Sub-ciclos epocales forzados en walk-forwards",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Fraccionamiento perimetral cross-fold (K)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Densidad iterativa escalar paralela (Batch Size)",
    )
    parser.add_argument(
        "--use-checkpointing",
        action="store_true",
        help="Dispara Gradient Checkpointing sacrificando CPU por VRAM",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Escalones virtuales permitidos para reventar gradientes",
    )
    parser.add_argument(
        "--finetune-from",
        default=None,
        help="Directorio relativo de memoria .pt para transbordo warm-start.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Inhibe el reajuste convolucional y se limita a redes normalizadoras exclusivas.",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Ratio paramétrico inespecífico de convergencia descendiente del warm-start.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hash natural anclado para evitar pseudo-azar",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Exige CUDNN bloqueante matemático purista (Muerte al Benchmark)",
    )
    parser.add_argument(
        "--save-fig",
        default=None,
        help="Directorio destino del render gráfico de retornos del portafolio",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Anula explícitamente despliegues de render X11 (Bloqueos en head-less server).",
    )
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> List[str]:
    """Asegura la coherencia básica del mapeo interactivo de usuario evitando fallas ruidosas posteriores."""
    if not os.path.exists(args.csv):
        logger.error("Dataset inaccesible o inexistente bajo: %s", args.csv)
        raise FileNotFoundError(f"Lectura imposible sobre {args.csv}")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        logger.error("Ninguna meta subyacente de análisis provista")
        raise ValueError("Indique al menos una variable predictiva en --targets")

    return targets


def _create_config(args: argparse.Namespace, targets: List[str]) -> FEDformerConfig:
    """Preinstala instancias de comportamiento dictadas externamente al manifest nativo."""
    config = FEDformerConfig(
        file_path=args.csv,
        target_features=targets,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        label_len=args.label_len,
        n_epochs_per_fold=args.epochs,
        batch_size=args.batch_size,
        use_gradient_checkpointing=args.use_checkpointing,
        gradient_accumulation_steps=args.grad_accum_steps,
        finetune_from=args.finetune_from,
        freeze_backbone=args.freeze_backbone,
        finetune_lr=args.finetune_lr,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        date_column=args.date_col,
        wandb_run_name=f"Modular-Flow-FEDformer_{int(time.time())}",
        seed=args.seed,
        deterministic=args.deterministic,
    )

    logger.info("Transmisión paramétrica asimilada de manera segura")
    logger.info(
        "Métricas inyectadas en topología: Dimensión Modelada=%s, Cabezas(Atención)=%s",
        config.d_model,
        config.n_heads,
    )
    logger.info(
        "Carga iterativa dispuesta: Ciclos(Fold)=%s, Tamaño de Bloque=%s",
        config.n_epochs_per_fold,
        config.batch_size,
    )
    return config


def _load_dataset(config: FEDformerConfig) -> TimeSeriesDataset:
    """Pre-computa el cargador matricial general y escaladores numéricos."""
    logger.info("Apertura de lecturas y transformaciones escalares de features...")
    full_dataset = TimeSeriesDataset(config=config, flag="all")
    logger.info(
        "Tubería de datos inyectada: %s ventanas transicionales analizadas",
        len(full_dataset),
    )
    return full_dataset


def _run_backtest(
    config: FEDformerConfig, full_dataset: TimeSeriesDataset, splits: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acciona el motor rotatorio de fraccionamiento evaluativo (Walk-forward)."""
    logger.info("Iniciando rampa de backtesting secuencial dinámico...")
    wf_trainer = WalkForwardTrainer(config, full_dataset)
    predictions_oos, ground_truth_oos, samples_oos = wf_trainer.run_backtest(
        n_splits=splits
    )

    if predictions_oos.size == 0:
        logger.error(
            "Imposible procesar sin simulaciones emitidas por la heurística evaluativa. Se fuerza salida."
        )
        raise RuntimeError(
            "Defecto numérico, colapso predictivo sin outputs del Walk-Forward."
        )

    logger.info(
        "Rutina del emulador inter-folds (Walk-Forward) completada brillantemente"
    )
    logger.info(
        "Receptado tensor de previsiones fuera-de-muestra (%s items predictivos)",
        len(predictions_oos),
    )
    return predictions_oos, ground_truth_oos, samples_oos


def _log_risk_summary(var: np.ndarray, cvar: np.ndarray) -> None:
    """Reporte superficial en bitácora de los tensores estadísticos caídos."""
    logger.info("Mitigador VaR (Valor en Riesgo) 95%% Medio: %.4f", float(np.mean(var)))
    logger.info(
        "Margen Severo CVaR (Riesgo Condicional Promedio) 95%%: %.4f",
        float(np.mean(cvar)),
    )


def _log_portfolio_metrics(metrics: Dict[str, Any]) -> None:
    """Transmite al estándar I/O de consola resumen técnico de capitales y ratio financiero."""
    logger.info("Bitácora de Desglose de Rendimientos Estructurales:")
    logger.info(
        "  Exceso de Beneficio por Volatilidad (Ratio Sharpe Anual): %.3f",
        float(metrics.get("sharpe_ratio", 0.0)),
    )
    logger.info(
        "  Hundimiento en el peor tramo predictivo (Max Drawdown): %.2f%%",
        float(metrics.get("max_drawdown", 0.0)) * 100,
    )
    logger.info(
        "  Flujo desviacional del portafolio (Volatilidad Anual): %.2f%%",
        float(metrics.get("volatility", 0.0)) * 100,
    )
    logger.info(
        "  Seguimiento asimétrico sin trampa bajista (Sortino Ratio): %.3f",
        float(metrics.get("sortino_ratio", 0.0)),
    )


def _prepare_unscaled_series(
    data: SimulationData,
    config: FEDformerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna los datos pre-escalados a su ambiente bursátil genuino usando la magia del transformado inverso."""
    preprocessor = getattr(data.dataset, "preprocessor", None)
    if preprocessor is not None and hasattr(preprocessor, "inverse_transform_targets"):
        unscaled_preds = preprocessor.inverse_transform_targets(
            data.predictions, config.target_features
        )
        unscaled_gt = preprocessor.inverse_transform_targets(
            data.ground_truth, config.target_features
        )
        return unscaled_preds, unscaled_gt

    scaler = getattr(data.dataset, "scaler", None)
    target_indices = getattr(data.dataset, "target_indices", None)
    if scaler is None or not target_indices:
        raise ValueError(
            "No pudimos localizar constructos asimilables en las directrices maestras (Scaler / Targets missing)."
        )

    # Validar que los índices estén dentro de rango
    if max(target_indices) >= config.enc_in:
        raise ValueError(
            f"target_indices contiene índices inválidos: máximo es {max(target_indices)}, "
            f"pero config.enc_in es {config.enc_in}"
        )

    # Garantizar que enc_in es un entero válido
    enc_in: int = int(config.enc_in) if config.enc_in is not None else 0
    if enc_in <= 0:
        raise ValueError("config.enc_in debe ser un valor positivo")

    dummy_preds = np.zeros(
        (data.predictions.shape[0], data.predictions.shape[1], enc_in)
    )
    for feature_idx, target_idx in enumerate(target_indices):
        dummy_preds[..., target_idx] = data.predictions[..., feature_idx]

    unscaled_preds = scaler.inverse_transform(dummy_preds.reshape(-1, enc_in)).reshape(
        dummy_preds.shape
    )[..., target_indices]

    dummy_gt = np.zeros(
        (data.ground_truth.shape[0], data.ground_truth.shape[1], enc_in)
    )
    for feature_idx, target_idx in enumerate(target_indices):
        dummy_gt[..., target_idx] = data.ground_truth[..., feature_idx]

    unscaled_gt = scaler.inverse_transform(dummy_gt.reshape(-1, enc_in)).reshape(
        dummy_gt.shape
    )[..., target_indices]

    return unscaled_preds, unscaled_gt


def _create_portfolio_figure(
    metrics: Dict[str, Any],
    var: np.ndarray,
    cvar: np.ndarray,
) -> Figure:
    """Engendra una topografía vectorizada renderizando mitigadores estocástico-comerciales."""
    if plt is None:
        raise RuntimeError(
            "Requerimiento denegado, entorno de dibujado (matplotlib) fue destruido o no se encontró."
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(
        metrics.get("cumulative_returns", np.array([0.0])),
        label="Acumulación Patrimonial",
        color="#1f77b4",
        linewidth=2,
    )
    ax1.set_title(
        "Evolución y Desempeño Operativo de Portafolio OoS",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Desfase Temporal Evaluado (Periodos)")
    ax1.set_ylabel("Margen Financiero Retornado Acumulado")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    time_steps = range(var.shape[0])
    ax2.plot(
        time_steps,
        np.mean(var, axis=1),
        label="Zona VaR Riesgo Crítico Promediado (95%)",
        color="red",
        alpha=0.8,
    )
    ax2.plot(
        time_steps,
        np.mean(cvar, axis=1),
        label="Derrumbe CVaR Excepcional Condicional (95%)",
        color="darkred",
        alpha=0.8,
    )
    ax2.set_title(
        "Examen Longitudinal Cíclico Sobre Severidad Estocástica",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Desfase Temporal Evaluado (Periodos)")
    ax2.set_ylabel("Magnitud Pérdida Muestral Proyectada")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    return fig


def _log_metrics_to_wandb(
    fig: Figure,
    metrics: Dict[str, Any],
    var: np.ndarray,
    cvar: np.ndarray,
) -> None:
    """Envía el clúster consolidado de variables hacia el entorno colaborativo Web de ser accesible."""
    if wandb is None:
        logger.debug(
            "Módulo wandb no está disponible, abortando sincronización de métricas."
        )
        return
    assert wandb is not None  # Para satisfacer type checkers
    if not wandb.run or not hasattr(wandb.run, "log"):
        logger.debug("Sesión activa de wandb no disponible, omitiendo log de métricas.")
        return

    with contextlib.suppress(RuntimeError, ValueError, AttributeError):
        wandb.run.log(
            {
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "volatility": float(metrics.get("volatility", 0.0)),
                "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
                "avg_var": float(np.mean(var)),
                "avg_cvar": float(np.mean(cvar)),
                "performance_chart": wandb.Image(fig),
            }
        )
        logger.info("Transmisión satisfactoria finalizada (W&B Metrics Synced)")


def _handle_visualization_output(fig: Figure, args: argparse.Namespace) -> None:
    """Discierne entre despliegue en host u opcionales exportados de render estático al disco duro."""
    if plt is None:
        return

    if args.save_fig:
        try:
            fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
            logger.info(
                "Artefacto Visual exportado a matriz estática sobre: %s", args.save_fig
            )
        except OSError as exc:
            logger.warning(
                "Caída súbita intentando forzar grabado sobre el binario visualizado: %s",
                exc,
            )

    if not args.no_show:
        with contextlib.suppress(RuntimeError):
            plt.show()

    plt.close(fig)


def _run_portfolio_simulation(
    data: SimulationData,
    config: FEDformerConfig,
    risk_stats: Tuple[np.ndarray, np.ndarray],
) -> Tuple[Dict[str, Any], Figure]:
    """Acciona el emulador lógico de trade-in uniendo lo predicho versus lo comprobado."""
    var, cvar = risk_stats
    unscaled_preds, unscaled_gt = _prepare_unscaled_series(data, config)

    portfolio_sim = PortfolioSimulator(unscaled_preds, unscaled_gt)
    strategy_returns = portfolio_sim.run_simple_strategy()

    metrics = portfolio_sim.calculate_metrics(strategy_returns)
    fig = _create_portfolio_figure(metrics, var, cvar)

    return metrics, fig


def _run_simulations_and_visualize(
    data: SimulationData,
    args: argparse.Namespace,
    config: FEDformerConfig,
) -> None:
    """Ejecuta los peritajes estadísticos marginales derivados de toda la secuencia matemática general."""
    logger.info(
        "Lanzando subsistemas analíticos (Validador Riesgo & Estrategia Portafolio)..."
    )

    risk_sim = RiskSimulator(data.samples)
    var = risk_sim.calculate_var()
    cvar = risk_sim.calculate_cvar()
    _log_risk_summary(var, cvar)

    if data.ground_truth.shape[1] <= 1:
        logger.info(
            "Omitiendo cálculos de simulación por límite infranqueable predictivo (Paso Ciego de TimeStep <= 1)"
        )
        return

    try:
        metrics, fig = _run_portfolio_simulation(data, config, (var, cvar))
    except ValueError as exc:
        logger.warning(
            "Evaluador financiero corrompido, bloque ignorado preventivamente: %s", exc
        )
        return

    _log_portfolio_metrics(metrics)
    _log_metrics_to_wandb(fig, metrics, var, cvar)
    _handle_visualization_output(fig, args)


def main() -> None:
    """Nodo central asimilador operando los subsistemas acoplados iterativos."""
    try:
        args = _parse_arguments()
        set_seed(args.seed, deterministic=args.deterministic)

        targets = _validate_inputs(args)
        config = _create_config(args, targets)

        full_dataset = _load_dataset(config)

        predictions_oos, ground_truth_oos, samples_oos = _run_backtest(
            config, full_dataset, args.splits
        )

        sim_data = SimulationData(
            predictions=predictions_oos,
            ground_truth=ground_truth_oos,
            samples=samples_oos,
            dataset=full_dataset,
        )

        _run_simulations_and_visualize(sim_data, args, config)

        logger.info(
            "Validación, Entrenamiento e Inferencias resueltas triunfalmente. Flujo terminado."
        )

    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception(
            "Secuencia destructiva abordó el Thread Main general, paralizando arquitectura."
        )
        raise


if __name__ == "__main__":
    main()
