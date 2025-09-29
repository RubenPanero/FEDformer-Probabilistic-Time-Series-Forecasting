# -*- coding: utf-8 -*-
"""
Vanguard FEDformer: A Production-Ready Time Series Forecasting System (Modularizado).

Este script implementa la versión modularizada del FEDformer optimizado con critical bug fixes,
mejoras de rendimiento, y mejor mantenibilidad.

Uso:
1. Asegurar que todas las dependencias estén instaladas
2. Tener listo el dataset CSV
3. Ejecutar: python main.py --csv path/to/data.csv --targets col1,col2 [opciones]
"""

import argparse
import contextlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import wandb

from config import FEDformerConfig
from data import TimeSeriesDataset
from training import WalkForwardTrainer
from simulations import RiskSimulator, PortfolioSimulator
from utils import setup_cuda_optimizations, get_device
from utils.helpers import set_seed

# Setup global configurations
set_seed(42, deterministic=False)
setup_cuda_optimizations()
device = get_device()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationData:
    """Container for artifacts shared with visualization steps."""

    predictions: np.ndarray
    ground_truth: np.ndarray
    samples: np.ndarray
    dataset: TimeSeriesDataset


def _parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run optimized FEDformer with Normalizing Flows"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument(
        "--targets", required=True, help="Comma-separated target column names"
    )
    parser.add_argument("--date-col", default=None, help="Date/time column to exclude")
    parser.add_argument(
        "--wandb-project", default="vanguard-fedformer-flow", help="W&B project name"
    )
    parser.add_argument("--wandb-entity", default=None, help="W&B entity")
    parser.add_argument("--pred-len", type=int, default=24, help="Prediction horizon")
    parser.add_argument("--seq-len", type=int, default=96, help="Sequence length")
    parser.add_argument("--label-len", type=int, default=48, help="Label length")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per fold")
    parser.add_argument("--splits", type=int, default=5, help="Number of splits")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--use-checkpointing", action="store_true", help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="Enable deterministic mode (cuDNN)"
    )
    parser.add_argument(
        "--save-fig",
        default=None,
        help="Path to save generated figures instead of showing",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures (useful for headless)",
    )
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> List[str]:
    """Validates input arguments and returns target column names."""
    if not os.path.exists(args.csv):
        logger.error("Dataset not found at %s", args.csv)
        raise FileNotFoundError(f"Dataset not found at {args.csv}")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        logger.error("No valid targets provided")
        raise ValueError("No valid targets provided")
    return targets


def _create_config(args: argparse.Namespace, targets: List[str]) -> FEDformerConfig:
    """Creates and logs the FEDformer configuration."""
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
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        date_column=args.date_col,
        wandb_run_name=f"Modular-Flow-FEDformer_{int(time.time())}",
        seed=args.seed,
        deterministic=args.deterministic,
    )

    logger.info("Configuration validated successfully")
    logger.info(
        "Model parameters: d_model=%s, n_heads=%s",
        config.d_model,
        config.n_heads,
    )
    logger.info(
        "Training: epochs_per_fold=%s, batch_size=%s",
        config.n_epochs_per_fold,
        config.batch_size,
    )
    return config


def _load_dataset(config: FEDformerConfig) -> TimeSeriesDataset:
    """Loads and processes the time series dataset."""
    logger.info("Loading and processing dataset...")
    full_dataset = TimeSeriesDataset(config=config, flag="all")
    logger.info("Dataset loaded: %s samples", len(full_dataset))
    return full_dataset


def _run_backtest(
    config: FEDformerConfig, full_dataset: TimeSeriesDataset, splits: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executes the walk-forward backtest."""
    logger.info("Starting walk-forward backtest...")
    wf_trainer = WalkForwardTrainer(config, full_dataset)
    predictions_oos, ground_truth_oos, samples_oos = wf_trainer.run_backtest(
        n_splits=splits
    )

    if predictions_oos.size == 0:
        logger.error("No predictions generated. Exiting.")
        raise RuntimeError("No predictions generated from backtest.")

    logger.info("Backtest completed successfully")
    logger.info("Generated %s out-of-sample predictions", len(predictions_oos))
    return predictions_oos, ground_truth_oos, samples_oos


def _log_risk_summary(var: np.ndarray, cvar: np.ndarray) -> None:
    """Log aggregate risk statistics."""
    logger.info("Average VaR (95%%): %.4f", float(np.mean(var)))
    logger.info("Average CVaR (95%%): %.4f", float(np.mean(cvar)))


def _log_portfolio_metrics(metrics: Dict[str, Any]) -> None:
    """Log derived portfolio performance metrics."""
    logger.info("Portfolio Performance Metrics:")
    logger.info(
        "  Annualized Sharpe Ratio: %.3f", float(metrics.get("sharpe_ratio", 0.0))
    )
    logger.info(
        "  Maximum Drawdown: %.2f%%", float(metrics.get("max_drawdown", 0.0)) * 100
    )
    logger.info(
        "  Annualized Volatility: %.2f%%", float(metrics.get("volatility", 0.0)) * 100
    )
    logger.info("  Sortino Ratio: %.3f", float(metrics.get("sortino_ratio", 0.0)))


def _prepare_unscaled_series(
    data: SimulationData,
    config: FEDformerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale model predictions and ground truth back to original units."""
    scaler = getattr(data.dataset, "scaler", None)
    target_indices = getattr(data.dataset, "target_indices", None)
    if scaler is None or not target_indices:
        raise ValueError("Dataset scaler or target indices are missing.")

    target_idx = target_indices[0]

    dummy_preds = np.zeros(
        (data.predictions.shape[0], data.predictions.shape[1], config.enc_in)
    )
    dummy_preds[..., target_idx] = data.predictions[..., 0]
    unscaled_preds = scaler.inverse_transform(
        dummy_preds.reshape(-1, config.enc_in)
    ).reshape(dummy_preds.shape)[..., target_idx : target_idx + 1]

    dummy_gt = np.zeros(
        (data.ground_truth.shape[0], data.ground_truth.shape[1], config.enc_in)
    )
    dummy_gt[..., target_idx] = data.ground_truth[..., 0]
    unscaled_gt = scaler.inverse_transform(dummy_gt.reshape(-1, config.enc_in)).reshape(
        dummy_gt.shape
    )[..., target_idx : target_idx + 1]

    return unscaled_preds, unscaled_gt


def _create_portfolio_figure(
    metrics: Dict[str, Any],
    var: np.ndarray,
    cvar: np.ndarray,
) -> Figure:
    """Create matplotlib visualizations for portfolio and risk metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(
        metrics.get("cumulative_returns", np.array([0.0])),
        label="Strategy Returns",
        color="#1f77b4",
        linewidth=2,
    )
    ax1.set_title("Portfolio Strategy Performance", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Cumulative Returns")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    time_steps = range(var.shape[0])
    ax2.plot(
        time_steps, np.mean(var, axis=1), label="VaR (95%)", color="red", alpha=0.8
    )
    ax2.plot(
        time_steps,
        np.mean(cvar, axis=1),
        label="CVaR (95%)",
        color="darkred",
        alpha=0.8,
    )
    ax2.set_title("Risk Metrics Over Time", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Risk Value")
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
    """Log portfolio metrics to Weights & Biases when available."""
    if not wandb.run or not hasattr(wandb.run, "log"):
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
        logger.info("Metrics logged to W&B successfully")


def _handle_visualization_output(fig: Figure, args: argparse.Namespace) -> None:
    """Persist or display the generated figure based on CLI arguments."""
    if args.save_fig:
        try:
            fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
            logger.info("Figure saved to %s", args.save_fig)
        except OSError as exc:
            logger.warning("Saving figure failed: %s", exc)

    if not args.no_show:
        with contextlib.suppress(RuntimeError):
            plt.show()

    plt.close(fig)


def _run_portfolio_simulation(
    data: SimulationData,
    config: FEDformerConfig,
    risk_stats: Tuple[np.ndarray, np.ndarray],
) -> Tuple[Dict[str, Any], Figure]:
    """Execute portfolio simulation and return metrics with visualization."""
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
    """Run risk analysis, portfolio simulation, and visualization."""
    logger.info("Running risk and portfolio simulation...")

    risk_sim = RiskSimulator(data.samples)
    var = risk_sim.calculate_var()
    cvar = risk_sim.calculate_cvar()
    _log_risk_summary(var, cvar)

    if data.ground_truth.shape[1] <= 1:
        logger.info("Skipping portfolio simulation (single timestep prediction)")
        return

    try:
        metrics, fig = _run_portfolio_simulation(data, config, (var, cvar))
    except ValueError as exc:
        logger.warning("Portfolio simulation skipped: %s", exc)
        return

    _log_portfolio_metrics(metrics)
    _log_metrics_to_wandb(fig, metrics, var, cvar)
    _handle_visualization_output(fig, args)


def main() -> None:
    """Main function to run the FEDformer forecasting and simulation pipeline."""
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

        logger.info("Analysis completed successfully!")

    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception("Main execution failed")
        raise


if __name__ == "__main__":
    main()
