
import os
import time
import argparse
import logging
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import List, Tuple

from config import FEDformerConfig
from data import TimeSeriesDataset
from training import WalkForwardTrainer
from simulations import RiskSimulator, PortfolioSimulator
from utils import setup_cuda_optimizations, get_device
from utils.helpers import set_seed

logger = logging.getLogger(__name__)
device = get_device()

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
        logger.error(f"Dataset not found at {args.csv}")
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
        f"Model parameters: d_model={config.d_model}, n_heads={config.n_heads}"
    )
    logger.info(
        f"Training: epochs_per_fold={config.n_epochs_per_fold}, batch_size={config.batch_size}"
    )
    return config


def _load_dataset(config: FEDformerConfig) -> TimeSeriesDataset:
    """Loads and processes the time series dataset."""
    logger.info("Loading and processing dataset...")
    full_dataset = TimeSeriesDataset(config=config, flag="all")
    logger.info(f"Dataset loaded: {len(full_dataset)} samples")
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
    logger.info(f"Generated {len(predictions_oos)} out-of-sample predictions")
    return predictions_oos, ground_truth_oos, samples_oos


def _run_simulations_and_visualize(
    predictions_oos: np.ndarray,
    ground_truth_oos: np.ndarray,
    samples_oos: np.ndarray,
    full_dataset: TimeSeriesDataset,
    args: argparse.Namespace,
    config: FEDformerConfig,
) -> None:
    """Runs risk and portfolio simulations and handles visualization."""
    logger.info("Running risk and portfolio simulation...")

    risk_sim = RiskSimulator(samples_oos)
    var = risk_sim.calculate_var()
    cvar = risk_sim.calculate_cvar()

    logger.info(f"Average VaR (95%): {np.mean(var):.4f}")
    logger.info(f"Average CVaR (95%): {np.mean(cvar):.4f}")

    if ground_truth_oos.shape[1] > 1:
        try:
            scaler = full_dataset.scaler
            target_idx = full_dataset.target_indices[0]

            dummy_preds = np.zeros(
                (predictions_oos.shape[0], predictions_oos.shape[1], config.enc_in)
            )
            dummy_preds[..., target_idx] = predictions_oos[..., 0]
            unscaled_preds = scaler.inverse_transform(
                dummy_preds.reshape(-1, config.enc_in)
            ).reshape(dummy_preds.shape)[..., target_idx : target_idx + 1]

            dummy_gt = np.zeros(
                (
                    ground_truth_oos.shape[0],
                    ground_truth_oos.shape[1],
                    config.enc_in,
                )
            )
            dummy_gt[..., target_idx] = ground_truth_oos[..., 0]
            unscaled_gt = scaler.inverse_transform(
                dummy_gt.reshape(-1, config.enc_in)
            ).reshape(dummy_gt.shape)[..., target_idx : target_idx + 1]

            portfolio_sim = PortfolioSimulator(unscaled_preds, unscaled_gt)
            strategy_returns = portfolio_sim.run_simple_strategy()
            metrics = portfolio_sim.calculate_metrics(strategy_returns)

            logger.info("Portfolio Performance Metrics:")
            logger.info(f"  Annualized Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"  Annualized Volatility: {metrics['volatility']:.2%}")
            logger.info(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")

            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

                ax1.plot(
                    metrics["cumulative_returns"],
                    label="Strategy Returns",
                    color="#1f77b4",
                    linewidth=2,
                )
                ax1.set_title(
                    "Portfolio Strategy Performance", fontsize=14, fontweight="bold"
                )
                ax1.set_xlabel("Time Steps")
                ax1.set_ylabel("Cumulative Returns")
                ax1.grid(True, linestyle="--", alpha=0.6)
                ax1.legend()

                time_steps = range(var.shape[0])
                ax2.plot(
                    time_steps,
                    np.mean(var, axis=1),
                    label="VaR (95%)",
                    color="red",
                    alpha=0.8,
                )
                ax2.plot(
                    time_steps,
                    np.mean(cvar, axis=1),
                    label="CVaR (95%)",
                    color="darkred",
                    alpha=0.8,
                )
                ax2.set_title(
                    "Risk Metrics Over Time", fontsize=14, fontweight="bold"
                )
                ax2.set_xlabel("Time Steps")
                ax2.set_ylabel("Risk Value")
                ax2.grid(True, linestyle="--", alpha=0.6)
                ax2.legend()

                plt.tight_layout()

                with contextlib.suppress(Exception):
                    if wandb.run and hasattr(wandb.run, "log"):
                        wandb.run.log(
                            {
                                "sharpe_ratio": metrics["sharpe_ratio"],
                                "max_drawdown": metrics["max_drawdown"],
                                "volatility": metrics["volatility"],
                                "sortino_ratio": metrics["sortino_ratio"],
                                "avg_var": np.mean(var),
                                "avg_cvar": np.mean(cvar),
                                "performance_chart": wandb.Image(fig),
                            }
                        )
                        logger.info("Metrics logged to W&B successfully")

                if args.save_fig:
                    try:
                        fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
                        logger.info(f"Figure saved to {args.save_fig}")
                    except Exception as e:
                        logger.warning(f"Saving figure failed: {e}")
                if not args.no_show:
                    with contextlib.suppress(Exception):
                        plt.show()
                plt.close(fig)

            except Exception as e:
                logger.error(f"Visualization failed: {e}")

        except Exception as e:
            logger.error(f"Portfolio simulation failed: {e}")
    else:
        logger.info("Skipping portfolio simulation (single timestep prediction)")
