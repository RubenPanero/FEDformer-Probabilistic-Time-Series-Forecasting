# -*- coding: utf-8 -*-
"""
Sistema de entrenamiento walk-forward para el modelo FEDformer.
"""

import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
import wandb
from wandb.errors import Error as WandbError

from config import FEDformerConfig
from data import TimeSeriesDataset
from models import Flow_FEDformer
from utils import MetricsTracker, get_device
from training.utils import mc_dropout_inference

logger = logging.getLogger(__name__)
device = get_device()


@dataclass(frozen=True)
class TrainingComponents:
    """Aggregates model, optimizer, scaler, and loaders for a training fold."""

    model: Flow_FEDformer
    optimizer: Optimizer
    scaler: Optional[GradScaler]
    train_loader: DataLoader
    test_loader: DataLoader
    fold: int


@dataclass(frozen=True)
class BatchTensors:
    """Batch tensors transferred to the target device."""

    encoder: torch.Tensor
    decoder: torch.Tensor
    target: torch.Tensor
    regime: torch.Tensor


class WalkForwardTrainer:
    """Enhanced walk-forward trainer with better error handling and monitoring."""

    def __init__(
        self, config: FEDformerConfig, full_dataset: TimeSeriesDataset
    ) -> None:
        self.config = config
        self.full_dataset = full_dataset
        self.wandb_run = None
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _get_model(self) -> Flow_FEDformer:
        """Create and optionally compile model."""
        try:
            model = Flow_FEDformer(self.config).to(device, non_blocking=True)
            if (
                self.config.compile_mode
                and device.type == "cuda"
                and hasattr(torch, "compile")
            ):
                logger.info("Compiling model with mode: %s", self.config.compile_mode)
                return torch.compile(model, mode=self.config.compile_mode)
            return model
        except (RuntimeError, TypeError) as exc:
            logger.warning(
                "Model compilation failed (%s). Using uncompiled model.",
                exc,
            )
            return Flow_FEDformer(self.config).to(device, non_blocking=True)

    def _prepare_data_loaders(
        self, train_subset: Subset, test_subset: Subset
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare and return train and test DataLoaders."""
        num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
        generator = torch.Generator()
        generator.manual_seed(self.config.__dict__.get("seed", 42))

        def _worker_init_fn(worker_id: int) -> None:
            base_seed = self.config.__dict__.get("seed", 42)
            np.random.seed(base_seed + worker_id)
            torch.manual_seed(base_seed + worker_id)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=_worker_init_fn,
            generator=generator,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=_worker_init_fn,
            generator=generator,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
        return train_loader, test_loader

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> BatchTensors:
        """Move batch tensors to the configured device."""
        return BatchTensors(
            encoder=batch["x_enc"].to(device, non_blocking=True),
            decoder=batch["x_dec"].to(device, non_blocking=True),
            target=batch["y_true"].to(device, non_blocking=True),
            regime=batch["x_regime"].to(device, non_blocking=True),
        )

    def _forward_and_compute_loss(
        self,
        model: Flow_FEDformer,
        tensors: BatchTensors,
        scaler: Optional[GradScaler],
        accumulation_steps: int,
    ) -> Optional[torch.Tensor]:
        """Run forward pass and scale loss for gradient accumulation."""
        enabled = scaler.is_enabled() if scaler else False
        with autocast(enabled=enabled):
            dist = model(tensors.encoder, tensors.decoder, tensors.regime)
            loss = self._nll_loss(dist, tensors.target) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Invalid loss detected. Skipping batch.")
            return None

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss

    @staticmethod
    def _should_step(
        batch_idx: int, total_batches: int, accumulation_steps: int
    ) -> bool:
        """Determine whether to perform an optimizer step."""
        return (batch_idx + 1) % accumulation_steps == 0 or (
            batch_idx + 1
        ) == total_batches

    @staticmethod
    def _optimizer_step(
        optimizer: Optimizer,
        scaler: Optional[GradScaler],
        model: nn.Module,
    ) -> None:
        """Apply optimizer step with optional scaler logic."""
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def _build_training_components(
        self, train_subset: Subset, test_subset: Subset, fold_idx: int
    ) -> TrainingComponents:
        """Instantiate model, optimizer, scaler and loaders for a fold."""
        train_loader, test_loader = self._prepare_data_loaders(
            train_subset, test_subset
        )
        model = self._get_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )
        scaler = (
            GradScaler(enabled=self.config.use_amp and device.type == "cuda")
            if self.config.use_amp
            else None
        )
        return TrainingComponents(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            test_loader=test_loader,
            fold=fold_idx,
        )

    def save_checkpoint(
        self,
        components: TrainingComponents,
        epoch: int,
        loss: float,
        best: bool = False,
    ) -> Path:
        """Save model checkpoint for fault tolerance."""
        checkpoint = {
            "model_state_dict": components.model.state_dict(),
            "optimizer_state_dict": components.optimizer.state_dict(),
            "scaler_state_dict": components.scaler.state_dict()
            if components.scaler
            else None,
            "epoch": epoch,
            "fold": components.fold,
            "loss": loss,
            "config": asdict(self.config),
        }

        if best:
            path = self.checkpoint_dir / f"best_model_fold_{components.fold}.pt"
        else:
            path = self.checkpoint_dir / (
                f"checkpoint_fold_{components.fold}_epoch_{epoch}.pt"
            )

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)
        return path

    def load_checkpoint(
        self,
        model: Flow_FEDformer,
        optimizer: Optimizer,
        scaler: Optional[GradScaler],
        checkpoint_path: str,
    ) -> Tuple[int, int, float]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("Checkpoint loaded from %s", checkpoint_path)
        return checkpoint["epoch"], checkpoint["fold"], checkpoint["loss"]

    def _nll_loss(self, dist: Distribution, y_true: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss with numerical stability."""
        try:
            log_prob = dist.log_prob(y_true)
            log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)
            return -log_prob.mean()
        except (RuntimeError, ValueError) as exc:
            logger.warning("Loss calculation failed: %s. Using MSE fallback.", exc)
            return F.mse_loss(dist.mean, y_true)

    def _initialize_wandb(self) -> None:
        """Initializes Weights & Biases run."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=self.config.wandb_run_name,
                reinit=True,
            )
            logger.info("W&B initialization successful")
        except (WandbError, ValueError) as exc:
            logger.warning(
                "W&B initialization failed: %s. Continuing without logging.",
                exc,
            )
            self.wandb_run = None

    def _train_epoch(self, components: TrainingComponents, epoch: int) -> float:
        """Handle the training loop for a single epoch."""
        components.model.train()
        epoch_losses = []
        accumulation_steps = self.config.gradient_accumulation_steps
        total_batches = len(components.train_loader)

        for batch_idx, batch in enumerate(components.train_loader):
            try:
                tensors = self._prepare_batch(batch)
                loss = self._forward_and_compute_loss(
                    components.model,
                    tensors,
                    components.scaler,
                    accumulation_steps,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.error(
                        "GPU OOM encountered. Consider reducing batch size or enabling gradient checkpointing."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                logger.warning(
                    "Batch %s failed on fold %s: %s. Continuing.",
                    batch_idx,
                    components.fold,
                    exc,
                )
                continue

            if loss is None:
                continue

            loss_value = loss.item() * accumulation_steps

            if self._should_step(batch_idx, total_batches, accumulation_steps):
                self._optimizer_step(
                    components.optimizer, components.scaler, components.model
                )

            epoch_losses.append(loss_value)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        logger.info(
            "  Epoch %s/%s, Avg Loss: %.4f",
            epoch + 1,
            self.config.n_epochs_per_fold,
            avg_loss,
        )
        self.metrics_tracker.log_metrics({"train_loss": avg_loss}, epoch)
        return avg_loss

    def _evaluate_model(
        self, model: Flow_FEDformer, test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle the evaluation loop for a single fold."""
        model.eval()
        fold_preds, fold_gt, fold_samples = [], [], []

        with torch.inference_mode():
            for batch in test_loader:
                try:
                    samples = mc_dropout_inference(
                        model, batch, n_samples=50, use_flow_sampling=True
                    )
                    fold_samples.append(samples.cpu().numpy())
                    fold_preds.append(torch.median(samples, dim=0)[0].cpu().numpy())
                    fold_gt.append(batch["y_true"].cpu().numpy())
                except (RuntimeError, ValueError) as exc:
                    logger.warning("Evaluation batch failed: %s", exc)
                    continue

        if fold_preds and fold_gt and fold_samples:
            return (
                np.concatenate(fold_preds, axis=0),
                np.concatenate(fold_gt, axis=0),
                np.concatenate(fold_samples, axis=1),
            )

        logger.warning("No valid predictions for this evaluation.")
        return np.array([]), np.array([]), np.array([])

    def _run_single_fold(  # pylint: disable=too-many-locals
        self,
        fold_idx: int,
        split_size: int,
        total_size: int,
        total_folds: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Train and evaluate the model for a single walk-forward fold."""
        train_end_idx = fold_idx * split_size
        test_end_idx = min((fold_idx + 1) * split_size, total_size)

        if test_end_idx - train_end_idx < self.config.seq_len + self.config.pred_len:
            logger.warning("Insufficient data for fold %s. Skipping.", fold_idx)
            return None

        logger.info(
            "--- Fold %s/%s: Training on [0, %s], Testing on [%s, %s] ---",
            fold_idx,
            total_folds,
            train_end_idx,
            train_end_idx,
            test_end_idx,
        )

        train_subset = Subset(
            self.full_dataset, range(min(train_end_idx, len(self.full_dataset)))
        )
        test_indices = range(train_end_idx, min(test_end_idx, len(self.full_dataset)))
        test_subset = Subset(self.full_dataset, list(test_indices))

        components = self._build_training_components(
            train_subset, test_subset, fold_idx
        )

        best_loss = float("inf")
        for epoch in range(self.config.n_epochs_per_fold):
            avg_loss = self._train_epoch(components, epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(components, epoch, avg_loss, best=True)

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(components, epoch, avg_loss, best=False)

            if self.wandb_run:
                self.wandb_run.log(
                    {"train_loss": avg_loss, "epoch": epoch, "fold": fold_idx}
                )

        fold_preds, fold_gt, fold_samples = self._evaluate_model(
            components.model, components.test_loader
        )

        if fold_preds.size == 0:
            logger.warning("No valid predictions for fold %s", fold_idx)
            return None

        if self.wandb_run:
            self.wandb_run.log({"fold": fold_idx, "fold_completed": True})

        return fold_preds, fold_gt, fold_samples

    def run_backtest(
        self, n_splits: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced backtest with comprehensive error handling."""
        self._initialize_wandb()

        total_size = len(self.full_dataset)
        split_size = max(
            total_size // n_splits, self.config.seq_len + self.config.pred_len
        )
        total_folds = max(1, n_splits - 1)

        all_preds, all_gt, all_samples = [], [], []

        try:
            for fold_idx in range(1, n_splits):
                fold_outputs = self._run_single_fold(
                    fold_idx, split_size, total_size, total_folds
                )
                if not fold_outputs:
                    continue

                preds, gt, samples = fold_outputs
                all_preds.append(preds)
                all_gt.append(gt)
                all_samples.append(samples)
        except (RuntimeError, ValueError):
            logger.exception("Backtest failed")
            raise
        finally:
            if self.wandb_run:
                self.wandb_run.finish()

        if not all_preds:
            logger.error("No successful predictions from any fold")
            return np.array([]), np.array([]), np.array([])

        return (
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_gt, axis=0),
            np.concatenate(all_samples, axis=1),
        )
