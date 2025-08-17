# -*- coding: utf-8 -*-
"""
Sistema de entrenamiento walk-forward para el modelo FEDformer.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataclasses import asdict
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from config import FEDformerConfig
from data import TimeSeriesDataset
from models import Flow_FEDformer
from utils import MetricsTracker, get_device
from training.utils import mc_dropout_inference

logger = logging.getLogger(__name__)
device = get_device()


class WalkForwardTrainer:
    """Enhanced walk-forward trainer with better error handling and monitoring"""
    def __init__(self, config: FEDformerConfig, full_dataset: TimeSeriesDataset):
        self.config = config
        self.full_dataset = full_dataset
        self.wandb_run = None
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _get_model(self):
        """Create and optionally compile model"""
        try:
            model = Flow_FEDformer(self.config).to(device, non_blocking=True)
            if self.config.compile_mode and device.type == 'cuda' and hasattr(torch, 'compile'):
                logger.info(f"Compiling model with mode: {self.config.compile_mode}")
                return torch.compile(model, mode=self.config.compile_mode)
            return model
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Using uncompiled model.")
            return Flow_FEDformer(self.config).to(device, non_blocking=True)
    
    def save_checkpoint(self, model, optimizer, scaler, epoch, fold, loss, best=False):
        """Save model checkpoint for fault tolerance"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'fold': fold,
            'loss': loss,
            'config': asdict(self.config)
        }
        
        if best:
            path = self.checkpoint_dir / f'best_model_fold_{fold}.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_fold_{fold}_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        return path
    
    def load_checkpoint(self, model, optimizer, scaler, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['fold'], checkpoint['loss']

    def _nll_loss(self, dist, y_true):
        """Negative log-likelihood loss with numerical stability"""
        try:
            log_prob = dist.log_prob(y_true)
            # Add numerical stability
            log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)
            return -log_prob.mean()
        except Exception as e:
            logger.warning(f"Loss calculation failed: {e}. Using MSE fallback.")
            return F.mse_loss(dist.mean, y_true)

    def run_backtest(self, n_splits=5):
        """Enhanced backtest with comprehensive error handling"""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=self.config.wandb_run_name,
                reinit=True
            )
            logger.info("W&B initialization successful")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}. Continuing without logging.")
            self.wandb_run = None
        
        total_size = len(self.full_dataset)
        split_size = max(total_size // n_splits, self.config.seq_len + self.config.pred_len)
        all_preds, all_gt, all_samples = [], [], []

        try:
            for i in range(1, n_splits):
                train_end_idx = i * split_size
                test_end_idx = min((i + 1) * split_size, total_size)
                
                if test_end_idx - train_end_idx < self.config.seq_len + self.config.pred_len:
                    logger.warning(f"Insufficient data for fold {i}. Skipping.")
                    continue
                    
                logger.info(f"--- Fold {i}/{n_splits-1}: Training on [0, {train_end_idx}], Testing on [{train_end_idx}, {test_end_idx}] ---")

                train_subset = Subset(self.full_dataset, range(min(train_end_idx, len(self.full_dataset))))
                test_indices = range(train_end_idx, min(test_end_idx, len(self.full_dataset)))
                test_subset = Subset(self.full_dataset, list(test_indices))
                
                # OPTIMIZED: Enhanced data loading with prefetch_factor
                num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
                train_loader = DataLoader(
                    train_subset, 
                    batch_size=self.config.batch_size,
                    shuffle=True, 
                    drop_last=True,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=num_workers > 0,
                    **({'prefetch_factor': 2} if num_workers > 0 else {})  # OPTIMIZED: Better GPU utilization
                )
                test_loader = DataLoader(
                    test_subset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=num_workers > 0,
                    **({'prefetch_factor': 2} if num_workers > 0 else {})  # OPTIMIZED: Better GPU utilization
                )

                model = self._get_model()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    eps=1e-8
                )
                scaler = GradScaler(enabled=self.config.use_amp and device.type == 'cuda')

                # OPTIMIZED: Training loop with gradient accumulation and checkpointing
                best_loss = float('inf')
                for epoch in range(self.config.n_epochs_per_fold):
                    model.train()
                    epoch_losses = []
                    accumulation_steps = self.config.gradient_accumulation_steps
                    
                    try:
                        for batch_idx, batch in enumerate(train_loader):
                            try:
                                x_enc = batch['x_enc'].to(device, non_blocking=True)
                                x_dec = batch['x_dec'].to(device, non_blocking=True)  
                                y_true = batch['y_true'].to(device, non_blocking=True)
                                x_regime = batch['x_regime'].to(device, non_blocking=True)
                                
                                with autocast(enabled=scaler.is_enabled()):
                                    dist = model(x_enc, x_dec, x_regime)
                                    loss = self._nll_loss(dist, y_true)
                                    loss = loss / accumulation_steps  # Scale loss for accumulation
                                
                                if torch.isnan(loss) or torch.isinf(loss):
                                    logger.warning(f"Invalid loss detected: {loss.item()}. Skipping batch.")
                                    continue
                                
                                scaler.scale(loss).backward()
                                
                                # OPTIMIZED: Gradient accumulation
                                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                                    # Gradient clipping
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad(set_to_none=True)
                                
                                epoch_losses.append(loss.item() * accumulation_steps)  # Unscale for logging
                                
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    logger.error("GPU OOM. Try reducing batch_size or enabling gradient_checkpointing")
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    raise
                                else:
                                    logger.warning(f"Batch {batch_idx} failed: {e}. Continuing.")
                                    continue
                                    
                        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                        logger.info(f"  Epoch {epoch+1}/{self.config.n_epochs_per_fold}, Avg Loss: {avg_loss:.4f}")
                        
                        self.metrics_tracker.log_metrics({'train_loss': avg_loss}, epoch)
                        
                        # Save checkpoint if best loss
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            self.save_checkpoint(model, optimizer, scaler, epoch, i, avg_loss, best=True)
                        
                        # Save regular checkpoint every 5 epochs
                        if (epoch + 1) % 5 == 0:
                            self.save_checkpoint(model, optimizer, scaler, epoch, i, avg_loss, best=False)
                        
                        if self.wandb_run:
                            self.wandb_run.log({'train_loss': avg_loss, 'epoch': epoch, 'fold': i})
                            
                    except Exception as e:
                        logger.error(f"Epoch {epoch} failed: {e}")
                        continue

                # Evaluation with error handling
                model.eval()
                fold_preds, fold_gt, fold_samples = [], [], []
                
                try:
                    for batch in test_loader:
                        try:
                            samples = mc_dropout_inference(model, batch, n_samples=50, use_flow_sampling=True)
                            fold_samples.append(samples.cpu().numpy())
                            fold_preds.append(torch.median(samples, dim=0)[0].cpu().numpy())
                            fold_gt.append(batch['y_true'].cpu().numpy())
                        except Exception as e:
                            logger.warning(f"Evaluation batch failed: {e}")
                            continue
                    
                    if fold_preds and fold_gt and fold_samples:
                        all_preds.append(np.concatenate(fold_preds, axis=0))
                        all_gt.append(np.concatenate(fold_gt, axis=0))
                        all_samples.append(np.concatenate(fold_samples, axis=1))
                        
                        if self.wandb_run:
                            self.wandb_run.log({'fold': i, 'fold_completed': True})
                    else:
                        logger.warning(f"No valid predictions for fold {i}")
                        
                except Exception as e:
                    logger.error(f"Evaluation failed for fold {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
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
            np.concatenate(all_samples, axis=1)
        )

