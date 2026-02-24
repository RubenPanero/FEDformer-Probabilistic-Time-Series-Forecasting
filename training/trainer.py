# -*- coding: utf-8 -*-
"""
Sistema de entrenamiento walk-forward para el modelo FEDformer.
Refactorizado a Python 3.10+ para garantizar typing nativo purificado, tolerancia a fallos, y eficiencia PEP 8.
"""

import gc
import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import autocast
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    from wandb.errors import Error as WandbError
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore
    WandbError = Exception

from config import FEDformerConfig
from data import TimeSeriesDataset
from models import Flow_FEDformer
from training.utils import mc_dropout_inference
from utils import MetricsTracker, get_device

logger = logging.getLogger(__name__)
device = get_device()


@dataclass(frozen=True)
class TrainingComponents:
    """Consolida el modelo, optimizador, escalador y cargadores (loaders) para iterar en entrenamiento."""

    model: Flow_FEDformer
    optimizer: Optimizer
    scaler: torch.amp.GradScaler | None
    train_loader: DataLoader
    test_loader: DataLoader
    fold: int


@dataclass(frozen=True)
class BatchTensors:
    """Estructura de tensores en bloque transbordados al dispositivo físico."""

    encoder: torch.Tensor
    decoder: torch.Tensor
    target: torch.Tensor
    regime: torch.Tensor


class WalkForwardTrainer:
    """Monitor de entrenamiento avanzado walk-forward optimizado y a prueba del GC."""

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
        """Crea y preferiblemente compila el submodelo instanciado."""
        try:
            model = Flow_FEDformer(self.config).to(device, non_blocking=True)
            if self.config.finetune_from or self.config.freeze_backbone:
                return model
            if (
                self.config.compile_mode
                and device.type == "cuda"
                and hasattr(torch, "compile")
            ):
                logger.info(
                    "Compilando el modelo con modo dinámico: %s",
                    self.config.compile_mode,
                )
                return torch.compile(model, mode=self.config.compile_mode)
            return model
        except (RuntimeError, TypeError) as exc:
            logger.warning(
                "Fallo durante invocación compilada del modelo de PyTorch (%s). Transicionando a fallback uncompiled.",
                exc,
            )
            return Flow_FEDformer(self.config).to(device, non_blocking=True)

    def _prepare_data_loaders(
        self, train_subset: Subset, test_subset: Subset
    ) -> tuple[DataLoader, DataLoader]:
        """Prepara y devuelve los cargadores en memoria DataLoaders purificados."""
        num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        def _worker_init_fn(worker_id: int) -> None:
            base_seed = self.config.seed
            np.random.seed(base_seed + worker_id)
            torch.manual_seed(base_seed + worker_id)

        # Evitando memory leaks sobre workers persistentes reseteando la sesión adecuadamente
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,  # Explicitely false per safe memory policy between folds
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
            persistent_workers=False,  # Safe parallel evaluation
            worker_init_fn=_worker_init_fn,
            generator=generator,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
        return train_loader, test_loader

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> BatchTensors:
        """Transborda el bloque de tensores al dispositivo hardware principal."""
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
        scaler: torch.amp.GradScaler | None,
        accumulation_steps: int,
    ) -> torch.Tensor | None:
        """Rutina unificada para saltos (forwards) penalizados para auto-retorno acumulativo."""
        enabled = scaler.is_enabled() if scaler else False
        with autocast(device_type=device.type, enabled=enabled):
            dist = model(tensors.encoder, tensors.decoder, tensors.regime)
            loss = self._nll_loss(dist, tensors.target) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "Caída de pérdida (loss) detectada a infinito o vacío (NaN). Omitiendo inferencia en bloque."
            )
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
        """Verifica estáticamente cuándo los acumuladores exigen salto optimizador."""
        return (batch_idx + 1) % accumulation_steps == 0 or (
            batch_idx + 1
        ) == total_batches

    @staticmethod
    def _optimizer_step(
        optimizer: Optimizer,
        scaler: torch.amp.GradScaler | None,
        model: nn.Module,
    ) -> None:
        """Asienta saltos en optimizer, validando clípeos dinámicos en pesos (Scale steps)."""
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
        """Instancia un bloque paramétrico ciego de modelo y loaders preajustados."""
        train_loader, test_loader = self._prepare_data_loaders(
            train_subset, test_subset
        )
        model = self._get_model()
        self._maybe_load_finetune_checkpoint(model)

        if self.config.freeze_backbone:
            self._apply_freeze_backbone(model)

        lr = (
            self.config.finetune_lr
            if self.config.finetune_from and self.config.finetune_lr is not None
            else self.config.learning_rate
        )

        trainable_params = list(self._trainable_params(model))
        if not trainable_params:
            raise RuntimeError(
                "Excepción bloqueante: No hay parámetros en el modelo que reescribir con optimizador."
            )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )

        scaler = (
            torch.amp.GradScaler(
                "cuda", enabled=self.config.use_amp and device.type == "cuda"
            )
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

    def _maybe_load_finetune_checkpoint(self, model: Flow_FEDformer) -> None:
        """Rutina warm-up de subpesos extraída desde los dicts de persistencia."""
        ckpt_path = self.config.finetune_from
        if not ckpt_path:
            return

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Archivo de warm-start no encontrado en el sistema: {ckpt_path}"
            )

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Pesos ajustados precargados mediante finetuning de %s (Perdidos=%d, Inesperados=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )

    def _apply_freeze_backbone(self, model: Flow_FEDformer) -> None:
        """Congela componentes estructurales preservando conectivas activas."""
        for param in model.parameters():
            param.requires_grad = False

        for flow in model.flows:
            for param in flow.parameters():
                param.requires_grad = True

        trainable_component_keys = {
            "flow_conditioner_proj",
            "enc_embedding",
            "dec_embedding",
            "regime_embedding",
        }
        for key, module in model.components.items():
            if key in trainable_component_keys:
                for param in module.parameters():
                    param.requires_grad = True

        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True

    @staticmethod
    def _trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
        """Itera parámetros que no fueron encapsulados intencionalmente estáticos."""
        for param in model.parameters():
            if param.requires_grad:
                yield param

    def save_checkpoint(
        self,
        components: TrainingComponents,
        epoch: int,
        loss: float,
        best: bool = False,
    ) -> Path:
        """Punto de auto-restauración atómica tras cada cierre heurístico favorable."""
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
        logger.info("Persistencia matemática salvada en %s", path)
        return path

    def load_checkpoint(
        self,
        model: Flow_FEDformer,
        optimizer: Optimizer,
        scaler: torch.amp.GradScaler | None,
        checkpoint_path: str,
    ) -> tuple[int, int, float]:
        """Transbordo temporal desde dict persistido devuelta a RAM GPU."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info("Modelo recuperado desde respaldo histórico en %s", checkpoint_path)
        return checkpoint["epoch"], checkpoint["fold"], checkpoint["loss"]

    def _nll_loss(self, dist: Distribution, y_true: torch.Tensor) -> torch.Tensor:
        """Log-Probabilidad marginal negativa computada para la simulación iterativa."""
        try:
            log_prob = dist.log_prob(y_true)
            log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)
            return -log_prob.mean()
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Tubería probabilística decaída: %s. Aplicando función costo residual asintótica MSE.",
                exc,
            )
            return F.mse_loss(dist.mean, y_true)

    def _initialize_wandb(self) -> None:
        """Resuelve interconexiones de monitoreo experimental W&B."""
        try:
            if wandb is None:
                logger.info(
                    "No detectado entorno nativo de Weights & Biases. Procediendo sin bitácora externa."
                )
                self.wandb_run = None
                return

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=self.config.wandb_run_name,
                reinit=True,
            )
            logger.info("Vínculo seguro creado junto a infraestructura de W&B")
        except (WandbError, ValueError) as exc:
            logger.warning(
                "Falló la comunicación teleoperada (W&B): %s. Bloqueando reportes asíncronos.",
                exc,
            )
            self.wandb_run = None

    def _train_epoch(self, components: TrainingComponents, epoch: int) -> float:
        """Gestiona un fragmento totalitario del fold a lo largo del subsistema."""
        components.model.train()
        epoch_losses: list[float] = []
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
                        "Desbordamiento fatal detectado (GPU OOM). Reduzca batch_size, reinicie memoria VRAM interna."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

                logger.warning(
                    "Operación en bloque [%s] inestable sobre fold [%s]: %s. Avanzando a recuperación cíclica.",
                    batch_idx,
                    components.fold,
                    exc,
                )
                continue

            if loss is None:
                continue

            loss_value = float(loss.item() * accumulation_steps)

            if self._should_step(batch_idx, total_batches, accumulation_steps):
                self._optimizer_step(
                    components.optimizer, components.scaler, components.model
                )

            epoch_losses.append(loss_value)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        logger.info(
            "  Ciclo Epocal %s/%s, Loss Base: %.4f",
            epoch + 1,
            self.config.n_epochs_per_fold,
            avg_loss,
        )
        self.metrics_tracker.log_metrics({"train_loss": avg_loss}, epoch)
        return avg_loss

    def _evaluate_model(
        self, model: Flow_FEDformer, test_loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gestiona el pase lógico de evaluación probabilística sin contaminación cruzada."""
        model.eval()
        fold_preds: list[np.ndarray] = []
        fold_gt: list[np.ndarray] = []
        fold_samples: list[np.ndarray] = []

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
                    logger.warning(
                        "Bloque estocástico de evaluador corrompido: %s", exc
                    )
                    continue

        if fold_preds and fold_gt and fold_samples:
            return (
                np.concatenate(fold_preds, axis=0),
                np.concatenate(fold_gt, axis=0),
                np.concatenate(fold_samples, axis=1),
            )

        logger.warning(
            "No se validaron proyecciones aptas en el subsistema post-evaluador."
        )
        return np.array([]), np.array([]), np.array([])

    def _run_single_fold(
        self,
        fold_idx: int,
        split_size: int,
        total_size: int,
        total_folds: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Ciclo acoplado de Entrenamiento/Validación sobre un salto iterativo delimitado (Leak Tolerant)."""
        train_end_idx = fold_idx * split_size
        test_end_idx = min((fold_idx + 1) * split_size, total_size)

        if test_end_idx - train_end_idx < self.config.seq_len + self.config.pred_len:
            logger.warning(
                "Insuficiente densidad de datos subyacentes operando sobre fold %s. Suspendido.",
                fold_idx,
            )
            return None

        logger.info(
            "--- Fold Analítico %s/%s: Entrenamiento precomputado [%s], Predicción empírica [%s, %s] ---",
            fold_idx,
            total_folds,
            train_end_idx,
            train_end_idx,
            test_end_idx,
        )

        self.full_dataset.refit_for_cutoff(train_end_idx)

        train_indices, test_indices = self._build_fold_indices(
            train_end_idx, test_end_idx
        )
        if not train_indices:
            logger.warning(
                "Secuencia histórica insuficiente procesando fold transitorio %s post-límites.",
                fold_idx,
            )
            return None
        if not test_indices:
            logger.warning(
                "Secuencia de ensayo colapsada o nula operando fold activo %s.",
                fold_idx,
            )
            return None

        train_subset = Subset(self.full_dataset, train_indices)
        test_subset = Subset(self.full_dataset, test_indices)

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
            logger.warning(
                "Fold %s vacío con salida divergente de predicción", fold_idx
            )
            return None

        if self.wandb_run:
            self.wandb_run.log({"fold": fold_idx, "fold_completed": True})

        # GC Force clean per iteration
        del components, train_subset, test_subset, train_indices, test_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return fold_preds, fold_gt, fold_samples

    def _build_fold_indices(
        self, train_end_idx: int, test_end_idx: int
    ) -> tuple[list[int], list[int]]:
        """Aisla perimetralmente vectores que pudiesen provocar fallos ciegos o leaks temporales."""
        train_max_start = train_end_idx - self.config.seq_len - self.config.pred_len
        if train_max_start < 0:
            return [], []
        train_indices = list(range(train_max_start + 1))

        test_max_start = test_end_idx - self.config.seq_len - self.config.pred_len
        if test_max_start < train_end_idx:
            return train_indices, []

        test_limit = min(test_max_start + 1, len(self.full_dataset))
        test_indices = list(range(train_end_idx, test_limit))
        return train_indices, test_indices

    def run_backtest(
        self, n_splits: int = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Despliegue operativo automatizado del algoritmo sobre cortes asincrónos."""
        self._initialize_wandb()

        total_size = len(self.full_dataset)
        split_size = max(
            total_size // n_splits, self.config.seq_len + self.config.pred_len
        )
        total_folds = max(1, n_splits - 1)

        all_preds: list[np.ndarray] = []
        all_gt: list[np.ndarray] = []
        all_samples: list[np.ndarray] = []

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
            logger.exception(
                "Corrupción general forzando colapso del Backtest iterativo walk-forward"
            )
            raise
        finally:
            if self.wandb_run:
                self.wandb_run.finish()

        if not all_preds:
            logger.error(
                "No existió ni un sólo bloque precalculado de inferencias. Colapso."
            )
            return np.array([]), np.array([]), np.array([])

        return (
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_gt, axis=0),
            np.concatenate(all_samples, axis=1),
        )
