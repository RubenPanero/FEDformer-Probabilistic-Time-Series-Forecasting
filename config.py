# -*- coding: utf-8 -*-
"""
Configuracion del sistema Vanguard FEDformer.
"""

import os
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Set

import logging

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SequenceSettings:
    """Sequence segmentation lengths for encoder/decoder."""

    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24


@dataclass
class TransformerSettings:
    """Transformer backbone hyper-parameters."""

    # pylint: disable=too-many-instance-attributes

    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    modes: int = 64
    activation: str = "gelu"
    dropout: float = 0.1


@dataclass
class DecompositionSettings:
    """Moving average kernel configuration for seasonal/trend split."""

    moving_avg: Optional[List[int]] = None


@dataclass
class RegimeSettings:
    """Latent regime embedding configuration."""

    n_regimes: int = 3
    regime_embedding_dim: int = 16


@dataclass
class FlowSettings:
    """Normalizing flow depth and hidden size."""

    n_flow_layers: int = 4
    flow_hidden_dim: int = 64


@dataclass
class ModelSettings:
    """Grouped model-related settings."""

    sequence: SequenceSettings = field(default_factory=SequenceSettings)
    transformer: TransformerSettings = field(default_factory=TransformerSettings)
    decomposition: DecompositionSettings = field(default_factory=DecompositionSettings)
    regime: RegimeSettings = field(default_factory=RegimeSettings)
    flow: FlowSettings = field(default_factory=FlowSettings)


@dataclass
class OptimizationSettings:
    """Optimizer-level hyper-parameters."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


@dataclass
class LoopSettings:
    """Training loop batch/epoch controls."""

    n_epochs_per_fold: int = 5
    batch_size: int = 32
    gradient_accumulation_steps: int = 1


@dataclass
class RuntimeSettings:
    """Runtime toggles for training."""

    use_amp: bool = True
    use_gradient_checkpointing: bool = False
    compile_mode: str = "max-autotune"


@dataclass
class TrainingSettings:
    """Grouped training-related settings."""

    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    loop: LoopSettings = field(default_factory=LoopSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)


@dataclass
class MonitoringSettings:
    """External monitoring/metadata options (W&B, dataset columns)."""

    wandb_project: str = "vanguard-fedformer-flow"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    date_column: Optional[str] = None


@dataclass
class ReproSettings:
    """Reproducibility toggles and seed configuration."""

    seed: int = 42
    deterministic: bool = False


@dataclass
class DerivedSettings:
    """Derived values computed from the dataset headers."""

    enc_in: Optional[int] = None
    dec_in: Optional[int] = None
    c_out: Optional[int] = None


@dataclass
class ConfigSections:
    """Container for grouped configuration sections."""

    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    reproducibility: ReproSettings = field(default_factory=ReproSettings)
    derived: DerivedSettings = field(default_factory=DerivedSettings)


@dataclass(init=False)
class FEDformerConfig:
    """Enhanced configuration class with validation and better organization."""

    # pylint: disable=missing-function-docstring,too-many-public-methods,too-many-instance-attributes

    target_features: List[str]
    file_path: str
    sections: ConfigSections = field(init=False)

    _ALLOWED_KEYS: ClassVar[Set[str]] = {
        "seq_len",
        "label_len",
        "pred_len",
        "d_model",
        "n_heads",
        "e_layers",
        "d_layers",
        "d_ff",
        "modes",
        "moving_avg",
        "activation",
        "dropout",
        "n_regimes",
        "regime_embedding_dim",
        "n_flow_layers",
        "flow_hidden_dim",
        "learning_rate",
        "weight_decay",
        "n_epochs_per_fold",
        "batch_size",
        "use_amp",
        "use_gradient_checkpointing",
        "gradient_accumulation_steps",
        "compile_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "date_column",
        "seed",
        "deterministic",
    }

    def __init__(
        self,
        target_features: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        **kwargs
    ) -> None:
        # Set defaults if not provided
        if file_path is None:
            # Use smoke_test.csv if available, otherwise nvidia_stock
            default_path = os.path.join(
                os.path.dirname(__file__), "data", "smoke_test.csv"
            )
            if not os.path.exists(default_path):
                default_path = os.path.join(
                    os.path.dirname(__file__), "data", "nvidia_stock_2024-08-20_to_2025-08-20.csv"
                )
            file_path = default_path

        # Auto-detect target features if not provided
        if target_features is None:
            try:
                df_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
                # Try to find a price column
                for col in ["Close", "close", "Close_Price", "close_price"]:
                    if col in df_cols:
                        target_features = [col]
                        break
                # If no price column found, use first non-date column
                if target_features is None:
                    non_date_cols = [col for col in df_cols if col.lower() not in ["date", "time"]]
                    if non_date_cols:
                        target_features = [non_date_cols[0]]
                    else:
                        target_features = [df_cols[0]]
            except Exception:
                # Fallback if file cannot be read
                target_features = ["Close"]

        self.target_features = target_features
        self.file_path = file_path
        self.sections = ConfigSections()

        unexpected = set(kwargs) - self._ALLOWED_KEYS
        if unexpected:
            unexpected_str = ", ".join(sorted(unexpected))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected_str}")

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self) -> None:
        """Sets derived configuration parameters and validates config"""
        if self.moving_avg is None:
            self.moving_avg = [24, 48]

        try:
            df_cols = pd.read_csv(self.file_path, nrows=0).columns
            if self.date_column and self.date_column in df_cols:
                feature_cols = [col for col in df_cols if col != self.date_column]
            else:
                feature_cols = list(df_cols)
            self.enc_in = len(feature_cols)
            self.dec_in = len(feature_cols)
            self.c_out = len(self.target_features)
        except Exception as exc:
            logger.error("Failed to read CSV file %s: %s", self.file_path, exc)
            raise

        max_modes = max(1, self.seq_len // 2)
        if self.modes > max_modes:
            logger.warning(
                "modes (%s) > seq_len//2 (%s), clamping modes to %s",
                self.modes,
                max_modes,
                max_modes,
            )
            self.modes = max_modes
        if self.modes < 1:
            logger.warning("modes (%s) < 1, setting to 1", self.modes)
            self.modes = 1

        self.validate(df_cols)

    def validate(self, df_columns: Optional[pd.Index] = None) -> None:
        """Validate configuration consistency"""
        if df_columns is None:
            df_columns = pd.read_csv(self.file_path, nrows=0).columns

        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.label_len <= self.seq_len, (
            f"label_len ({self.label_len}) cannot exceed seq_len ({self.seq_len})"
        )
        assert 1 <= self.modes <= max(1, self.seq_len // 2), (
            f"modes ({self.modes}) must be in [1, seq_len//2] ({self.seq_len // 2})"
        )
        assert self.activation in ["gelu", "relu"], (
            f"activation must be 'gelu' or 'relu', got {self.activation}"
        )
        assert all(col in df_columns for col in self.target_features), (
            "All target features must exist in the dataset"
        )
        assert 0 <= self.dropout < 1, f"Dropout must be in [0, 1), got {self.dropout}"
        assert self.learning_rate > 0, (
            f"Learning rate must be positive, got {self.learning_rate}"
        )
        assert self.weight_decay >= 0, (
            f"Weight decay must be non-negative, got {self.weight_decay}"
        )
        assert self.batch_size > 0, (
            f"Batch size must be positive, got {self.batch_size}"
        )
        assert self.gradient_accumulation_steps > 0, (
            f"Gradient accumulation steps must be positive, got {self.gradient_accumulation_steps}"
        )
        assert self.pred_len % 2 == 0, (
            f"pred_len ({self.pred_len}) must be even for affine coupling"
        )

    # -- Model settings proxies -------------------------------------------------
    @property
    def seq_len(self) -> int:
        return self.sections.model.sequence.seq_len

    @seq_len.setter
    def seq_len(self, value: int) -> None:
        self.sections.model.sequence.seq_len = value

    @property
    def label_len(self) -> int:
        return self.sections.model.sequence.label_len

    @label_len.setter
    def label_len(self, value: int) -> None:
        self.sections.model.sequence.label_len = value

    @property
    def pred_len(self) -> int:
        return self.sections.model.sequence.pred_len

    @pred_len.setter
    def pred_len(self, value: int) -> None:
        self.sections.model.sequence.pred_len = value

    @property
    def d_model(self) -> int:
        return self.sections.model.transformer.d_model

    @d_model.setter
    def d_model(self, value: int) -> None:
        self.sections.model.transformer.d_model = value

    @property
    def n_heads(self) -> int:
        return self.sections.model.transformer.n_heads

    @n_heads.setter
    def n_heads(self, value: int) -> None:
        self.sections.model.transformer.n_heads = value

    @property
    def e_layers(self) -> int:
        return self.sections.model.transformer.e_layers

    @e_layers.setter
    def e_layers(self, value: int) -> None:
        self.sections.model.transformer.e_layers = value

    @property
    def d_layers(self) -> int:
        return self.sections.model.transformer.d_layers

    @d_layers.setter
    def d_layers(self, value: int) -> None:
        self.sections.model.transformer.d_layers = value

    @property
    def d_ff(self) -> int:
        return self.sections.model.transformer.d_ff

    @d_ff.setter
    def d_ff(self, value: int) -> None:
        self.sections.model.transformer.d_ff = value

    @property
    def modes(self) -> int:
        return self.sections.model.transformer.modes

    @modes.setter
    def modes(self, value: int) -> None:
        self.sections.model.transformer.modes = value

    @property
    def moving_avg(self) -> Optional[List[int]]:
        return self.sections.model.decomposition.moving_avg

    @moving_avg.setter
    def moving_avg(self, value: Optional[List[int]]) -> None:
        self.sections.model.decomposition.moving_avg = value

    @property
    def activation(self) -> str:
        return self.sections.model.transformer.activation

    @activation.setter
    def activation(self, value: str) -> None:
        self.sections.model.transformer.activation = value

    @property
    def dropout(self) -> float:
        return self.sections.model.transformer.dropout

    @dropout.setter
    def dropout(self, value: float) -> None:
        self.sections.model.transformer.dropout = value

    @property
    def n_regimes(self) -> int:
        return self.sections.model.regime.n_regimes

    @n_regimes.setter
    def n_regimes(self, value: int) -> None:
        self.sections.model.regime.n_regimes = value

    @property
    def regime_embedding_dim(self) -> int:
        return self.sections.model.regime.regime_embedding_dim

    @regime_embedding_dim.setter
    def regime_embedding_dim(self, value: int) -> None:
        self.sections.model.regime.regime_embedding_dim = value

    @property
    def n_flow_layers(self) -> int:
        return self.sections.model.flow.n_flow_layers

    @n_flow_layers.setter
    def n_flow_layers(self, value: int) -> None:
        self.sections.model.flow.n_flow_layers = value

    @property
    def flow_hidden_dim(self) -> int:
        return self.sections.model.flow.flow_hidden_dim

    @flow_hidden_dim.setter
    def flow_hidden_dim(self, value: int) -> None:
        self.sections.model.flow.flow_hidden_dim = value

    # -- Training settings proxies --------------------------------------------
    @property
    def learning_rate(self) -> float:
        return self.sections.training.optimization.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.sections.training.optimization.learning_rate = value

    @property
    def weight_decay(self) -> float:
        return self.sections.training.optimization.weight_decay

    @weight_decay.setter
    def weight_decay(self, value: float) -> None:
        self.sections.training.optimization.weight_decay = value

    @property
    def n_epochs_per_fold(self) -> int:
        return self.sections.training.loop.n_epochs_per_fold

    @n_epochs_per_fold.setter
    def n_epochs_per_fold(self, value: int) -> None:
        self.sections.training.loop.n_epochs_per_fold = value

    @property
    def batch_size(self) -> int:
        return self.sections.training.loop.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.sections.training.loop.batch_size = value

    @property
    def use_amp(self) -> bool:
        return self.sections.training.runtime.use_amp

    @use_amp.setter
    def use_amp(self, value: bool) -> None:
        self.sections.training.runtime.use_amp = value

    @property
    def use_gradient_checkpointing(self) -> bool:
        return self.sections.training.runtime.use_gradient_checkpointing

    @use_gradient_checkpointing.setter
    def use_gradient_checkpointing(self, value: bool) -> None:
        self.sections.training.runtime.use_gradient_checkpointing = value

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.sections.training.loop.gradient_accumulation_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, value: int) -> None:
        self.sections.training.loop.gradient_accumulation_steps = value

    @property
    def compile_mode(self) -> str:
        return self.sections.training.runtime.compile_mode

    @compile_mode.setter
    def compile_mode(self, value: str) -> None:
        self.sections.training.runtime.compile_mode = value

    # -- Monitoring settings proxies -----------------------------------------
    @property
    def wandb_project(self) -> str:
        return self.sections.monitoring.wandb_project

    @wandb_project.setter
    def wandb_project(self, value: str) -> None:
        self.sections.monitoring.wandb_project = value

    @property
    def wandb_entity(self) -> Optional[str]:
        return self.sections.monitoring.wandb_entity

    @wandb_entity.setter
    def wandb_entity(self, value: Optional[str]) -> None:
        self.sections.monitoring.wandb_entity = value

    @property
    def wandb_run_name(self) -> Optional[str]:
        return self.sections.monitoring.wandb_run_name

    @wandb_run_name.setter
    def wandb_run_name(self, value: Optional[str]) -> None:
        self.sections.monitoring.wandb_run_name = value

    @property
    def date_column(self) -> Optional[str]:
        return self.sections.monitoring.date_column

    @date_column.setter
    def date_column(self, value: Optional[str]) -> None:
        self.sections.monitoring.date_column = value

    # -- Reproducibility settings proxies ------------------------------------
    @property
    def seed(self) -> int:
        return self.sections.reproducibility.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.sections.reproducibility.seed = value

    @property
    def deterministic(self) -> bool:
        return self.sections.reproducibility.deterministic

    @deterministic.setter
    def deterministic(self, value: bool) -> None:
        self.sections.reproducibility.deterministic = value

    # -- Derived values proxies ----------------------------------------------
    @property
    def enc_in(self) -> Optional[int]:
        return self.sections.derived.enc_in

    @enc_in.setter
    def enc_in(self, value: Optional[int]) -> None:
        self.sections.derived.enc_in = value

    @property
    def dec_in(self) -> Optional[int]:
        return self.sections.derived.dec_in

    @dec_in.setter
    def dec_in(self, value: Optional[int]) -> None:
        self.sections.derived.dec_in = value

    @property
    def c_out(self) -> Optional[int]:
        return self.sections.derived.c_out

    @c_out.setter
    def c_out(self, value: Optional[int]) -> None:
        self.sections.derived.c_out = value
