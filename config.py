# -*- coding: utf-8 -*-
"""
Configuración del sistema Vanguard FEDformer.
"""

import logging
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FEDformerConfig:
    """
    Enhanced configuration class with validation and better organization.
    """
    # Required fields
    target_features: List[str]
    file_path: str
    
    # Model architecture
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    modes: int = 64
    moving_avg: List[int] = None
    activation: str = 'gelu'  # FIXED: Added missing activation parameter
    dropout: float = 0.1
    
    # Regime detection
    n_regimes: int = 3
    regime_embedding_dim: int = 16
    
    # Normalizing Flow
    n_flow_layers: int = 4
    flow_hidden_dim: int = 64
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_epochs_per_fold: int = 5
    batch_size: int = 32
    use_amp: bool = True
    use_gradient_checkpointing: bool = False  # NEW: Memory optimization option
    gradient_accumulation_steps: int = 1  # NEW: Gradient accumulation for larger effective batch size
    compile_mode: str = 'max-autotune'
    
    # Logging and monitoring
    wandb_project: str = "vanguard-fedformer-flow"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Data configuration
    date_column: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Derived fields (set automatically)
    enc_in: int = None
    dec_in: int = None
    c_out: int = None

    def __post_init__(self):
        """Sets derived configuration parameters and validates config"""
        if self.moving_avg is None:
            self.moving_avg = [24, 48]
        
        # Set derived parameters
        try:
            # Use header-only read to avoid heavy IO
            df_cols = pd.read_csv(self.file_path, nrows=0).columns
            if self.date_column and self.date_column in df_cols:
                feature_cols = [c for c in df_cols if c != self.date_column]
            else:
                feature_cols = list(df_cols)
            self.enc_in = len(feature_cols)
            self.dec_in = len(feature_cols)
            self.c_out = len(self.target_features)
        except Exception as e:
            logger.error(f"Failed to read CSV file {self.file_path}: {e}")
            raise
        
        # Adjust and validate modes to avoid runtime assertion failures
        max_modes = max(1, self.seq_len // 2)
        if self.modes > max_modes:
            logger.warning(f"modes ({self.modes}) > seq_len//2 ({max_modes}), clamping modes to {max_modes}")
            self.modes = max_modes
        if self.modes < 1:
            logger.warning(f"modes ({self.modes}) < 1, setting to 1")
            self.modes = 1
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate configuration consistency"""
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.label_len <= self.seq_len, f"label_len ({self.label_len}) cannot exceed seq_len ({self.seq_len})"
        assert 1 <= self.modes <= max(1, self.seq_len // 2), f"modes ({self.modes}) must be in [1, seq_len//2] ({self.seq_len // 2})"
        assert self.activation in ['gelu', 'relu'], f"activation must be 'gelu' or 'relu', got {self.activation}"
        # Check that targets exist using header-only read (avoid full read)
        assert all(col in pd.read_csv(self.file_path, nrows=0).columns for col in self.target_features), "All target features must exist in the dataset"
        # OPTIMIZED: Additional validation for numerical stability
        assert 0 <= self.dropout < 1, f"Dropout must be in [0, 1), got {self.dropout}"
        assert self.learning_rate > 0, f"Learning rate must be positive, got {self.learning_rate}"
        assert self.weight_decay >= 0, f"Weight decay must be non-negative, got {self.weight_decay}"
        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}"
        assert self.gradient_accumulation_steps > 0, f"Gradient accumulation steps must be positive, got {self.gradient_accumulation_steps}"
        # Ensure flow coupling split works
        assert self.pred_len % 2 == 0, f"pred_len ({self.pred_len}) must be even for affine coupling"

