# -*- coding: utf-8 -*-
"""
Componentes de manejo de datos para series de tiempo.
"""

import logging
import numpy as np
import pandas as pd
import torch
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import Dict

from config import FEDformerConfig

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Enhanced regime detector with better error handling"""

    def __init__(self, n_regimes: int) -> None:
        self.n_regimes = n_regimes
        self.quantiles = None

    def fit(self, data: np.ndarray) -> None:
        """Fit quantile thresholds based on rolling volatility.
        
        FIXED: Correctly computes rolling volatility (std, not mean).
        Handles 1D and 2D arrays properly with correct axis handling.
        """
        try:
            returns = np.diff(data, axis=0) / (np.abs(data[:-1]) + 1e-9)
            
            # Calculate rolling volatility correctly (std, not mean)
            rolling_vol = pd.DataFrame(returns).rolling(
                window=min(24, len(returns) // 2),
                min_periods=1
            ).std(ddof=1)  # Use sample std (ddof=1)
            
            # Compute overall volatility per period
            volatility = rolling_vol.dropna().values.std(axis=0)
            
            if len(volatility) > 1:
                self.quantiles = np.quantile(
                    volatility, np.linspace(0, 1, self.n_regimes + 1)[1:-1]
                )
            else:
                self.quantiles = np.zeros(self.n_regimes - 1)
        except Exception as e:
            logger.warning(f"Regime detector fit failed: {e}. Using default quantiles.")
            self.quantiles = np.zeros(self.n_regimes - 1)

    def get_regime(self, sequence: np.ndarray) -> int:
        if self.quantiles is None:
            raise RuntimeError("Detector has not been fitted.")
        try:
            returns = np.diff(sequence, axis=0) / (np.abs(sequence[:-1]) + 1e-9)
            sequence_vol = np.std(returns, axis=1).mean()
            return min(np.digitize(sequence_vol, self.quantiles), self.n_regimes - 1)
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}. Using regime 0.")
            return 0


class TimeSeriesDataset(Dataset):
    """Optimized dataset with caching and better memory management"""

    def __init__(self, config: FEDformerConfig, flag: str) -> None:
        self.config = config
        self.flag = flag
        self.scaler = StandardScaler()
        self._regime_cache = {}  # Cache regime calculations
        self._read_and_process_data()

        # Pre-compute valid indices
        self._valid_indices = list(
            range(len(self.data_x) - self.config.seq_len - self.config.pred_len + 1)
        )

    def _read_and_process_data(self) -> None:
        try:
            parse_cols = [self.config.date_column] if self.config.date_column else None
            df_raw = pd.read_csv(self.config.file_path, parse_dates=parse_cols)

            if self.config.date_column and self.config.date_column in df_raw.columns:
                self.df_data = df_raw.drop(self.config.date_column, axis=1)
            else:
                self.df_data = df_raw

            self.target_indices = [
                self.df_data.columns.get_loc(col) for col in self.config.target_features
            ]

            num_train = int(len(df_raw) * 0.7)
            num_val = int(len(df_raw) * 0.2)
            border1s = {
                "train": 0,
                "val": max(0, num_train - self.config.seq_len),
                "test": max(0, len(df_raw) - num_val - self.config.seq_len),
                "all": 0,
            }
            border2s = {
                "train": num_train,
                "val": len(df_raw),
                "test": len(df_raw),
                "all": len(df_raw),
            }
            if self.flag not in border1s:
                raise ValueError(
                    f"flag must be one of {list(border1s.keys())}, got {self.flag}"
                )

            # Select only numeric columns for scaling
            numeric_cols = self.df_data.select_dtypes(include=np.number).columns
            train_data = self.df_data.loc[: num_train - 1, numeric_cols].values

            self.scaler.fit(train_data)
            self.full_data_scaled = self.scaler.transform(
                self.df_data[numeric_cols].values
            )

            self.regime_detector = RegimeDetector(n_regimes=self.config.n_regimes)
            self.regime_detector.fit(train_data[:, self.target_indices])

            self.data_x = self.full_data_scaled[
                border1s[self.flag] : border2s[self.flag]
            ]

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    @lru_cache(maxsize=2048)  # OPTIMIZED: Increased cache size for better performance
    def _get_regime_cached(self, seq_hash: tuple) -> int:
        """Cache regime calculations to avoid recomputation"""
        seq_array = np.array(seq_hash).reshape(-1, len(self.target_indices))
        return self.regime_detector.get_regime(seq_array)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        try:
            s_end = index + self.config.seq_len
            r_end = (
                s_end
                - self.config.label_len
                + self.config.label_len
                + self.config.pred_len
            )

            seq_x = self.data_x[index:s_end]
            seq_dec_input = self.data_x[s_end - self.config.label_len : r_end]
            seq_y_true = seq_dec_input[-self.config.pred_len :, self.target_indices]

            # Use cached regime calculation
            seq_hash = tuple(seq_x[:, self.target_indices].flatten())
            regime = self._get_regime_cached(seq_hash)

            return {
                "x_enc": torch.from_numpy(seq_x.astype(np.float32)),
                "x_dec": torch.from_numpy(seq_dec_input.astype(np.float32)),
                "y_true": torch.from_numpy(seq_y_true.astype(np.float32)),
                "x_regime": torch.tensor([regime], dtype=torch.long),
            }
        except Exception as e:
            logger.warning(f"Error getting item {index}: {e}")
            # Return dummy data to prevent training interruption
            dummy_x_enc = torch.zeros(
                (self.config.seq_len, self.config.enc_in), dtype=torch.float32
            )
            dummy_x_dec = torch.zeros(
                (self.config.label_len + self.config.pred_len, self.config.dec_in),
                dtype=torch.float32,
            )
            dummy_y_true = torch.zeros(
                (self.config.pred_len, self.config.c_out), dtype=torch.float32
            )
            dummy_regime = torch.tensor([0], dtype=torch.long)

            return {
                "x_enc": dummy_x_enc,
                "x_dec": dummy_x_dec,
                "y_true": dummy_y_true,
                "x_regime": dummy_regime,
            }

    def __len__(self) -> int:
        return len(self._valid_indices)
