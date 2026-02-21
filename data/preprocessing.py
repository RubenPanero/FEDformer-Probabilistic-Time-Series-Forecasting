# -*- coding: utf-8 -*-
"""
Reusable preprocessing pipeline for robust time-series training/inference.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import FEDformerConfig

logger = logging.getLogger(__name__)


class _IdentityScaler:
    """No-op scaler with sklearn-like interface."""

    def fit(self, x: np.ndarray) -> "_IdentityScaler":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x


class PreprocessingPipeline:
    """Configurable preprocessing with schema validation and leakage-safe fitting."""

    VERSION = "1.0"

    def __init__(
        self,
        config: FEDformerConfig,
        target_features: List[str],
        date_column: Optional[str] = None,
        strict_mode: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.settings = config.sections.preprocessing
        self.target_features = list(target_features)
        self.date_column = date_column
        self.strict_mode = (
            self.settings.strict_mode if strict_mode is None else strict_mode
        )
        self.fit_scope = self.settings.fit_scope

        self.fitted = False
        self.fit_end_idx: Optional[int] = None
        self.source_columns: List[str] = []
        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.time_feature_columns: List[str] = []
        self.target_indices: List[int] = []
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.onehot_columns: List[str] = []
        self.fill_values: Dict[str, float] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.fit_stats: Dict[str, Dict[str, float]] = {}
        self.artifact_dir = Path(self.settings.artifact_dir)

        self.scaler: Any = _IdentityScaler()

    @classmethod
    def from_config(
        cls,
        config: FEDformerConfig,
        target_features: List[str],
        date_column: Optional[str] = None,
        strict_mode: Optional[bool] = None,
    ) -> "PreprocessingPipeline":
        return cls(config, target_features, date_column=date_column, strict_mode=strict_mode)

    def _fail_or_warn(self, message: str) -> None:
        if self.strict_mode:
            raise ValueError(message)
        logger.warning(message)

    def _create_scaler(self) -> Any:
        strategy = self.settings.scaling_strategy
        if strategy == "standard":
            return StandardScaler()
        if strategy == "robust":
            return RobustScaler()
        if strategy == "minmax":
            return MinMaxScaler()
        return _IdentityScaler()

    def _infer_column_roles(self, df: pd.DataFrame) -> None:
        roles = self.settings.feature_roles or {}
        self.source_columns = list(df.columns)

        if self.date_column and self.date_column not in df.columns:
            self._fail_or_warn(f"Date column '{self.date_column}' not present in input data.")

        declared_numeric = [c for c, role in roles.items() if role == "numeric" and c in df.columns]
        declared_categorical = [c for c, role in roles.items() if role == "categorical" and c in df.columns]
        excluded = {self.date_column} if self.date_column else set()
        excluded.update(declared_categorical)

        inferred_numeric = [
            c
            for c in df.select_dtypes(include=np.number).columns.tolist()
            if c not in excluded
        ]
        numeric = sorted(set(declared_numeric + inferred_numeric))

        inferred_categorical = [
            c
            for c in df.columns
            if c not in numeric and c != self.date_column and c not in declared_numeric
        ]
        categorical = sorted(set(declared_categorical + inferred_categorical))

        missing_targets = [c for c in self.target_features if c not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns in dataset: {missing_targets}")
        non_numeric_targets = [c for c in self.target_features if c not in numeric]
        if non_numeric_targets:
            raise ValueError(f"Target columns must be numeric: {non_numeric_targets}")

        self.numeric_columns = numeric
        self.categorical_columns = categorical

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        self.time_feature_columns = []
        if not self.date_column or self.date_column not in out.columns:
            return out
        if not self.settings.time_features:
            return out

        dt = pd.to_datetime(out[self.date_column], errors="coerce")
        if dt.isna().all():
            self._fail_or_warn(
                f"Failed to parse date column '{self.date_column}' for time features."
            )
            return out

        for feature_name in self.settings.time_features:
            col_name = f"__time_{feature_name}"
            if feature_name == "dayofweek":
                out[col_name] = dt.dt.dayofweek.astype(float)
            elif feature_name == "month":
                out[col_name] = dt.dt.month.astype(float)
            elif feature_name == "day":
                out[col_name] = dt.dt.day.astype(float)
            elif feature_name == "hour":
                out[col_name] = dt.dt.hour.astype(float)
            elif feature_name == "is_month_start":
                out[col_name] = dt.dt.is_month_start.astype(float)
            elif feature_name == "is_month_end":
                out[col_name] = dt.dt.is_month_end.astype(float)
            else:
                self._fail_or_warn(f"Unsupported time feature: {feature_name}")
                continue
            self.time_feature_columns.append(col_name)
        return out

    def _encode_categoricals_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        encoding = self.settings.categorical_encoding
        if encoding == "none" or not self.categorical_columns:
            self.onehot_columns = []
            return out

        if encoding == "ordinal":
            for col in self.categorical_columns:
                cats = out[col].astype("string").fillna("__nan__")
                unique = sorted(cats.unique().tolist())
                mapping = {cat: idx for idx, cat in enumerate(unique)}
                self.category_mappings[col] = mapping
                out[col] = cats.map(mapping).astype(float)
            return out

        # onehot
        onehot = pd.get_dummies(
            out[self.categorical_columns].astype("string").fillna("__nan__"),
            prefix=self.categorical_columns,
            dtype=float,
        )
        self.onehot_columns = onehot.columns.tolist()
        out = out.drop(columns=self.categorical_columns)
        out = pd.concat([out, onehot], axis=1)
        return out

    def _encode_categoricals_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        encoding = self.settings.categorical_encoding
        if encoding == "none" or not self.categorical_columns:
            return out

        if encoding == "ordinal":
            for col in self.categorical_columns:
                mapping = self.category_mappings.get(col, {})
                cats = out[col].astype("string").fillna("__nan__")
                out[col] = cats.map(mapping).fillna(-1.0).astype(float)
            return out

        onehot = pd.get_dummies(
            out[self.categorical_columns].astype("string").fillna("__nan__"),
            prefix=self.categorical_columns,
            dtype=float,
        )
        onehot = onehot.reindex(columns=self.onehot_columns, fill_value=0.0)
        out = out.drop(columns=self.categorical_columns)
        out = pd.concat([out, onehot], axis=1)
        return out

    def _build_feature_frame(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        out = self._add_time_features(df)
        if fit:
            self._infer_column_roles(out)

        selected = self.numeric_columns + self.time_feature_columns
        if self.settings.categorical_encoding != "none":
            selected += self.categorical_columns
        missing_selected = [c for c in selected if c not in out.columns]
        if missing_selected:
            raise ValueError(f"Missing selected columns in input frame: {missing_selected}")

        feature_df = out[selected].copy()
        if fit:
            feature_df = self._encode_categoricals_fit(feature_df)
        else:
            feature_df = self._encode_categoricals_transform(feature_df)
        return feature_df

    def _fit_missing_params(self, fit_df: pd.DataFrame) -> None:
        self.fill_values = {}
        if self.settings.missing_policy != "impute_median":
            return
        for col in fit_df.columns:
            series = fit_df[col]
            if pd.api.types.is_numeric_dtype(series):
                self.fill_values[col] = float(series.median()) if not series.dropna().empty else 0.0
            else:
                mode = series.mode(dropna=True)
                self.fill_values[col] = mode.iloc[0] if not mode.empty else "__nan__"

    def _apply_missing_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        policy = self.settings.missing_policy
        if policy == "drop":
            return df.dropna().reset_index(drop=True)
        if policy == "ffill_bfill":
            return df.ffill().bfill()
        if policy == "impute_median":
            return df.fillna(self.fill_values)
        if df.isna().any().any():
            raise ValueError("Missing values found with missing_policy='error'.")
        return df

    def _fit_outlier_params(self, fit_df: pd.DataFrame) -> None:
        self.outlier_bounds = {}
        if self.settings.outlier_policy == "none":
            return
        for col in fit_df.columns:
            if self.settings.outlier_policy == "winsorize":
                lower = float(fit_df[col].quantile(0.01))
                upper = float(fit_df[col].quantile(0.99))
            else:
                q1 = float(fit_df[col].quantile(0.25))
                q3 = float(fit_df[col].quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
            self.outlier_bounds[col] = (lower, upper)

    def _apply_outlier_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.settings.outlier_policy == "none":
            return df
        out = df.copy()
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lower, upper=upper)
        return out

    def _update_fit_stats(self, fit_df: pd.DataFrame) -> None:
        self.fit_stats = {}
        for col in fit_df.columns:
            series = fit_df[col]
            self.fit_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0) + 1e-9),
                "null_rate": float(series.isna().mean()),
            }

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        if not self.fitted:
            return
        missing = [col for col in self.source_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def validate_input_schema(
        self, df: pd.DataFrame, feature_df: Optional[pd.DataFrame] = None
    ) -> None:
        self._validate_required_columns(df)
        if not self.fitted:
            return
        checks = self.settings.drift_checks
        if not checks.get("enabled", False):
            return

        allow_extra = bool(checks.get("allow_extra_columns", True))
        if not allow_extra:
            extras = [c for c in df.columns if c not in self.source_columns]
            if extras:
                self._fail_or_warn(f"Unexpected extra columns: {extras}")

        null_thr = float(checks.get("null_rate_threshold", 0.2))
        mean_thr = float(checks.get("mean_shift_threshold", 6.0))
        std_bounds = checks.get("std_ratio_bounds", [0.2, 5.0])
        std_low = float(std_bounds[0])
        std_high = float(std_bounds[1])

        drift_frame = feature_df
        if drift_frame is None:
            keep = [col for col in self.numeric_columns if col in df.columns]
            drift_frame = df[keep].copy()

        for col in drift_frame.columns:
            if col not in self.fit_stats:
                continue
            series = pd.to_numeric(drift_frame[col], errors="coerce")
            null_rate = float(series.isna().mean())
            if null_rate > null_thr:
                self._fail_or_warn(
                    f"Column '{col}' exceeds null rate threshold: {null_rate:.3f} > {null_thr:.3f}"
                )
            stats = self.fit_stats[col]
            curr_mean = float(series.mean())
            curr_std = float(series.std(ddof=0) + 1e-9)
            mean_shift = abs(curr_mean - stats["mean"]) / (stats["std"] + 1e-9)
            std_ratio = curr_std / (stats["std"] + 1e-9)
            if mean_shift > mean_thr:
                self._fail_or_warn(
                    f"Column '{col}' mean shift too large: {mean_shift:.3f} > {mean_thr:.3f}"
                )
            if not (std_low <= std_ratio <= std_high):
                self._fail_or_warn(
                    f"Column '{col}' std ratio out of bounds: {std_ratio:.3f} not in [{std_low:.3f}, {std_high:.3f}]"
                )

    def fit(self, df: pd.DataFrame, fit_end_idx: Optional[int] = None) -> "PreprocessingPipeline":
        raw_features = self._build_feature_frame(df, fit=True)
        cutoff = fit_end_idx if fit_end_idx is not None else len(raw_features)
        cutoff = int(max(1, min(cutoff, len(raw_features))))
        self.fit_end_idx = cutoff

        fit_df = raw_features.iloc[:cutoff].copy()
        self._fit_missing_params(fit_df)
        fit_df = self._apply_missing_policy(fit_df)
        self._fit_outlier_params(fit_df)
        fit_df = self._apply_outlier_policy(fit_df)
        self._update_fit_stats(fit_df)

        self.scaler = self._create_scaler()
        self.scaler.fit(fit_df.values)

        self.feature_columns = fit_df.columns.tolist()
        self.target_indices = [self.feature_columns.index(t) for t in self.target_features]
        self.fitted = True

        if self.settings.persist_artifacts:
            self.save_artifacts(self.artifact_dir)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted before transform.")
        self._validate_required_columns(df)

        feature_df = self._build_feature_frame(df, fit=False)
        feature_df = self._apply_missing_policy(feature_df)
        feature_df = self._apply_outlier_policy(feature_df)
        self.validate_input_schema(df, feature_df=feature_df)
        feature_df = feature_df.reindex(columns=self.feature_columns, fill_value=0.0)
        transformed = self.scaler.transform(feature_df.values)
        return pd.DataFrame(transformed, columns=self.feature_columns, index=feature_df.index)

    def inverse_transform_targets(
        self, y: np.ndarray, target_names: List[str]
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted before inverse transform.")
        arr = np.asarray(y, dtype=float)
        if arr.shape[-1] != len(target_names):
            raise ValueError("Last dimension of y must match target_names length.")

        flat = arr.reshape(-1, arr.shape[-1])
        full = np.zeros((flat.shape[0], len(self.feature_columns)), dtype=float)
        for idx, name in enumerate(target_names):
            if name not in self.feature_columns:
                raise ValueError(f"Target '{name}' not found in feature columns.")
            full[:, self.feature_columns.index(name)] = flat[:, idx]
        inv = self.scaler.inverse_transform(full)
        out = np.stack(
            [inv[:, self.feature_columns.index(name)] for name in target_names], axis=-1
        )
        return out.reshape(*arr.shape[:-1], len(target_names))

    def save_artifacts(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save artifacts before fitting preprocessing pipeline.")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        schema = {
            "version": self.VERSION,
            "source_columns": self.source_columns,
            "feature_columns": self.feature_columns,
            "target_features": self.target_features,
            "target_indices": self.target_indices,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "time_feature_columns": self.time_feature_columns,
            "onehot_columns": self.onehot_columns,
            "category_mappings": self.category_mappings,
        }
        metadata = {
            "fit_end_idx": self.fit_end_idx,
            "fill_values": self.fill_values,
            "outlier_bounds": {k: list(v) for k, v in self.outlier_bounds.items()},
            "fit_stats": self.fit_stats,
            "settings": {
                "feature_roles": self.settings.feature_roles,
                "scaling_strategy": self.settings.scaling_strategy,
                "missing_policy": self.settings.missing_policy,
                "outlier_policy": self.settings.outlier_policy,
                "fit_scope": self.settings.fit_scope,
                "persist_artifacts": self.settings.persist_artifacts,
                "drift_checks": self.settings.drift_checks,
                "strict_mode": self.settings.strict_mode,
                "categorical_encoding": self.settings.categorical_encoding,
                "time_features": self.settings.time_features,
                "artifact_dir": self.settings.artifact_dir,
            },
        }

        (out_dir / "schema.json").write_text(
            json.dumps(schema, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        with (out_dir / "scaler.pkl").open("wb") as f:
            pickle.dump(self.scaler, f)

    def load_artifacts(self, path: str | Path) -> "PreprocessingPipeline":
        in_dir = Path(path)
        schema = json.loads((in_dir / "schema.json").read_text(encoding="utf-8"))
        metadata = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))
        with (in_dir / "scaler.pkl").open("rb") as f:
            self.scaler = pickle.load(f)

        self.source_columns = schema["source_columns"]
        self.feature_columns = schema["feature_columns"]
        self.target_features = schema["target_features"]
        self.target_indices = schema["target_indices"]
        self.numeric_columns = schema["numeric_columns"]
        self.categorical_columns = schema["categorical_columns"]
        self.time_feature_columns = schema["time_feature_columns"]
        self.onehot_columns = schema.get("onehot_columns", [])
        self.category_mappings = schema.get("category_mappings", {})

        self.fit_end_idx = metadata.get("fit_end_idx")
        self.fill_values = metadata.get("fill_values", {})
        self.outlier_bounds = {
            k: (float(v[0]), float(v[1]))
            for k, v in metadata.get("outlier_bounds", {}).items()
        }
        self.fit_stats = metadata.get("fit_stats", {})
        self.fitted = True
        return self
