#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic benchmarks for phase-1 critical bottlenecks optimizations."""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEDformerConfig, apply_preset
from models.fedformer import Flow_FEDformer
from models.layers import FourierAttention
from training.utils import mc_dropout_inference

FIXTURE_CSV = str(PROJECT_ROOT / "tests" / "fixtures" / "NVDA_features.csv")


def _measure(function, *args, **kwargs) -> tuple[Any, float, int]:
    tracemalloc.start()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def _make_config(**kwargs: object) -> FEDformerConfig:
    config_kwargs: dict[str, object] = {
        "target_features": ["Close"],
        "file_path": FIXTURE_CSV,
        "seq_len": 24,
        "label_len": 8,
        "pred_len": 4,
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "e_layers": 1,
        "d_layers": 1,
        "modes": 8,
        "dropout": 0.1,
        "n_flow_layers": 2,
        "flow_hidden_dim": 16,
    }
    config_kwargs.update(kwargs)
    return FEDformerConfig(
        **config_kwargs,
    )


def _make_batch(
    config: FEDformerConfig, batch_size: int = 4
) -> dict[str, torch.Tensor]:
    return {
        "x_enc": torch.randn(batch_size, config.seq_len, config.enc_in),
        "x_dec": torch.randn(
            batch_size, config.label_len + config.pred_len, config.dec_in
        ),
        "y_true": torch.randn(batch_size, config.pred_len, config.c_out),
        "x_regime": torch.randint(
            0, config.n_regimes, (batch_size, 1), dtype=torch.long
        ),
    }


def benchmark_mc_dropout() -> dict[str, float]:
    config = _make_config()
    model = Flow_FEDformer(config).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch = _make_batch(config)

    torch.manual_seed(2026)
    baseline, baseline_time, baseline_peak = _measure(
        mc_dropout_inference,
        model,
        batch,
        20,
        True,
        1,
    )
    torch.manual_seed(2026)
    optimized, optimized_time, optimized_peak = _measure(
        mc_dropout_inference,
        model,
        batch,
        20,
        True,
        10,
    )

    assert baseline.shape == optimized.shape
    return {
        "baseline_time_s": baseline_time,
        "optimized_time_s": optimized_time,
        "time_delta_pct": ((optimized_time - baseline_time) / baseline_time) * 100.0,
        "baseline_peak_bytes": float(baseline_peak),
        "optimized_peak_bytes": float(optimized_peak),
        "memory_delta_pct": ((optimized_peak - baseline_peak) / baseline_peak) * 100.0
        if baseline_peak
        else 0.0,
    }


def benchmark_fourier_modes() -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(8, 4, 96, 8, device=device)
    baseline = FourierAttention(d_keys=8, seq_len=96, modes=64).to(device)
    optimized_cfg = _make_config(seq_len=96, label_len=48, pred_len=20, modes=64)
    apply_preset(optimized_cfg, "fourier_optimized")
    optimized = FourierAttention(d_keys=8, seq_len=96, modes=optimized_cfg.modes).to(
        device
    )

    _, baseline_time, baseline_peak = _measure(baseline, x)
    _, optimized_time, optimized_peak = _measure(optimized, x)
    return {
        "baseline_modes": 64.0,
        "optimized_modes": float(optimized_cfg.modes),
        "baseline_time_s": baseline_time,
        "optimized_time_s": optimized_time,
        "time_delta_pct": ((optimized_time - baseline_time) / baseline_time) * 100.0,
        "baseline_peak_bytes": float(baseline_peak),
        "optimized_peak_bytes": float(optimized_peak),
        "memory_delta_pct": ((optimized_peak - baseline_peak) / baseline_peak) * 100.0
        if baseline_peak
        else 0.0,
    }


def benchmark_flow_checkpointing() -> dict[str, float]:
    base_cfg = _make_config(dropout=0.0, use_gradient_checkpointing=False)
    ckpt_cfg = _make_config(dropout=0.0, use_gradient_checkpointing=True)
    model_plain = Flow_FEDformer(base_cfg)
    model_ckpt = Flow_FEDformer(ckpt_cfg)
    model_ckpt.load_state_dict(model_plain.state_dict())
    model_plain.train()
    model_ckpt.train()
    batch = _make_batch(base_cfg)

    def _run_log_prob(model: Flow_FEDformer) -> torch.Tensor:
        return model(batch["x_enc"], batch["x_dec"], batch["x_regime"]).log_prob(
            batch["y_true"]
        )

    lp_plain, baseline_time, baseline_peak = _measure(_run_log_prob, model_plain)
    lp_ckpt, optimized_time, optimized_peak = _measure(_run_log_prob, model_ckpt)
    assert torch.allclose(lp_plain, lp_ckpt, atol=1e-6)
    return {
        "baseline_time_s": baseline_time,
        "optimized_time_s": optimized_time,
        "time_delta_pct": ((optimized_time - baseline_time) / baseline_time) * 100.0,
        "baseline_peak_bytes": float(baseline_peak),
        "optimized_peak_bytes": float(optimized_peak),
        "memory_delta_pct": ((optimized_peak - baseline_peak) / baseline_peak) * 100.0
        if baseline_peak
        else 0.0,
    }


def run_all_benchmarks() -> dict[str, dict[str, float]]:
    return {
        "mc_dropout": benchmark_mc_dropout(),
        "fourier_modes": benchmark_fourier_modes(),
        "flow_checkpointing": benchmark_flow_checkpointing(),
    }


if __name__ == "__main__":
    print(json.dumps(run_all_benchmarks(), indent=2, sort_keys=True))
