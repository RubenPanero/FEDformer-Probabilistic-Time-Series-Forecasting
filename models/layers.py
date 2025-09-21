# -*- coding: utf-8 -*-
"""
Componentes basicos del modelo FEDformer: capas de atencion y descomposicion.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, cast

import torch
from torch import nn
from torch.fft import irfft, rfft
from torch.nn.functional import conv1d, interpolate, pad



def _apply_conv1d(input_tensor: torch.Tensor, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Wrapper around torch.nn.functional.conv1d for static analysis."""
    conv = cast(Callable[..., torch.Tensor], conv1d)  # pylint: disable=not-callable
    return conv(input_tensor, weight, **kwargs)  # pylint: disable=not-callable


def _apply_rfft(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Wrapper around torch.fft.rfft for static analysis."""
    rfft_fn = cast(Callable[..., torch.Tensor], rfft)  # pylint: disable=not-callable
    return rfft_fn(x, dim=dim)  # pylint: disable=not-callable


def _apply_irfft(x: torch.Tensor, *, n: int, dim: int) -> torch.Tensor:
    """Wrapper around torch.fft.irfft for static analysis."""
    irfft_fn = cast(Callable[..., torch.Tensor], irfft)  # pylint: disable=not-callable
    return irfft_fn(x, n=n, dim=dim)  # pylint: disable=not-callable


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for AttentionLayer construction."""

    d_model: int
    n_heads: int
    seq_len: int
    modes: int
    dropout: float



class OptimizedSeriesDecomp(nn.Module):
    """Optimized decomposition with reduced memory footprint"""

    def __init__(self, kernel_sizes: List[int]) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input into seasonal and trend signals via moving averages."""
        x_t = x.transpose(1, 2)  # [batch, len, channels] -> [batch, channels, len]
        _, num_channels, _ = x_t.shape
        trends = []
        for kernel_size in self.kernel_sizes:
            if kernel_size <= 1:
                trends.append(x_t)
                continue
            left = (kernel_size - 1) // 2
            right = kernel_size - 1 - left
            x_padded = pad(
                x_t, (left, right), mode="replicate"
            )  # [batch, channels, len + left + right]
            weight = torch.ones(
                num_channels, 1, kernel_size, device=x_t.device, dtype=x_t.dtype
            ) / float(kernel_size)
            trend = _apply_conv1d(x_padded, weight, groups=num_channels)
            trends.append(trend)
        trend = torch.stack(trends).mean(0).transpose(1, 2)  # Back to [batch, len, channels]
        return x - trend, trend



class FourierAttention(nn.Module):
    """Memory-optimized Fourier attention with better initialization"""

    def __init__(self, d_keys: int, seq_len: int, modes: int = 64) -> None:
        super().__init__()
        self.modes = min(modes, max(1, seq_len // 2))
        # FIXED: Use register_buffer for proper device handling
        indices = torch.randperm(max(1, seq_len // 2))[: self.modes].sort()[0]
        self.register_buffer("index", indices)

        # OPTIMIZED: Stable weight initialization (separate real/imag)
        # Small std to avoid large FFT outputs
        std = 0.02 / max(1.0, math.sqrt(d_keys))
        self.weights_real = nn.Parameter(torch.randn(d_keys, d_keys, self.modes) * std)
        self.weights_imag = nn.Parameter(torch.randn(d_keys, d_keys, self.modes) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier attention over the provided multi-head representations."""
        _, _, seq_len, _ = x.shape
        x = x.transpose(-1, -2)

        x_ft = _apply_rfft(x, dim=-1)

        weights_c = torch.complex(self.weights_real, self.weights_imag)
        selected = x_ft[..., self.index]
        processed_modes = torch.einsum("bhei,eoi->bhoi", selected, weights_c)

        out_ft = torch.zeros_like(x_ft)
        out_ft[..., self.index] = processed_modes

        return _apply_irfft(out_ft, n=seq_len, dim=-1).transpose(-1, -2)



class AttentionLayer(nn.Module):
    """Enhanced attention layer with better memory management"""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_keys = config.d_model // config.n_heads
        self.fourier_attention = FourierAttention(self.d_keys, config.seq_len, config.modes)
        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Project queries/keys/values and apply Fourier attention."""
        # pylint: disable=too-many-locals
        batch_q, len_q, _ = q.shape
        batch_k, len_k, _ = k.shape

        q_heads = (
            self.query_proj(q)
            .view(batch_q, len_q, self.n_heads, self.d_keys)
            .transpose(1, 2)
        )
        k_heads = (
            self.key_proj(k)
            .view(batch_k, len_k, self.n_heads, self.d_keys)
            .transpose(1, 2)
        )
        v_heads = (
            self.key_proj(v)
            .view(batch_k, len_k, self.n_heads, self.d_keys)
            .transpose(1, 2)
        )

        if len_k != len_q:
            head_batch = batch_k * self.n_heads
            k_reshaped = (
                k_heads.transpose(1, 2).contiguous().view(head_batch, self.d_keys, len_k)
            )
            v_reshaped = (
                v_heads.transpose(1, 2).contiguous().view(head_batch, self.d_keys, len_k)
            )
            k_resampled = interpolate(
                k_reshaped, size=len_q, mode="linear", align_corners=False
            )
            v_resampled = interpolate(
                v_reshaped, size=len_q, mode="linear", align_corners=False
            )
            k_heads = (
                k_resampled.view(batch_k, self.n_heads, self.d_keys, len_q).transpose(2, 3)
            )
            v_heads = (
                v_resampled.view(batch_k, self.n_heads, self.d_keys, len_q).transpose(2, 3)
            )

        attn_out = self.fourier_attention(q_heads * k_heads)
        batch_out, len_out = q.shape[:2]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_out, len_out, -1)
        return self.dropout(self.out_proj(attn_out))
