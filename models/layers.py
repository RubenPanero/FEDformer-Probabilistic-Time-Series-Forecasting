# -*- coding: utf-8 -*-
"""
Componentes básicos del modelo FEDformer: capas de atención y descomposición.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class OptimizedSeriesDecomp(nn.Module):
    """Optimized decomposition with reduced memory footprint"""
    def __init__(self, kernel_sizes: List[int]):
        super().__init__()
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # OPTIMIZED: Direct F.avg_pool1d instead of nn.AvgPool1d modules
        x_t = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        trends = []
        for k in self.kernel_sizes:
            trend = F.avg_pool1d(x_t, kernel_size=k, stride=1, 
                               padding=k//2, count_include_pad=False)
            trends.append(trend)
        trend = torch.stack(trends).mean(0).transpose(1, 2)  # Back to [B, L, C]
        return x - trend, trend


class FourierAttention(nn.Module):
    """Memory-optimized Fourier attention with better initialization"""
    def __init__(self, d_keys: int, seq_len: int, modes: int = 64):
        super().__init__()
        self.modes = min(modes, seq_len // 2)
        # FIXED: Use register_buffer for proper device handling
        indices = torch.randperm(seq_len // 2)[:self.modes].sort()[0]
        self.register_buffer('index', indices)
        
        # OPTIMIZED: Better weight initialization
        self.weights = nn.Parameter(
            torch.randn(d_keys, d_keys, self.modes, 2) / math.sqrt(d_keys)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, E = x.shape
        x = x.transpose(-1, -2)  # OPTIMIZED: More efficient than permute
        
        # OPTIMIZED: Sparse memory management for FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Work only with selected modes to reduce memory allocation
        weights_c = torch.view_as_complex(self.weights)
        selected_modes = x_ft[..., self.index]
        processed_modes = torch.einsum('bhei,eoi->bhoi', selected_modes, weights_c)
        
        # Create sparse output - only allocate when needed
        out_ft = torch.zeros(x_ft.shape, dtype=x_ft.dtype, device=x_ft.device)
        out_ft[..., self.index] = processed_modes
        
        return torch.fft.irfft(out_ft, n=L, dim=-1).transpose(-1, -2)


class AttentionLayer(nn.Module):
    """Enhanced attention layer with better memory management"""
    def __init__(self, d_model: int, n_heads: int, seq_len: int, modes: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.fourier_attention = FourierAttention(self.d_keys, seq_len, modes)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model) 
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, L, _ = q.shape
        q_h = self.query_proj(q).view(B, L, self.n_heads, self.d_keys).transpose(1, 2)
        k_h = self.key_proj(k).view(B, L, self.n_heads, self.d_keys).transpose(1, 2)
        
        attn_out = self.fourier_attention(q_h * k_h)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.dropout(self.out_proj(attn_out))

