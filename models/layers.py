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
        # OPTIMIZED: Compute moving average via grouped conv1d with explicit asymmetric padding
        # This ensures output length equals input length for both odd and even kernel sizes.
        x_t = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        B, C, L = x_t.shape
        trends = []
        for k in self.kernel_sizes:
            if k <= 1:
                trends.append(x_t)
                continue
            left = (k - 1) // 2
            right = k - 1 - left
            # Pad with replication to avoid artificial zeros at borders
            x_padded = F.pad(x_t, (left, right), mode='replicate')  # [B, C, L + left + right]
            # Create grouped conv weights: one filter per channel
            weight = torch.ones(C, 1, k, device=x_t.device, dtype=x_t.dtype) / float(k)
            # Apply grouped conv to compute per-channel moving average
            trend = F.conv1d(x_padded, weight, groups=C)
            # After conv, trend shape [B, C, L]
            trends.append(trend)
        trend = torch.stack(trends).mean(0).transpose(1, 2)  # Back to [B, L, C]
        return x - trend, trend


class FourierAttention(nn.Module):
    """Memory-optimized Fourier attention with better initialization"""
    def __init__(self, d_keys: int, seq_len: int, modes: int = 64):
        super().__init__()
        self.modes = min(modes, max(1, seq_len // 2))
        # FIXED: Use register_buffer for proper device handling
        indices = torch.randperm(max(1, seq_len // 2))[:self.modes].sort()[0]
        self.register_buffer('index', indices)
        
        # OPTIMIZED: Stable weight initialization (separate real/imag)
        # Small std to avoid large FFT outputs
        std = 0.02 / max(1.0, math.sqrt(d_keys))
        self.weights_real = nn.Parameter(torch.randn(d_keys, d_keys, self.modes) * std)
        self.weights_imag = nn.Parameter(torch.randn(d_keys, d_keys, self.modes) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, E = x.shape
        x = x.transpose(-1, -2)  # OPTIMIZED: More efficient than permute
        
        # OPTIMIZED: Sparse memory management for FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Build complex weights explicitly
        weights_c = torch.complex(self.weights_real, self.weights_imag)
        # selected modes safe indexing
        selected = x_ft[..., self.index]
        # Perform complex multiplication in a controlled manner
        processed_modes = torch.einsum('bhei,eoi->bhoi', selected, weights_c)
        
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., self.index] = processed_modes
        
        # If L was odd and rfft produced a different length, irfft with n=L ensures correct output length
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
        # Support cross-attention where query and key/value sequence lengths differ.
        B_q, L_q, _ = q.shape
        B_k, L_k, _ = k.shape

        q_h = self.query_proj(q).view(B_q, L_q, self.n_heads, self.d_keys).transpose(1, 2)
        k_h = self.key_proj(k).view(B_k, L_k, self.n_heads, self.d_keys).transpose(1, 2)
        v_h = self.key_proj(v).view(B_k, L_k, self.n_heads, self.d_keys).transpose(1, 2)

        # If key/value sequence length differs from query length, resample k_h and v_h to L_q
        if L_k != L_q:
            # interpolate along the sequence axis (dim=2)
            # reshape to (B*n_heads, d_keys, L_k) for interpolation
            bh = B_k * self.n_heads
            k_h_reshaped = k_h.transpose(1, 2).contiguous().view(bh, self.d_keys, L_k)
            v_h_reshaped = v_h.transpose(1, 2).contiguous().view(bh, self.d_keys, L_k)
            k_h_resampled = F.interpolate(k_h_reshaped, size=L_q, mode='linear', align_corners=False)
            v_h_resampled = F.interpolate(v_h_reshaped, size=L_q, mode='linear', align_corners=False)
            # restore shape to (B, n_heads, L_q, d_keys)
            k_h = k_h_resampled.view(B_k, self.n_heads, self.d_keys, L_q).transpose(2, 3)
            v_h = v_h_resampled.view(B_k, self.n_heads, self.d_keys, L_q).transpose(2, 3)

        # Now q_h and k_h share the same sequence length (L_q)
        attn_out = self.fourier_attention(q_h * k_h)
        # Use the query batch/length for final reshape
        B_out, L_out = q.shape[0], q.shape[1]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B_out, L_out, -1)
        return self.dropout(self.out_proj(attn_out))

