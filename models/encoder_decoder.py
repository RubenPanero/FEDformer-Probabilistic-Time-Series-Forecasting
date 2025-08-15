# -*- coding: utf-8 -*-
"""
Componentes Encoder y Decoder del modelo FEDformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .layers import OptimizedSeriesDecomp, AttentionLayer


class EncoderLayer(nn.Module):
    """Optimized encoder layer with optional gradient checkpointing"""
    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int, 
                 modes: int, dropout: float, activation: str, moving_avg: List[int]):
        super().__init__()
        self.attention = AttentionLayer(d_model, n_heads, seq_len, modes, dropout)
        self.decomp1 = OptimizedSeriesDecomp(moving_avg)
        self.decomp2 = OptimizedSeriesDecomp(moving_avg)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, x_norm, x_norm)
        x, _ = self.decomp1(x + attn_out)
        
        x_norm2 = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(x_norm2.transpose(1, 2))))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        res, _ = self.decomp2(x + y)
        return res


class DecoderLayer(nn.Module):
    """Optimized decoder layer"""
    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int,
                 modes: int, dropout: float, activation: str, moving_avg: List[int]):
        super().__init__()
        self.self_attention = AttentionLayer(d_model, n_heads, seq_len, modes, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, seq_len, modes, dropout)
        self.decomp1 = OptimizedSeriesDecomp(moving_avg)
        self.decomp2 = OptimizedSeriesDecomp(moving_avg)
        self.decomp3 = OptimizedSeriesDecomp(moving_avg)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        x_res, trend1 = self.decomp1(x + self.self_attention(x_norm, x_norm, x_norm))
        
        x_norm2 = self.norm2(x_res)
        cross_norm = self.norm3(cross)
        x_res, trend2 = self.decomp2(x_res + self.cross_attention(x_norm2, cross_norm, cross_norm))
        
        y = self.dropout(self.activation(self.conv1(x_res.transpose(1, 2))))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        x_res, trend3 = self.decomp3(x_res + y)
        return x_res, trend1 + trend2 + trend3

