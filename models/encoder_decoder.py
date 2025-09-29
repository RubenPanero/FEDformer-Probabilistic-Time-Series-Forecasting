# -*- coding: utf-8 -*-
"""
Componentes Encoder y Decoder del modelo FEDformer.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.nn.functional import gelu, relu

from .layers import AttentionConfig, AttentionLayer, OptimizedSeriesDecomp


@dataclass(frozen=True)
class LayerConfig:
    """Configuration container for encoder and decoder layers."""

    # pylint: disable=too-many-instance-attributes

    d_model: int
    n_heads: int
    seq_len: int
    d_ff: int
    modes: int
    dropout: float
    activation: str
    moving_avg: List[int]


class EncoderLayer(nn.Module):
    """Optimized encoder layer with optional gradient checkpointing"""

    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        attention_cfg = AttentionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
            modes=config.modes,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleDict(
            {
                "attention": AttentionLayer(attention_cfg),
                "decomp": nn.ModuleList(
                    [
                        OptimizedSeriesDecomp(config.moving_avg),
                        OptimizedSeriesDecomp(config.moving_avg),
                    ]
                ),
                "conv": nn.ModuleList(
                    [
                        nn.Conv1d(config.d_model, config.d_ff, 1),
                        nn.Conv1d(config.d_ff, config.d_model, 1),
                    ]
                ),
                "norm": nn.ModuleList(
                    [
                        nn.LayerNorm(config.d_model),
                        nn.LayerNorm(config.d_model),
                    ]
                ),
                "dropout": nn.Dropout(config.dropout),
            }
        )
        self._use_gelu = config.activation == "gelu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply encoder self-attention with seasonal-trend decomposition."""
        # pylint: disable=too-many-locals
        attention = self.layers["attention"]
        decomp_layers = self.layers["decomp"]
        conv_layers = self.layers["conv"]
        norm_layers = self.layers["norm"]
        dropout = self.layers["dropout"]

        x_norm = norm_layers[0](x)
        attn_out = attention(x_norm, x_norm, x_norm)
        x, _ = decomp_layers[0](x + attn_out)

        x_norm2 = norm_layers[1](x)
        y = conv_layers[0](x_norm2.transpose(1, 2))
        y = gelu(y) if self._use_gelu else relu(y)  # pylint: disable=not-callable
        y = dropout(y)
        y = conv_layers[1](y)
        y = dropout(y).transpose(1, 2)
        res, _ = decomp_layers[1](x + y)
        return res


class DecoderLayer(nn.Module):
    """Optimized decoder layer"""

    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        attention_cfg = AttentionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
            modes=config.modes,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleDict(
            {
                "self_attention": AttentionLayer(attention_cfg),
                "cross_attention": AttentionLayer(attention_cfg),
                "decomp": nn.ModuleList(
                    [
                        OptimizedSeriesDecomp(config.moving_avg),
                        OptimizedSeriesDecomp(config.moving_avg),
                        OptimizedSeriesDecomp(config.moving_avg),
                    ]
                ),
                "conv": nn.ModuleList(
                    [
                        nn.Conv1d(config.d_model, config.d_ff, 1),
                        nn.Conv1d(config.d_ff, config.d_model, 1),
                    ]
                ),
                "norm": nn.ModuleList(
                    [
                        nn.LayerNorm(config.d_model),
                        nn.LayerNorm(config.d_model),
                        nn.LayerNorm(config.d_model),
                    ]
                ),
                "dropout": nn.Dropout(config.dropout),
            }
        )
        self._use_gelu = config.activation == "gelu"

    def forward(
        self, x: torch.Tensor, cross: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply decoder self/cross-attention and return residual/trend."""
        # pylint: disable=too-many-locals
        self_attn = self.layers["self_attention"]
        cross_attn = self.layers["cross_attention"]
        decomp_layers = self.layers["decomp"]
        conv_layers = self.layers["conv"]
        norm_layers = self.layers["norm"]
        dropout = self.layers["dropout"]

        x_norm = norm_layers[0](x)
        x_res, trend1 = decomp_layers[0](x + self_attn(x_norm, x_norm, x_norm))

        x_norm2 = norm_layers[1](x_res)
        cross_norm = norm_layers[2](cross)
        x_res, trend2 = decomp_layers[1](
            x_res + cross_attn(x_norm2, cross_norm, cross_norm)
        )

        y = conv_layers[0](x_res.transpose(1, 2))
        y = gelu(y) if self._use_gelu else relu(y)  # pylint: disable=not-callable
        y = dropout(y)
        y = conv_layers[1](y)
        y = dropout(y).transpose(1, 2)
        x_res, trend3 = decomp_layers[2](x_res + y)
        return x_res, trend1 + trend2 + trend3
