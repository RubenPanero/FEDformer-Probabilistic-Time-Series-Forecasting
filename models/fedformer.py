# -*- coding: utf-8 -*-
"""
Modelo principal FEDformer con Normalizing Flows.
"""

import torch
import torch.nn as nn
from typing import Tuple
import torch.distributions

from config import FEDformerConfig
from .layers import OptimizedSeriesDecomp
from .encoder_decoder import EncoderLayer, DecoderLayer
from .flows import NormalizingFlow


class Flow_FEDformer(nn.Module):
    """Enhanced FEDformer with gradient checkpointing and better error handling"""
    def __init__(self, config: FEDformerConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        
        self.decomp = OptimizedSeriesDecomp(config.moving_avg)
        self.regime_embedding = nn.Embedding(config.n_regimes, config.regime_embedding_dim)
        
        self.enc_embedding = nn.Linear(config.enc_in + config.regime_embedding_dim, config.d_model)
        self.dec_embedding = nn.Linear(config.dec_in + config.regime_embedding_dim, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # FIXED: Pass activation parameter
        self.encoder = EncoderLayer(
            config.d_model, config.n_heads, config.seq_len, config.d_ff, 
            config.modes, config.dropout, config.activation, config.moving_avg
        )
        
        dec_seq_len = config.label_len + config.pred_len
        self.decoder = DecoderLayer(
            config.d_model, config.n_heads, dec_seq_len, config.d_ff,
            config.modes, config.dropout, config.activation, config.moving_avg
        )
        
        self.flow_conditioner_proj = nn.Linear(config.d_model, config.c_out * config.flow_hidden_dim)
        
        self.flows = nn.ModuleList([
            NormalizingFlow(
                n_layers=config.n_flow_layers,
                d_model=config.pred_len,
                hidden_dim=config.flow_hidden_dim,
                context_dim=config.flow_hidden_dim,
            )
            for _ in range(config.c_out)
        ])

    def _prepare_decoder_input(self, x_dec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.mean(x_dec[:, :self.config.label_len, :], dim=1, keepdim=True)
        seasonal_init = torch.zeros_like(x_dec[:, -self.config.pred_len:, :])
        trend_init = mean.expand(-1, self.config.pred_len, -1)
        seasonal_dec_hist, trend_dec_hist = self.decomp(x_dec[:, :self.config.label_len, :])
        seasonal_out = torch.cat([seasonal_dec_hist, seasonal_init], dim=1)
        trend_out = torch.cat([trend_dec_hist, trend_init], dim=1)
        return seasonal_out, trend_out
        
    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor, 
                x_regime: torch.Tensor) -> torch.distributions.Distribution:
        seasonal_init, trend_init = self._prepare_decoder_input(x_dec)
        
        regime_vec = self.regime_embedding(x_regime.squeeze(-1))
        regime_vec_enc = regime_vec.unsqueeze(1).expand(-1, self.config.seq_len, -1)
        regime_vec_dec = regime_vec.unsqueeze(1).expand(-1, self.config.label_len + self.config.pred_len, -1)
        
        x_enc_with_regime = torch.cat([x_enc, regime_vec_enc], dim=-1)
        seasonal_init_with_regime = torch.cat([seasonal_init, regime_vec_dec], dim=-1)
        
        enc_out = self.dropout(self.enc_embedding(x_enc_with_regime))
        dec_out = self.dropout(self.dec_embedding(seasonal_init_with_regime))
        
        # OPTIMIZED: Optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            enc_out = torch.utils.checkpoint.checkpoint(self.encoder, enc_out, use_reentrant=False)
            dec_out, trend_part = torch.utils.checkpoint.checkpoint(
                self.decoder, dec_out, enc_out, use_reentrant=False
            )
        else:
            enc_out = self.encoder(enc_out)
            dec_out, trend_part = self.decoder(dec_out, enc_out)

        final_trend = trend_init + trend_part
        
        dec_ctx = dec_out[:, -self.config.pred_len:, :]
        cond_proj = self.flow_conditioner_proj(dec_ctx)
        cond_proj = cond_proj.view(
            cond_proj.size(0), cond_proj.size(1), 
            self.config.c_out, self.config.flow_hidden_dim
        )
        feature_context = cond_proj.mean(dim=1)
        mean_pred = final_trend[:, -self.config.pred_len:, :self.config.c_out]

        return NormalFlowDistribution(mean_pred, self.flows, feature_context)


class NormalFlowDistribution:
    """Distribución que encapsula los flows normalizadores"""
    def __init__(self, means, flows, contexts):
        self.means = means
        self.flows = flows
        self.contexts = contexts

    @property
    def mean(self):
        return self.means

    def log_prob(self, y_true: torch.Tensor) -> torch.Tensor:
        B, T, F = y_true.shape
        total_lp = torch.zeros(B, device=y_true.device, dtype=y_true.dtype)
        for f in range(F):
            y_f = y_true[..., f]
            mu_f = self.means[..., f]
            ctx_f = self.contexts[:, f, :]
            lp_f = self.flows[f].log_prob(y_f, base_mean=mu_f, context=ctx_f)
            total_lp = total_lp + lp_f
        return total_lp / T

    def sample(self, n_samples: int) -> torch.Tensor:
        B, T, F = self.means.shape
        samples = []
        for _ in range(n_samples):
            sample_f_list = []
            for f in range(F):
                ctx_f = self.contexts[:, f, :]
                z = torch.randn(B, T, device=self.means.device, dtype=self.means.dtype)
                x0 = self.flows[f].inverse(z, context=ctx_f)
                sample_f_list.append(x0.unsqueeze(-1))
            x0_all = torch.cat(sample_f_list, dim=-1)
            samples.append(x0_all + self.means)
        return torch.stack(samples, dim=0)

