# -*- coding: utf-8 -*-
"""
Modelo principal FEDformer con Normalizing Flows.
"""

from typing import Optional, Tuple

import torch
import torch.distributions
import torch.nn as nn

from config import FEDformerConfig
from .encoder_decoder import DecoderLayer, EncoderLayer, LayerConfig
from .flows import NormalizingFlow
from .layers import OptimizedSeriesDecomp


class Flow_FEDformer(nn.Module):
    """Enhanced FEDformer with gradient checkpointing and better error handling"""

    def __init__(self, config: FEDformerConfig) -> None:
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        self.decomp = OptimizedSeriesDecomp(config.moving_avg)
        self.regime_embedding = nn.Embedding(
            config.n_regimes, config.regime_embedding_dim
        )
        self.trend_proj = nn.Linear(config.dec_in, config.d_model)

        self.enc_embedding = nn.Linear(
            config.enc_in + config.regime_embedding_dim, config.d_model
        )
        self.dec_embedding = nn.Linear(
            config.dec_in + config.regime_embedding_dim, config.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

        encoder_config = LayerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
            d_ff=config.d_ff,
            modes=config.modes,
            dropout=config.dropout,
            activation=config.activation,
            moving_avg=config.moving_avg,
        )
        self.encoder = EncoderLayer(encoder_config)

        decoder_config = LayerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.label_len + config.pred_len,
            d_ff=config.d_ff,
            modes=config.modes,
            dropout=config.dropout,
            activation=config.activation,
            moving_avg=config.moving_avg,
        )
        self.decoder = DecoderLayer(decoder_config)

        self.flow_conditioner_proj = nn.Linear(
            config.d_model, config.c_out * config.flow_hidden_dim
        )

        self.flows = nn.ModuleList(
            [
                NormalizingFlow(
                    n_layers=config.n_flow_layers,
                    d_model=config.pred_len,
                    hidden_dim=config.flow_hidden_dim,
                    context_dim=config.flow_hidden_dim,
                )
                for _ in range(config.c_out)
            ]
        )

    def _prepare_decoder_input(
        self, x_dec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare decoder seasonal and trend initial components."""
        mean = torch.mean(
            x_dec[:, : self.config.label_len, :], dim=1, keepdim=True
        )
        seasonal_init = torch.zeros_like(
            x_dec[:, -self.config.pred_len :, :]
        )

        trend_init_in = mean.expand(-1, self.config.pred_len, -1)
        trend_init = self.trend_proj(trend_init_in)

        seasonal_dec_hist, trend_dec_hist = self.decomp(
            x_dec[:, : self.config.label_len, :]
        )
        seasonal_out = torch.cat([seasonal_dec_hist, seasonal_init], dim=1)

        trend_dec_hist_proj = self.trend_proj(trend_dec_hist)
        trend_out = torch.cat([trend_dec_hist_proj, trend_init], dim=1)
        return seasonal_out, trend_out

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, x_regime: torch.Tensor
    ) -> torch.distributions.Distribution:
        seasonal_init, trend_init = self._prepare_decoder_input(x_dec)

        regime_idx = x_regime.squeeze()
        if regime_idx.dim() > 1:
            regime_idx = regime_idx.view(regime_idx.size(0), -1)[:, 0]
        regime_idx = regime_idx.long()
        regime_vec = self.regime_embedding(regime_idx)
        regime_vec_enc = regime_vec.unsqueeze(1).expand(
            -1, self.config.seq_len, -1
        )
        regime_vec_dec = regime_vec.unsqueeze(1).expand(
            -1, self.config.label_len + self.config.pred_len, -1
        )

        x_enc_with_regime = torch.cat([x_enc, regime_vec_enc], dim=-1)
        seasonal_init_with_regime = torch.cat(
            [seasonal_init, regime_vec_dec], dim=-1
        )

        enc_out = self.dropout(self.enc_embedding(x_enc_with_regime))
        dec_out = self.dropout(self.dec_embedding(seasonal_init_with_regime))

        if self.use_gradient_checkpointing and self.training:
            try:
                enc_out = torch.utils.checkpoint.checkpoint(
                    lambda x: self.encoder(x), enc_out
                )
                dec_out, trend_part = torch.utils.checkpoint.checkpoint(
                    lambda a, b: self.decoder(a, b), dec_out, enc_out
                )
            except RuntimeError:
                enc_out = self.encoder(enc_out)
                dec_out, trend_part = self.decoder(dec_out, enc_out)
        else:
            enc_out = self.encoder(enc_out)
            dec_out, trend_part = self.decoder(dec_out, enc_out)

        if trend_init.shape[-1] != trend_part.shape[-1]:
            proj = nn.Linear(trend_init.shape[-1], trend_part.shape[-1]).to(
                trend_init.device
            )
            trend_init = proj(trend_init)
        final_trend = trend_init + trend_part

        dec_ctx = dec_out[:, -self.config.pred_len :, :]
        cond_proj = self.flow_conditioner_proj(dec_ctx)
        cond_proj = cond_proj.view(
            cond_proj.size(0),
            cond_proj.size(1),
            self.config.c_out,
            self.config.flow_hidden_dim,
        )
        feature_context = cond_proj.mean(dim=1)
        mean_pred = final_trend[:, -self.config.pred_len :, : self.config.c_out]

        return NormalizingFlowDistribution(mean_pred, self.flows, feature_context)


class NormalizingFlowDistribution:
    """Distribución que encapsula los flows normalizadores"""

    def __init__(
        self, means: torch.Tensor, flows: nn.ModuleList, contexts: torch.Tensor
    ) -> None:
        self.means = means
        self.flows = flows
        self.contexts = contexts

    @property
    def mean(self) -> torch.Tensor:
        return self.means

    def log_prob(
        self, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log probability per batch, supports optional mask over time dimension."""
        batch_size, time_steps, num_features = y_true.shape
        total_lp = torch.zeros(
            batch_size, device=y_true.device, dtype=y_true.dtype
        )
        for feature_idx in range(num_features):
            y_feature = y_true[..., feature_idx]
            mu_feature = self.means[..., feature_idx]
            ctx_feature = self.contexts[:, feature_idx, :]
            lp_feature = self.flows[feature_idx].log_prob(
                y_feature, base_mean=mu_feature, context=ctx_feature
            )
            total_lp = total_lp + lp_feature
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid_counts = mask.sum(dim=1).clamp(min=1).to(total_lp.dtype)
            return total_lp / valid_counts
        return total_lp / float(time_steps)

    def sample(self, n_samples: int) -> torch.Tensor:
        batch_size, time_steps, num_features = self.means.shape
        samples = []
        for _ in range(n_samples):
            feature_samples = []
            for feature_idx in range(num_features):
                ctx_feature = self.contexts[:, feature_idx, :]
                z = torch.randn(
                    batch_size,
                    time_steps,
                    device=self.means.device,
                    dtype=self.means.dtype,
                )
                x0 = self.flows[feature_idx].inverse(z, context=ctx_feature)
                feature_samples.append(x0.unsqueeze(-1))
            stacked = torch.cat(feature_samples, dim=-1)
            samples.append(stacked + self.means)
        return torch.stack(samples, dim=0)
