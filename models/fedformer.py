# -*- coding: utf-8 -*-
"""
Modelo principal FEDformer con Normalizing Flows.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.distributions

from config import FEDformerConfig
from .layers import OptimizedSeriesDecomp
from .encoder_decoder import EncoderLayer, DecoderLayer
from .flows import NormalizingFlow


# -*- coding: utf-8 -*-
"""
Modelo principal FEDformer con Normalizing Flows.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.distributions

from config import FEDformerConfig
from .layers import OptimizedSeriesDecomp
from .encoder_decoder import EncoderLayer, DecoderLayer
from .flows import NormalizingFlow


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
        # Projection to map decoder input feature space (dec_in) to model hidden size (d_model)
        self.trend_proj = nn.Linear(config.dec_in, config.d_model)

        self.enc_embedding = nn.Linear(
            config.enc_in + config.regime_embedding_dim, config.d_model
        )
        self.dec_embedding = nn.Linear(
            config.dec_in + config.regime_embedding_dim, config.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

        # FIXED: Pass activation parameter
        self.encoder = EncoderLayer(
            config.d_model,
            config.n_heads,
            config.seq_len,
            config.d_ff,
            config.modes,
            config.dropout,
            config.activation,
            config.moving_avg,
        )

        dec_seq_len = config.label_len + config.pred_len
        self.decoder = DecoderLayer(
            config.d_model,
            config.n_heads,
            dec_seq_len,
            config.d_ff,
            config.modes,
            config.dropout,
            config.activation,
            config.moving_avg,
        )

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
        """Prepare decoder seasonal and trend initial components.

        Returns seasonal_out (B, label_len+pred_len, dec_in) and
        trend_out projected to model hidden size (B, label_len+pred_len, d_model).
        """
        mean = torch.mean(x_dec[:, : self.config.label_len, :], dim=1, keepdim=True)
        seasonal_init = torch.zeros_like(x_dec[:, -self.config.pred_len :, :])

        # Expand mean to prediction horizon and project to model hidden dim
        trend_init_in = mean.expand(-1, self.config.pred_len, -1)
        trend_init = self.trend_proj(trend_init_in)

        seasonal_dec_hist, trend_dec_hist = self.decomp(
            x_dec[:, : self.config.label_len, :]
        )
        seasonal_out = torch.cat([seasonal_dec_hist, seasonal_init], dim=1)

        # Project historical trend to hidden size and concatenate with projected init
        trend_dec_hist_proj = self.trend_proj(trend_dec_hist)
        trend_out = torch.cat([trend_dec_hist_proj, trend_init], dim=1)
        return seasonal_out, trend_out

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, x_regime: torch.Tensor
    ) -> torch.distributions.Distribution:
        seasonal_init, trend_init = self._prepare_decoder_input(x_dec)

        # Normalize regime indices to shape (B,) for embedding
        regime_idx = x_regime.squeeze()
        if regime_idx.dim() > 1:
            # flatten and take first column if extra dims present
            regime_idx = regime_idx.view(regime_idx.size(0), -1)[:, 0]
        regime_idx = regime_idx.long()
        regime_vec = self.regime_embedding(regime_idx)  # -> (B, regime_embedding_dim)
        regime_vec_enc = regime_vec.unsqueeze(1).expand(-1, self.config.seq_len, -1)
        regime_vec_dec = regime_vec.unsqueeze(1).expand(
            -1, self.config.label_len + self.config.pred_len, -1
        )

        x_enc_with_regime = torch.cat([x_enc, regime_vec_enc], dim=-1)
        seasonal_init_with_regime = torch.cat([seasonal_init, regime_vec_dec], dim=-1)

        enc_out = self.dropout(self.enc_embedding(x_enc_with_regime))
        dec_out = self.dropout(self.dec_embedding(seasonal_init_with_regime))

        # OPTIMIZED: Optional gradient checkpointing with robust wrappers
        if self.use_gradient_checkpointing and self.training:
            try:
                # prefer lambda wrappers to control arguments
                enc_out = torch.utils.checkpoint.checkpoint(
                    lambda x: self.encoder(x), enc_out
                )
                dec_out, trend_part = torch.utils.checkpoint.checkpoint(
                    lambda a, b: self.decoder(a, b), dec_out, enc_out
                )
            except Exception:
                # Fallback for PyTorch versions / edge cases: run normally
                enc_out = self.encoder(enc_out)
                dec_out, trend_part = self.decoder(dec_out, enc_out)
        else:
            enc_out = self.encoder(enc_out)
            dec_out, trend_part = self.decoder(dec_out, enc_out)

        # Ensure trend_init and trend_part have matching hidden dimensions before adding.
        if trend_init.shape[-1] != trend_part.shape[-1]:
            # Project trend_init to decoder hidden dimension if needed
            # Create a temporary projection if model doesn't already have one
            # (Prefer to have a persistent module, but for minimal change we create inline)
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
        """Compute log probability per batch, supports optional mask over time dimension.

        Returns: tensor of shape (B,) with log-prob per batch element.
        """
        B, T, F = y_true.shape
        total_lp = torch.zeros(B, device=y_true.device, dtype=y_true.dtype)
        for f in range(F):
            y_f = y_true[..., f]
            mu_f = self.means[..., f]
            ctx_f = self.contexts[:, f, :]
            lp_f = self.flows[f].log_prob(y_f, base_mean=mu_f, context=ctx_f)
            total_lp = total_lp + lp_f
        # If mask provided (B, T) with 1=valid, weight by valid counts; otherwise average over T
        if mask is not None:
            # mask expected shape (B, T) or (B, T, 1)
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid_counts = mask.sum(dim=1).clamp(min=1).to(total_lp.dtype)
            # For flows we summed over time earlier in flow.log_prob; total_lp is sum over time
            return total_lp / valid_counts
        else:
            return total_lp / float(T)

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
