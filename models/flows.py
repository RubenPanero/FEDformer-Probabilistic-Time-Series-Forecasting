# -*- coding: utf-8 -*-
"""
Implementación de Normalizing Flows para el modelo FEDformer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AffineCouplingLayer(nn.Module):
    """Fixed affine coupling layer with proper device handling"""
    def __init__(self, d_model: int, hidden_dim: int, context_dim: int = 0):
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim
        cond_in = (d_model // 2) + (context_dim if context_dim > 0 else 0)
        self.conditioner = nn.Sequential(
            nn.Linear(cond_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=-1)
        if self.context_dim > 0 and context is not None:
            cond_in = torch.cat([x1, context], dim=-1)
        else:
            cond_in = x1
        s, t = self.conditioner(cond_in).chunk(2, dim=-1)
        s = torch.tanh(s)  # Stabilize scale
        y1 = x1
        y2 = x2 * s.exp() + t
        log_det_jacobian = s.sum(dim=-1)
        return torch.cat([y1, y2], dim=-1), log_det_jacobian

    def inverse(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=-1)
        if self.context_dim > 0 and context is not None:
            cond_in = torch.cat([y1, context], dim=-1)
        else:
            cond_in = y1
        s, t = self.conditioner(cond_in).chunk(2, dim=-1)
        s = torch.tanh(s)
        x1 = y1
        x2 = (y2 - t) * (-s).exp()
        return torch.cat([x1, x2], dim=-1)


class NormalizingFlow(nn.Module):
    """FIXED: Proper device handling for base distribution"""
    def __init__(self, n_layers: int, d_model: int, hidden_dim: int, context_dim: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCouplingLayer(d_model, hidden_dim, context_dim=context_dim) 
            for _ in range(n_layers)
        ])
        # FIXED: Use register_buffer for proper device handling
        self.register_buffer('base_mean', torch.zeros(d_model))
        self.register_buffer('base_std', torch.ones(d_model))
    
    @property
    def base_dist(self):
        """Base distribution that automatically handles device placement"""
        return torch.distributions.Normal(self.base_mean, self.base_std)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_jacobian = 0
        for layer in self.layers:
            x, ldj = layer(x, context=context)
            log_det_jacobian += ldj
        return x, log_det_jacobian

    def inverse(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in reversed(self.layers):
            z = layer.inverse(z, context=context)
        return z

    def log_prob(self, x: torch.Tensor, base_mean: Optional[torch.Tensor] = None, 
                 context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # OPTIMIZED: In-place operations where possible
        if base_mean is None:
            centered = x
        else:
            centered = x - base_mean
        z, log_det_jacobian = self.forward(centered, context=context)
        # Pre-compute log_prob more efficiently
        base_log_prob = self.base_dist.log_prob(z).sum(dim=-1)
        return base_log_prob + log_det_jacobian

    def sample(self, n_samples: int) -> torch.Tensor:
        z = self.base_dist.sample((n_samples,))
        return self.inverse(z)

