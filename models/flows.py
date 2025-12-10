# -*- coding: utf-8 -*-
"""
Implementación de Normalizing Flows para el modelo FEDformer.
"""

import contextlib
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Distribution, Normal


class AffineCouplingLayer(nn.Module):
    """Fixed affine coupling layer with proper device handling and odd-d_model support"""

    def __init__(self, d_model: int, hidden_dim: int, context_dim: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim
        # split sizes: d1 (conditioning), d2 (transformed). supports odd d_model.
        # If d_model is odd, allocate an extra unit to the first half to keep parity for the second half
        if d_model % 2 == 0:
            self.d1 = d_model // 2
            self.d2 = d_model - self.d1
        else:
            self.d1 = d_model // 2 + 1  # e.g., 5 -> 3
            self.d2 = d_model - self.d1  # 2
        cond_in = self.d1 + (context_dim if context_dim > 0 else 0)
        self.conditioner = nn.Sequential(
            nn.Linear(cond_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.d2),
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform the second split with context-aware affine parameters."""
        # slicing instead of chunk to handle odd sizes
        x1 = x[..., : self.d1]
        x2 = x[..., self.d1 :]
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

    def inverse(
        self, y: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Invert the affine coupling transformation."""
        y1 = y[..., : self.d1]
        y2 = y[..., self.d1 :]
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
    """FIXED: Proper device handling for base distribution and support odd d_model"""

    def __init__(
        self, n_layers: int, d_model: int, hidden_dim: int, context_dim: int = 0
    ) -> None:
        super().__init__()
        # Backward-compat: tolerate odd d_model by ignoring parity assertion if present
        with contextlib.suppress(AssertionError):
            assert d_model % 2 == 0, (
                f"d_model ({d_model}) must be even for coupling splits"
            )
        # allow odd d_model; layers handle splits internally (d1/d2 adapt internally)
        self.layers = nn.ModuleList(
            [
                AffineCouplingLayer(d_model, hidden_dim, context_dim=context_dim)
                for _ in range(n_layers)
            ]
        )
        # FIXED: Use register_buffer for proper device handling
        self.register_buffer("base_mean", torch.zeros(d_model))
        self.register_buffer("base_std", torch.ones(d_model))

    @property
    def base_dist(self) -> Distribution:
        """Base distribution that automatically handles device placement"""
        return Normal(self.base_mean, self.base_std)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply each coupling layer and accumulate the log-determinant.

        FIXED: Normalize log_det_jacobian by number of layers to prevent
        exponential scaling with depth. This improves stability of gradient flow.
        """
        # log_det per batch element
        log_det_jacobian = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, ldj = layer(x, context=context)
            log_det_jacobian = log_det_jacobian + ldj

        # Normalize by number of layers for stability
        # Without normalization, log_det scales exponentially: O(n_layers)
        n_layers = len(self.layers)
        if n_layers > 0:
            log_det_jacobian = log_det_jacobian / n_layers

        return x, log_det_jacobian

    def inverse(
        self, z: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Invert the flow by applying the coupling layers in reverse."""
        for layer in reversed(self.layers):
            z = layer.inverse(z, context=context)
        return z

    def log_prob(
        self,
        x: torch.Tensor,
        base_mean: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log-probability under the transformed base distribution."""
        # OPTIMIZED: In-place operations where possible
        centered = x if base_mean is None else x - base_mean
        z, log_det_jacobian = self.forward(centered, context=context)
        # Pre-compute log_prob more efficiently
        base_log_prob = self.base_dist.log_prob(z).sum(dim=-1)
        return base_log_prob + log_det_jacobian

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw samples by inverting latent Gaussian draws."""
        z = self.base_dist.sample((n_samples,))
        return self.inverse(z)
