# -*- coding: utf-8 -*-
"""
Utilidades para entrenamiento del modelo FEDformer.
"""

import logging
import torch
import torch.nn as nn
from utils import get_device

logger = logging.getLogger(__name__)
device = get_device()


def mc_dropout_inference(model, batch, n_samples=100, use_flow_sampling: bool = True):
    """Proper MC dropout inference with gradient management.

    If use_flow_sampling is True and the model returns a distribution with a
    .sample() method, draw stochastic samples from the distribution. Otherwise,
    fall back to the predictive mean under dropout noise.
    """
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    
    prev_mode = model.training
    model.apply(enable_dropout)
    
    x_enc = batch['x_enc'].to(device, non_blocking=True)
    x_dec = batch['x_dec'].to(device, non_blocking=True)
    x_regime = batch['x_regime'].to(device, non_blocking=True)
    
    samples = []
    # FIXED: Use torch.no_grad() to prevent memory leaks
    with torch.no_grad():
        for _ in range(n_samples):
            try:
                dist = model(x_enc, x_dec, x_regime)
                if use_flow_sampling and hasattr(dist, 'sample'):
                    s = dist.sample(1)  # [1, B, T, F] or [1, B, T]
                    # squeeze sample dimension
                    samples.append(s[0])
                else:
                    samples.append(dist.mean)
            except Exception as e:
                logger.warning(f"MC sample failed: {e}")
                # Return zeros if sampling fails
                if samples:
                    samples.append(torch.zeros_like(samples[0]))
                else:
                    dummy_shape = (x_enc.size(0), model.config.pred_len, model.config.c_out)
                    samples.append(torch.zeros(dummy_shape, device=device))
    
    if not samples:
        logger.error("All MC samples failed")
        dummy_shape = (x_enc.size(0), model.config.pred_len, model.config.c_out)
        out = torch.zeros((1,) + dummy_shape, device=device)
    else:
        out = torch.stack(samples)

    # Restore original mode
    model.train(prev_mode)
    return out

