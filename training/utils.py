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


def mc_dropout_inference(model, batch, n_samples=100):
    """FIXED: Proper MC dropout inference with gradient management"""
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    
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
        return torch.zeros((1,) + dummy_shape, device=device)
        
    return torch.stack(samples)

