# -*- coding: utf-8 -*-
"""
Funciones auxiliares y utilidades generales.
"""

import torch


def _select_amp_dtype() -> torch.dtype:
    """Select appropriate mixed precision dtype"""
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        return torch.float16
    except Exception:
        return torch.float16


def setup_cuda_optimizations():
    """Configure CUDA optimizations for better performance"""
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def get_device():
    """Get the appropriate device for computation"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

