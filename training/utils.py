# -*- coding: utf-8 -*-
"""
Utilidades para entrenamiento del modelo FEDformer.
Refactorizado a Python 3.10+ para garantizar typing nativo purificado y eficiencia PEP 8.
"""

import logging

import torch
from torch import nn

from utils import get_device

logger = logging.getLogger(__name__)
device = get_device()


def mc_dropout_inference(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    n_samples: int = 100,
    use_flow_sampling: bool = True,
) -> torch.Tensor:
    """Inferencia formal MC Dropout con manejo asertivo de gradientes.

    Si use_flow_sampling es True y el modelo expone una distribución capaz
    de muestrear (.sample()), extraeremos simulaciones finitas del ruido.
    De otro modo, el sistema recae predictivamente en la media esperada.
    """

    def enable_dropout(m: nn.Module) -> None:
        if isinstance(m, nn.Dropout):
            m.train()

    prev_mode = model.training
    model.apply(enable_dropout)

    x_enc = batch["x_enc"].to(device, non_blocking=True)
    x_dec = batch["x_dec"].to(device, non_blocking=True)
    x_regime = batch["x_regime"].to(device, non_blocking=True)

    samples: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            try:
                dist = model(x_enc, x_dec, x_regime)
                if use_flow_sampling and hasattr(dist, "sample"):
                    s = dist.sample(1)  # [1, B, T, F] or [1, B, T]
                    samples.append(s[0])
                else:
                    samples.append(dist.mean)
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "Fallo en el muestreo de distribución MC Dropout: %s", exc
                )
                if samples:
                    samples.append(torch.zeros_like(samples[0]))
                else:
                    # Acceso seguro al shape nativo asumiendo inicializado el config
                    pred_len = getattr(model, "config", None)
                    if pred_len is not None:
                        dummy_shape = (
                            int(x_enc.size(0)),
                            int(model.config.pred_len),  # type: ignore
                            int(model.config.c_out),  # type: ignore
                        )
                    else:
                        dummy_shape = (
                            int(x_enc.size(0)),
                            1,
                            1,
                        )  # Caída ciega defensiva
                    samples.append(torch.zeros(*dummy_shape, device=device))

    if not samples:
        logger.error("Se abortaron todos los muestreos de MonteCarlo.")
        pred_len = getattr(model, "config", None)
        if pred_len is not None:
            dummy_shape = (
                int(x_enc.size(0)),
                int(model.config.pred_len),  # type: ignore
                int(model.config.c_out),  # type: ignore
            )
        else:
            dummy_shape = (int(x_enc.size(0)), 1, 1)
        out = torch.zeros(1, *dummy_shape, device=device)
    else:
        out = torch.stack(samples)

    # Restauración inmutable del modo nativo del modelo
    if not prev_mode:
        model.eval()

    return out
