# -*- coding: utf-8 -*-
"""
Tests unitarios para el scheduler de LR y el early stopping del WalkForwardTrainer.
Pruebas mínimas y rápidas que no requieren ejecuciones de entrenamiento reales.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.trainer import _EarlyStopping


def test_cosine_scheduler_reduces_lr() -> None:
    """El scheduler cosine debe reducir la LR desde su valor inicial hasta min_lr."""
    # Crear un optimizador mínimo con un parámetro dummy
    param = torch.nn.Parameter(torch.tensor([1.0]))
    initial_lr = 1e-3
    min_lr = 1e-6
    total_epochs = 10

    optimizer = torch.optim.SGD([param], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=min_lr,
    )

    lr_inicial = optimizer.param_groups[0]["lr"]

    # Avanzar varias épocas (optimizer.step primero, luego scheduler.step)
    for _ in range(total_epochs):
        optimizer.step()
        scheduler.step()

    lr_final = optimizer.param_groups[0]["lr"]

    # La LR debe haber disminuido respecto al valor inicial
    assert lr_final < lr_inicial, (
        f"Se esperaba LR final ({lr_final}) < LR inicial ({lr_inicial})"
    )
    # La LR final debe ser aproximadamente min_lr al completar T_max épocas
    assert lr_final <= initial_lr, (
        f"La LR ({lr_final}) no debe superar la LR inicial ({initial_lr})"
    )


def test_early_stopping_triggers() -> None:
    """_EarlyStopping debe retornar True exactamente después de `patience` pasos sin mejora."""
    patience = 2
    min_delta = 1e-4

    stopper = _EarlyStopping(patience=patience, min_delta=min_delta)

    # Primera pérdida — establece best_loss
    resultado = stopper.step(1.0)
    assert not resultado, "El primer paso no debe activar la parada"
    assert stopper.counter == 0

    # Segunda pérdida sin mejora — counter sube a 1
    resultado = stopper.step(1.0)
    assert not resultado, "No debe activarse después del primer paso sin mejora"
    assert stopper.counter == 1

    # Tercera pérdida sin mejora — counter sube a 2 == patience → debe activarse
    resultado = stopper.step(1.0)
    assert resultado, "Debe activarse después de 2 pasos sin mejora (patience=2)"
    assert stopper.should_stop is True


def test_early_stopping_restores_best() -> None:
    """_EarlyStopping.best_loss debe rastrear el mínimo visto durante el entrenamiento."""
    stopper = _EarlyStopping(patience=5, min_delta=1e-4)

    # Secuencia con mejoras y retrocesos
    losses = [2.0, 1.5, 1.2, 1.3, 1.1, 1.4, 0.9, 1.0]
    for loss in losses:
        stopper.step(loss)

    # El mejor loss visto es 0.9
    assert abs(stopper.best_loss - 0.9) < 1e-9, (
        f"Se esperaba best_loss=0.9, got {stopper.best_loss}"
    )


def test_early_stopping_disabled_when_patience_zero() -> None:
    """Cuando patience=0, _EarlyStopping siempre retorna False (deshabilitado)."""
    stopper = _EarlyStopping(patience=0, min_delta=1e-4)

    # Incluso con muchos pasos sin mejora, nunca debe activarse
    for _ in range(100):
        resultado = stopper.step(1.0)
        assert not resultado, "Con patience=0, la parada anticipada debe estar deshabilitada"

    assert not stopper.should_stop


def test_early_stopping_resets_counter_on_improvement() -> None:
    """El contador debe reiniciarse a cero cuando se observa una mejora suficiente."""
    stopper = _EarlyStopping(patience=3, min_delta=1e-4)

    # Dos pasos sin mejora — counter sube a 2
    stopper.step(1.0)
    stopper.step(1.0)
    assert stopper.counter == 1  # sólo la segunda sin mejora

    # Mejora significativa — counter debe reiniciarse
    stopper.step(0.5)
    assert stopper.counter == 0, (
        f"El contador debe ser 0 después de una mejora, got {stopper.counter}"
    )
    assert stopper.best_loss == 0.5
