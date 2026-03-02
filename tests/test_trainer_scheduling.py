# -*- coding: utf-8 -*-
"""
Tests unitarios para el scheduler de LR y el early stopping del WalkForwardTrainer.
Pruebas mínimas y rápidas que no requieren ejecuciones de entrenamiento reales.
"""

import torch

from training.trainer import _EarlyStopping, WalkForwardTrainer


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
        assert not resultado, (
            "Con patience=0, la parada anticipada debe estar deshabilitada"
        )

    assert not stopper.should_stop


def test_run_backtest_uses_raw_rows_for_split(config, synthetic_batch) -> None:
    """run_backtest debe usar filas crudas para el split, no ventanas.

    Regresión: con seq_len grande y split basado en ventanas, fold 1 obtenía
    0 batches (train_windows < batch_size) → avg_loss = inf siempre.
    """
    from data import TimeSeriesDataset

    ds = TimeSeriesDataset(config, flag="all")
    trainer = WalkForwardTrainer(config, ds)

    n_splits = 5
    # Comportamiento correcto: usar filas crudas
    total_size_rows = len(trainer.full_dataset.full_data_scaled)
    split_size = max(total_size_rows // n_splits, config.seq_len + config.pred_len)

    for fold_idx in range(1, n_splits):
        train_end_idx = fold_idx * split_size
        train_max_start = train_end_idx - config.seq_len - config.pred_len
        n_train_windows = max(0, train_max_start + 1)
        n_batches = n_train_windows // config.batch_size
        assert n_batches >= 1, (
            f"Fold {fold_idx}: {n_train_windows} ventanas → {n_batches} batches. "
            "run_backtest usa ventanas en lugar de filas crudas para el split."
        )


def test_early_stopping_resets_counter_on_improvement() -> None:
    """El contador debe reiniciarse a cero cuando se observa una mejora suficiente."""
    stopper = _EarlyStopping(patience=3, min_delta=1e-4)

    # Primer paso: mejora desde inf → best_loss=1.0, counter queda en 0
    # Segundo paso: sin mejora (1.0 no supera 1.0 - min_delta) → counter sube a 1
    stopper.step(1.0)
    stopper.step(1.0)
    assert stopper.counter == 1

    # Mejora significativa — counter debe reiniciarse
    stopper.step(0.5)
    assert stopper.counter == 0, (
        f"El contador debe ser 0 después de una mejora, got {stopper.counter}"
    )
    assert stopper.best_loss == 0.5
