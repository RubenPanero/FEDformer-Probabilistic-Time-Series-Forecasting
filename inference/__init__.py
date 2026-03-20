"""API de inferencia para modelos canónicos FEDformer."""


def load_specialist(*args, **kwargs):
    """Carga un modelo canónico — lazy import para evitar import circular."""
    from inference.loader import load_specialist as _load

    return _load(*args, **kwargs)


def predict(*args, **kwargs):
    """Genera predicciones probabilísticas — lazy import."""
    from inference.predictor import predict as _predict  # pylint: disable=import-error,no-name-in-module

    return _predict(*args, **kwargs)


__all__ = ["load_specialist", "predict"]
