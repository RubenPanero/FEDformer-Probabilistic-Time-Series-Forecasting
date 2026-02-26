# FEDformer – Probabilistic Time Series Forecasting

Implementación de **FEDformer** (Frequency Enhanced Decomposed Transformer) con **Normalizing Flows** para predicción probabilística de series temporales, orientada a datos financieros.

## Stack

- Python 3.10+ · PyTorch 2.0+ · pandas · scikit-learn · W&B
- Virtual env: `.venv/` (activar con `source .venv/bin/activate`)

## Estructura clave

```
config.py                      # FEDformerConfig (dataclass anidado, auto-introspección de CSV)
main.py                        # CLI orchestrator (argparse → config → train → simulate)
models/
  fedformer.py                 # Flow_FEDformer: encoder-decoder + normalizing flows
  flows.py                     # AffineCouplingLayer + NormalizingFlow
  layers.py                    # FourierAttention, OptimizedSeriesDecomp
data/
  dataset.py                   # TimeSeriesDataset, RegimeDetector, PreprocessingPipeline
  financial_dataset_builder.py # Construye dataset OHLCV + indicadores técnicos
  alpha_vantage_client.py      # Cliente Alpha Vantage con retry logic
  vix_data.py                  # Descarga VIX via yfinance
training/
  trainer.py                   # WalkForwardTrainer (walk-forward backtesting)
  train_base_model.py          # Entrenamiento base con GOOGL
  sequential_finetuner.py      # Fine-tuning secuencial multi-ticker
simulations/
  risk.py                      # RiskSimulator: VaR, CVaR, Expected Shortfall
  portfolio.py                 # PortfolioSimulator: Sharpe, drawdown, Sortino
tests/                         # pytest (conftest.py, 8 archivos de test)
```

## Comandos

```bash
# Entrenar
python main.py --csv data/nvidia_stock_*.csv --targets "Close" --pred-len 24 --seq-len 96

# Fine-tuning
python main.py --csv data/financial_data.csv --targets "Close" \
  --finetune-from checkpoints/base_model.pt --freeze-backbone --finetune-lr 1e-5

# Construir dataset financiero
python data/financial_dataset_builder.py --symbol GOOGL --use_mock

# Tests rápidos (sin @slow)
pytest -q -m "not slow"

# Todos los tests
pytest -v

# Con cobertura
pytest --cov=. --cov-report=html
```

## Convenciones

- **Commits**: sin co-autoría de Claude. Mensajes en inglés (`fix:`, `feat:`, `chore:`).
- **Comentarios en código**: en español.
- **Configuración**: todo en `FEDformerConfig` (dataclass); no hardcodear parámetros.
- **Preprocesamiento**: siempre leakage-safe (scaler se ajusta solo en datos de entrenamiento).
- **Salidas del modelo**: objetos `Distribution` (`.mean`, `.log_prob`, `.sample()`), nunca escalares.
- **`pred_len` debe ser par** (requisito del affine coupling split).

## Arquitectura (flujo de datos)

```
CSV → TimeSeriesDataset (scale + regimes)
    → WalkForwardTrainer (n folds)
        → Flow_FEDformer: x_enc, x_dec, x_regime → Distribution
        → Loss: -log_prob(y)
    → preds, ground_truth, samples
    → RiskSimulator + PortfolioSimulator → métricas + plot
```

## Testing

- `pytest.ini`: `-q --ignore=reports`, markers: `slow`
- Fixtures globales en `tests/conftest.py`: `config`, `model_factory`, `synthetic_batch`
- `test_architecture_and_leakage.py`: verifica que no hay data leakage en splits walk-forward
- `test_flows.py`: verifica invertibilidad de los normalizing flows

## CI/CD (`.github/workflows/`)

- `ci.yml`: Ubuntu + Windows, Python 3.10–3.11 → smoke test + pytest sin `@slow`
- `pylint.yml`: errores E/F únicamente
- `ruff.yml`, `security.yml`, `compatibility.yml`

## Ramas

- `main`: rama estable (merge via PRs)
- `feature/model-improvements`: rama de desarrollo activa
