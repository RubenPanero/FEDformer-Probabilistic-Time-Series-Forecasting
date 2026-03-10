# FEDformer – Probabilistic Time Series Forecasting

Implementación de **FEDformer** (Frequency Enhanced Decomposed Transformer) con **Normalizing Flows** para predicción probabilística de series temporales, orientada a datos financieros.

## Project Overview

Este proyecto usa Python con `ruff` para linting (respetar límites de longitud de línea: 88 chars), `pytest` para testing (277+ tests), y YAML/Markdown para CI y documentación. Proyecto principal: FEDformer forecasting.

## Stack

- Python 3.10+ · PyTorch 2.0+ · pandas · scikit-learn · W&B
- Virtual env: `.venv/` (activar con `source .venv/bin/activate`)

## Estructura clave

```
config.py                      # FEDformerConfig (dataclass anidado) + apply_preset() + TRAINING_PRESETS
main.py                        # CLI orchestrator — soporta múltiples --csv (multi-ticker)
main_helpers.py                # Helpers de orquestación: parse, validate, simulate, export CSV
pyproject.toml                 # Config centralizada de ruff + pytest (no es paquete instalable)
AGENTS.md                      # Guías de contribución y notas de refactor para agentes
models/
  fedformer.py                 # Flow_FEDformer: encoder-decoder + normalizing flows
  encoder_decoder.py           # Capas encoder/decoder con nn.ModuleDict y AttentionConfig
  flows.py                     # AffineCouplingLayer + NormalizingFlow
  layers.py                    # FourierAttention, OptimizedSeriesDecomp
data/
  dataset.py                   # TimeSeriesDataset, RegimeDetector
  preprocessing.py             # PreprocessingPipeline (inverse_transform_targets, leakage-safe)
  financial_dataset_builder.py # Construye dataset OHLCV + indicadores técnicos
  alpha_vantage_client.py      # Cliente Alpha Vantage con retry logic
  vix_data.py                  # Descarga VIX via yfinance
training/
  trainer.py                   # WalkForwardTrainer → devuelve ForecastOutput
  forecast_output.py           # ForecastOutput: dataclass dual-space (scaled + real)
  train_base_model.py          # Entrenamiento base con GOOGL
  sequential_finetuner.py      # Fine-tuning secuencial multi-ticker
                               # CLI: --symbols-file, --main-epochs, --scheduler-type, --warmup-epochs, --patience, --min-delta, --return-transform, --metric-space, --time-features
                               # Rehearsal: implementado vía training/rehearsal_buffer.py + flags --rehearsal-k, --rehearsal-epochs, --rehearsal-lr-mult
  utils.py                     # mc_dropout_inference, utilidades de entrenamiento
simulations/
  risk.py                      # RiskSimulator: VaR, CVaR, Expected Shortfall
  portfolio.py                 # PortfolioSimulator: Sharpe, drawdown, Sortino
utils/
  helpers.py                   # get_device, setup_cuda_optimizations, AMP dtype selection
  metrics.py                   # MetricsTracker (seguimiento in-memory por época)
  calibration.py               # conformal_quantile (Conformal Prediction)
  probabilistic_metrics.py     # métricas puras: pinball_loss, crps_from_samples, empirical_coverage
  io_experiment.py             # serialize_config, build_run_manifest, save_probabilistic_metrics
  experiment_registry.py       # load_run_manifests, rank_runs, aggregate_seed_metrics, summarize_seed_stability
tune_hyperparams.py            # CLI Optuna: búsqueda de hiperparámetros con walk-forward
                               # CLI: --csv, --targets, --n-trials, --timeout, --study-name
                               #      --study-objective {sharpe,composite,multi-objective}
                               #      --composite-score-profile {balanced}
scripts/
  __init__.py                  # módulo vacío — necesario para importar scripts en tests
  run_ablation_matrix.py       # AblationJob, build_ablation_jobs, job_to_argv, run_ablation_job
  run_multi_seed.py            # run_single_seed, run_multi_seed_experiment — CLI: --seeds, --dry-run
tests/                         # pytest (conftest.py, 22 archivos de test — 277 fast + 7 @slow)
results/                       # CSVs exportados por --save-results (gitignored)
data/
  NVDA_features.csv            # Dataset permanente (1725 filas × 11 features, 2019–2026)
  GOOGL_features.csv           # Dataset GOOGL (mismo formato, 11 features)
  MSFT_features.csv            # Dataset MSFT  (1726 filas × 11 features, 2019–2026)
  AAPL_features.csv            # Dataset AAPL  (1726 filas × 11 features, 2019–2026)
  AMZN_features.csv            # Dataset AMZN  (mismo formato, 11 features)
  META_features.csv            # Dataset META  (mismo formato, 11 features)
  TSLA_features.csv            # Dataset TSLA  (mismo formato, 11 features)
variants.json                  # Variantes de ablación: dropout {0.05,0.1,0.2} × {log_return,none}
docs/
  plans/
    2026-03-09-validacion-corto-plazo.md  # Plan ablaciones + multi-seed + Optuna composite
  roadmap_multiticker.md       # Roadmap Fase 1–2 arquitectura multi-ticker
archivos auxiliares/
  backlog_pipeline_investigacion_probabilistica.md  # Backlog épicas 1–9 (COMPLETADO 2026-03-09)
  plan_ejecucion_pipeline_probabilistico.md         # Plan Épicas 1–9 (COMPLETADO 2026-03-09)
  session_log_20260309.txt     # Log sesión Épicas 5–9
  session_log_20260310.txt     # Log sesión validación corto plazo (ablaciones + multi-seed)
```

## Comandos

```bash
# NOTA: Linux Mint 22.3 — usar python3 en lugar de python

# Comando canónico NVDA (configuración validada — log-return space, n_splits=4, clip=0.5)
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" \
  --seq-len 96 --pred-len 20 --batch-size 64 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.5 \
  --save-results --no-show

# Comando canónico multi-ticker NVDA+GOOGL
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv data/GOOGL_features.csv \
  --targets "Close" --seq-len 96 --pred-len 20 --batch-size 64 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.5 \
  --save-results --no-show

# Alternativa: espacio de precios absolutos (baseline, peor generalización inter-folds)
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" \
  --seq-len 96 --pred-len 20 --batch-size 64 --save-results --no-show

# Fine-tuning
python3 main.py --csv data/financial_data.csv --targets "Close" \
  --finetune-from checkpoints/base_model.pt --freeze-backbone --finetune-lr 1e-5

# Construir dataset financiero
python3 data/financial_dataset_builder.py --symbol GOOGL --use_mock

# Tests rápidos (sin @slow)
pytest -q -m "not slow"

# Todos los tests
pytest -v

# Con cobertura
pytest --cov=. --cov-report=html

# Verificación pre-commit (paridad con CI) — obligatorio antes de cada commit
ruff check . --fix && ruff format . && pylint --errors-only models/ training/ data/ utils/ && pytest -q -m "not slow"

# Búsqueda de hiperparámetros con Optuna (score simple)
python3 tune_hyperparams.py --csv data/NVDA_features.csv --targets "Close" \
  --n-trials 50 --study-name nvda_study

# Optuna con score compuesto probabilístico
python3 tune_hyperparams.py --csv data/NVDA_features.csv --targets "Close" \
  --n-trials 50 --study-name nvda_study --study-objective composite

# Preset de entrenamiento (debug/cpu_safe/gpu_research/probabilistic_eval)
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" \
  --seq-len 96 --pred-len 20 --preset probabilistic_eval --save-results --no-show

# Ablación reproducible (--dry-run para ver comandos sin ejecutar)
python3 scripts/run_ablation_matrix.py --csv data/NVDA_features.csv \
  --targets Close --variants-json variants.json --dry-run

# Experimento multi-seed
python3 scripts/run_multi_seed.py --csv data/NVDA_features.csv \
  --seeds 42 123 7 --extra-args --seq-len 96 --pred-len 20 --batch-size 64
```

## Convenciones

- **Commits**: sin co-autoría de Claude. Mensajes en inglés (`fix:`, `feat:`, `chore:`).
- **Comentarios en código**: en español.
- **Configuración**: todo en `FEDformerConfig` (dataclass); no hardcodear parámetros.
  - Para subsecciones tipadas: `config.sections.model.transformer.dropout = 0.2`
- **Preprocesamiento**: siempre leakage-safe (scaler se ajusta solo en datos de entrenamiento).
- **Salidas del modelo**: objetos `Distribution` (`.mean`, `.log_prob`, `.sample()`), nunca escalares.
- **`pred_len` debe ser par** (requisito del affine coupling split).
- **Early stopping**: monitorea `val_loss` (no `train_loss`). `val_fraction=0.15` en `LoopSettings` reserva el último 15% del bloque train de cada fold. Defaults activos: `patience=5`, `min_delta=5e-3` (filtra ruido de folds pequeños), `gradient_accumulation_steps=2`. El checkpoint del mejor `val_loss` siempre se guarda **y se recarga antes de inferencia** independientemente de si el early stopping disparó o el loop agotó todas las épocas.
- **Salidas del entrenador**: `run_backtest()` retorna `ForecastOutput` (no tuplas). Tiene campos
  `preds_scaled/real`, `gt_scaled/real`, `samples_scaled/real`, `metric_space`, `return_transform`.
  Usar `.preds_for_metrics` y `.gt_for_metrics` para acceder al espacio configurado.
  `*_for_metrics` **siempre devuelve `*_real`**; `metric_space` controla qué contiene `preds_real` (vía `_inverse_transform_all`), no qué array selecciona la propiedad.
  `RiskSimulator` convierte automáticamente muestras en espacio de precios a retornos acumulados `(P_t - P_0) / P_0` cuando `return_transform="none"`. VaR en t=0 siempre es 0 (trivial pero correcto).
  `ForecastOutput.window_fold_ids`: `np.ndarray | None` shape `(n_windows,)` dtype int32 — fold de origen de cada ventana. Genera columna `fold` en `predictions_*.csv`. Es `None` en instancias construidas manualmente sin pasar el campo (retrocompatible).
- **`return_transform` + `metric_space`**: dos dimensiones independientes. `return_transform` controla qué ve el modelo (precios vs log-returns); `metric_space` controla qué contiene `preds_real` en `ForecastOutput`. Combinación recomendada para NVDA/tickers con gran drift de precio: `--return-transform log_return --metric-space returns`. El backend (preprocessing, trainer, RiskSimulator) implementa log_return completo; la CLI lo expone desde `feat/log-return-transform` (2026-03-03).
  - `metric_space="returns"` → `preds_real` son log-returns desescalados; `RiskSimulator` los usa directamente.
  - `metric_space="prices"` → `preds_real` reconstruye precios con `last_prices` del fold vía `_cumulative_returns_to_prices()`.

## Code Quality

- Siempre ejecutar `ruff check --fix` y `ruff format` después de editar archivos Python para evitar conflictos de linting/formatting.
- El pre-commit completo es: `ruff check . --fix && ruff format . && pylint --errors-only models/ training/ data/ utils/ && pytest -q -m "not slow"`

## Git Workflow

- **Sin co-autoría de Claude**: no agregar trailers `Co-Authored-By` en commits.
- **PRs manuales**: para crear PRs, proporcionar la URL de GitHub generada por `git push` para creación manual — no usar `gh` CLI salvo que se pida explícitamente (`gh` no está instalado).
- Commits atómicos con prefijo convencional en inglés: `fix:`, `feat:`, `chore:`.
- Features en rama corta → PR pequeño → merge a `main`. Sin commits directos a `main`.

## Estándares de desarrollo Python

**Tipado:**
- Type hints en todas las funciones públicas; sintaxis 3.10+: `X | Y`, no `Optional[X]`
- Dataclasses para contenedores de datos (patrón establecido: `ForecastOutput`, `FEDformerConfig`)
- `np.ndarray` y `torch.Tensor` con shapes documentados en docstrings cuando no son obvios

**Estilo (PEP 8 — enforced por `ruff`):**
- `ruff` es el linter/formatter autoritativo (CI lo impone); longitud de línea: 88 chars
- Nomenclatura: `snake_case` funciones/variables, `PascalCase` clases, `UPPER_CASE` constantes
- Imports ordenados: stdlib → third-party → local (ruff/isort lo gestiona)
- `logging` en lugar de `print()` en todo código de producción
- f-strings exclusivamente; no `.format()` ni `%`
- Sin argumentos mutables por defecto; sin lógica en `__init__.py`
- Docstrings en español en funciones públicas (formato Google-style una línea o bloque)

**Calidad:**
- Toda función pública nueva requiere al menos un test unitario en `tests/`
- Tests deben ser independientes del sistema de archivos (usar fixtures de `conftest.py`)
- Mocks para llamadas externas (Alpha Vantage, yfinance) — nunca red real en tests
  - Patrón para `financial_dataset_builder`: `patch("yfinance.download", return_value=ohlcv_df)` + `patch("data.vix_data.VixDataFetcher.get_vix_data", return_value=vix_df)`. El import inline de yfinance en `_fetch_ohlcv` se parchea a nivel de módulo `yfinance`.
- Cobertura mínima esperada para módulos nuevos: ramas principales cubiertas

**Seguridad:**
- `security.yml` ejecuta bandit; no usar `eval()`, `pickle` sin validación, ni `shell=True`

## Flujo de trabajo (Agile/Scrum)

**Antes de escribir código:**
- Identificar el caso de uso concreto (user story) y su Definition of Done
- Escribir o actualizar el test unitario primero (TDD), luego implementar

**Antes de cada commit:**
- Ejecutar verificación pre-commit completa (ver comando arriba)
- Commit atómico: un cambio lógico por commit, mensaje en inglés con prefijo convencional

**Definition of Done por tarea:**
1. Tests pasan (`pytest -q -m "not slow"`)
2. Linting verde (`ruff check . && pylint --errors-only ...`)
3. Sin regresiones en tests existentes
4. CLAUDE.md actualizado si se añaden archivos, APIs o convenciones nuevas

**Ramas y PRs:**
- Features en rama corta → PR pequeño y enfocado → merge a `main`
- Sin commits directos a `main`; PRs revisados antes de merge
- Una rama = una feature o fix; no acumular cambios no relacionados

## MLOps

**Reproducibilidad:**
- Toda ejecución usa `config.seed` (propagado a PyTorch, NumPy y DataLoader workers)
- `FEDformerConfig` es serializable; guardar config junto al checkpoint para reproducir experimentos
- Checkpoints: `checkpoints/best_model_fold_N.pt` — diccionario con `model_state_dict` + `config`

**Experiment tracking (W&B):**
- W&B se inicializa automáticamente en `WalkForwardTrainer` si está instalado (falla silenciosamente si no)
- Loguear siempre: hiperparámetros del config, pérdida por época, métricas por fold
- No loguear tensores crudos ni artefactos sin comprimir (coste de almacenamiento)

**Versionado de modelos:**
- Un checkpoint por fold: `best_model_fold_0.pt`, `best_model_fold_1.pt`, ...
- Fine-tuning siempre parte de un checkpoint base explícito (`--finetune-from`)
- Usar `weights_only=True` en `torch.load()` (seguridad — ya implementado en trainer)

**Validación temporal (anti-leakage):**
- Walk-forward es la única estrategia válida para datos financieros; no usar K-fold aleatorio
- El scaler se ajusta solo en el split de entrenamiento de cada fold (ver `PreprocessingPipeline`)

**Monitoreo:**
- `utils/metrics.py → MetricsTracker`: seguimiento in-memory con contexto de fold. `log_metrics(metrics, step, fold=0)` almacena `(fold, step, value)`; `to_dataframe()` exporta columnas `fold, epoch, metric, value`.
- `--save-results` exporta 7 artefactos con timestamp compartido: `predictions_*.csv`, `risk_metrics_*.csv`, `portfolio_metrics_*.csv`, `training_history_*.csv`, `run_manifest_*.json`, `probabilistic_metrics_*.csv`, `fold_metrics_*.csv`.
- `utils/calibration.py → conformal_quantile`: calibración post-hoc de intervalos de predicción
- `utils/experiment_registry.py → load_run_manifests / rank_runs / aggregate_seed_metrics / summarize_seed_stability`: consolidar y comparar corridas a partir de los manifiestos JSON.

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

- Después de modificar cualquier archivo Python fuente, ejecutar la suite completa con `pytest` antes de hacer commit.
- `pyproject.toml`: sección `[tool.pytest.ini_options]` — `-q --ignore=reports`, markers: `slow`
- Fixtures globales en `tests/conftest.py`: `config`, `model_factory`, `synthetic_batch`
- `test_architecture_and_leakage.py`: verifica que no hay data leakage en splits walk-forward
- `test_flows.py`: verifica invertibilidad de los normalizing flows
- `test_forecast_output.py`: verifica API dual-space de `ForecastOutput` + helpers p10/p50/p90
- `test_preprocessing_pipeline.py`: verifica `PreprocessingPipeline` e `inverse_transform`
- `test_trainer_scheduling.py`: verifica LR scheduling, early stopping y `_eval_epoch`
- `test_dataset.py`: carga, escala, inversión, shapes y no-solapamiento de splits 70/20/10
- `test_encoder_decoder.py`: forward pass, shapes y gradient flow de EncoderLayer/DecoderLayer
- `test_utils.py`: helpers (set_seed, get_device), calibration y MetricsTracker
- `test_trainer_integration.py` (`@slow`): run_backtest() end-to-end con ForecastOutput
- `test_fedformer_basic.py`: forward pass básico de Flow_FEDformer (shapes, distribution output)
- `test_finetune.py`: freeze-backbone y fine-tuning desde checkpoint
- `test_reproducibility.py`: determinismo con config.seed fijo
- `test_simulations.py`: RiskSimulator y PortfolioSimulator (VaR, CVaR, Sharpe)
- `test_probabilistic_metrics.py`: pinball_loss, crps_from_samples, empirical_coverage, calibration_gap
- `test_checkpoint_selection.py`: monitor_metric/mode en early stopping y selección de checkpoint
- `test_experiment_outputs.py`: run_manifest JSON, probabilistic_metrics CSV, fold_metrics CSV
- `test_experiment_registry.py`: load_run_manifests, rank_runs, build_experiment_table
- `test_config_presets.py`: apply_preset para los 4 presets, prioridad preset < CLI override
- `test_ablation_matrix.py`: AblationJob, build_ablation_jobs, job_to_argv (subprocess mockeado)
- `test_seed_aggregation.py`: aggregate_seed_metrics, summarize_seed_stability

## CI/CD (`.github/workflows/`)

- `ci.yml`: Ubuntu + Windows, Python 3.10–3.11 → smoke test + pytest sin `@slow`
- `pylint.yml`: errores E/F únicamente
- `ruff.yml`, `security.yml`, `compatibility.yml`

## Ramas

- `main`: rama estable y activa — incluye Optuna, rehearsal buffer, model registry, log-return transform
- Todas las features anteriores mergeadas: `feat/log-return-transform`, `feature/optuna`, `fix/last-prices-leakage`

## Hoja de ruta activa

- **Épicas 1–9 COMPLETADAS** (2026-03-09) — commits `9810be1`→`623c5b1` en `main`.
- Plan histórico: `archivos auxiliares/plan_ejecucion_pipeline_probabilistico.md`
- Backlog fuente: `archivos auxiliares/backlog_pipeline_investigacion_probabilistica.md`
- **Próximos pasos inmediatos** (sesión 2026-03-10):
  1. ✅ Re-establecer baseline canónico NVDA post-fix `00f119c`: Sharpe **+0.609** (seed=42, 2026-03-10). MEMORY.md + model_registry.json + CLAUDE.md actualizados.
  2. Tarea C: Optuna con `--study-objective composite` (30 trials) — comparar vs sharpe puro.
  3. Tarea D: Comparativa Optuna composite vs sharpe puro en CLAUDE.md (pendiente tras Tarea C).
- **Plan de validación corto plazo**: `docs/plans/2026-03-09-validacion-corto-plazo.md`

## Entrenamiento headless

- `plt.show()` bloquea indefinidamente en subprocesos aunque `DISPLAY` esté definido (socket X11 inaccesible desde background).
- Siempre usar `MPLBACKEND=Agg python3 main.py ... --no-show` para ejecuciones no interactivas.
- Comando NVDA validado: `MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" --seq-len 96 --pred-len 20 --batch-size 64 --save-results --no-show`
- Sin `--epochs`: usa el default de `LoopSettings` (20 épocas). Pasar `--epochs N` solo para override explícito.
- Dataset NVDA permanente: `data/NVDA_features.csv` (1725 filas × 11 features, 2019–2026).
- **Segmentación canónica**: `seq_len=96, n_splits=4` → split_size=431, fold-1 con 4.2 batches. `n_splits=5` descartado (fold-1 = 3.1 batches, margen +1% sobre split_min=342).
- **Benchmarks NVDA canónicos** (configuración validada `log_return + n_splits=4 + clip=0.5`):
  - vs precio+n_splits=5 (2026-03-03): Sharpe −0.26→**+0.61** · Sortino −0.41→**+0.99** · val loss uniforme 1.29–1.62 · VaR 5.3% / CVaR 7.0%
  - vs clip=1.0 (2026-03-04): Sharpe +0.61→**+0.653** · Sortino +0.99→**+1.050** · fold-2 ratio 1.279→**1.115** (reducción overfitting)
  - ~~⚠️ DISCONTINUIDAD post-`00f119c`~~ **RESUELTA (2026-03-10)**: baseline post-fix re-establecido con seed=42.
    **Nuevo baseline canónico post-`00f119c`**: Sharpe **+0.609** · Sortino **+0.993** · VaR 5.35% · CVaR 7.07% · MaxDD −72.69% · Vol 2.47%
    El +0.653 (pre-fix) ya no es la referencia válida — usar **+0.609** para todas las comparaciones futuras.
    model_registry.json actualizado manualmente con este valor (2026-03-10).
  - Ablación dropout × return_transform (2026-03-10, cpu_safe): `log_return` supera `none` en promedio +0.701 Sharpe. Ranking de dropout ruidoso en run único (alta varianza NF). `dropout_020_logret` mejor en este run (0.792) pero contradice serie B2/C1/C2 — se necesita multi-run para confirmar.
  - Multi-seed NVDA post-fix (seeds 42,123,7,456, 2026-03-10): mean=+0.369 · std=0.546 · rango [−0.348, +0.892]. Solo 2/4 seeds con Sharpe > 0.5. **Alta varianza inter-seed confirmada**. Baseline seed=42 re-establecido: **+0.609**.
- `metric_space="returns"` con `return_transform="log_return"`: VaR/CVaR en unidades de retorno (5–7%). Con `metric_space="prices"` se reconstruyen precios vía `_cumulative_returns_to_prices(last_prices, log_return)`.

## Bugs resueltos (fix/last-prices-leakage, 2026-03-03)

5 bugs corregidos en `data/preprocessing.py` y `training/trainer.py`: winsorize de targets,
leakage en `last_prices`/drift-check, scaler incorrecto en inverse_transform, y checkpoint no
recargado al agotar épocas. Ver `git log --oneline -10` para detalles por commit.

## Gotchas

- `FEDformerConfig` usa `enc_in`/`dec_in` para dimensiones de entrada (no `num_features`). Config por defecto: `seq_len=10, pred_len=5` (genera warnings); usar `seq_len=96, pred_len=24` para experimentos reales.
- `pytest -q` no muestra la línea de resumen con el hook actual — usar `pytest -v` para ver `N passed`.
- `build_financial_dataset(symbol, output_dir, use_mock)` es una **función**, no una clase. Tests deben llamarla directamente; no existe `FinancialDatasetBuilder`.
- `label_len` default en el código es **5** (no 63 como sugiere `plan_entrenamiento.md`).
- `len(TimeSeriesDataset)` retorna ventanas, no filas — usar `len(dataset.full_data_scaled)` para splits walk-forward.
- `torch.fft.rfft` no soporta bfloat16 (AMP en GPUs Ada Lovelace); `_apply_rfft` en `models/layers.py` castea a float32 y `FourierAttention.forward` restaura `orig_dtype` con `.to(orig_dtype)` tras irfft.
- `torch.compile(mode="max-autotune")` genera NaN en GPUs con <40 SMs (RTX 4050 tiene 20); el trainer degrada automáticamente a modo eager.
- Config anidado correcto: `config.sections.preprocessing.return_transform` (no `config.preprocessing.*`).
- `df.index.tz_localize(None)` lanza `TypeError: Already tz-naive` con yfinance moderno (retorna índice tz-naive). Usar siempre `if df.index.tz is not None: df.index = df.index.tz_localize(None)` (patrón de `vix_data.py`).
- `logging.basicConfig()` a nivel de módulo es anti-patrón pre-existente en `data/vix_data.py` y `data/alpha_vantage_client.py` — si se modifican esos archivos, corregirlo a `logger = logging.getLogger(__name__)` únicamente.
- **`drop_last=False` en train_loader**: fold 1 con `seq_len=252` tiene 63 ventanas < `batch_size=64`; `drop_last=False` (ya aplicado) evita 0 batches → `train_loss=inf`.
- **Mínimo de datos en fold-1**: `split_min = seq_len + pred_len + ⌈(3 × batch_size) / (1 − val_frac)⌉`. Con defaults (seq=96, pred=20, batch=64, val=0.15): `split_min=342`. Para `n_splits=5` y 1725 filas: fold-1 = 3.1 batches ✓. Para tickers heterogéneos: `n_splits = total_filas // split_min`.
- **Tests con `model_factory` que llaman a `_prepare_batch`**: los tensores van a `device` (CUDA si disponible) pero `model_factory()` crea el modelo en CPU. Usar `model_factory().to(get_device())` en esos tests.
- **`_set_split_view`** es solo para uso standalone/tests con `flag="train/val/test"`. El pipeline walk-forward usa siempre `flag="all"` + `Subset` con índices calculados por `WalkForwardTrainer`; esta vista no interviene en ese flujo.
- **`lr_lambda` debe usar `math.cos`/`math.pi`** (nunca `np`): `np.cos()` devuelve `numpy.float64` → LambdaLR lo escribe en `param_groups['lr']` → `torch.load(weights_only=True)` falla al deserializar. Usar `math.cos(math.pi * x)` en toda función de scheduling.
- **Retrocompatibilidad de checkpoints con numpy escalares**: usar `torch.serialization.safe_globals([numpy._core.multiarray.scalar, np.float64, np.float32, np.int64, np.int32])` al cargar checkpoints antiguos con `weights_only=True`. `numpy._core` es privado → añadir `# pylint: disable=no-member` en esa línea.
- **Al añadir flags a `main.py`**: actualizar el `argparse.Namespace` en `tests/test_finetune.py` (fixture de `test_create_config_wires_return_transform_and_metric_space`) con los nuevos atributos a `None`; si no, `_create_config()` lanza `AttributeError`.
- **`sequential_finetuner.py` no tiene `--save-results`**: para fine-tuning con CSVs comparables en `results/`, usar `main.py --finetune-from checkpoints/best_model_fold_N.pt --finetune-lr 1e-5`.
- **Dropout óptimo definitivo (Serie Exp B2/C1/C2 — CERRADO)**: `dropout=0.1` es el óptimo confirmado en este espacio de hiperparámetros. Configuración canónica: `dropout=0.1, weight_decay=1e-5, scheduler=none, epochs=20`. `dropout=0.2` sobre-regulariza (Sharpe NVDA +0.61→+0.22); no probar `0.15` — baseline ya es óptimo. Run de referencia: `results/training_history_20260303_192845_NVDA_features.csv`.
- **`gradient_clip_norm` expuesto en CLI** (`--gradient-clip-norm`, default config=1.0). Valor óptimo validado: **0.5** (Sharpe +0.61→+0.653, fold-2 ratio 1.279→1.115). Usar `--gradient-clip-norm 0.5` en todos los runs canónicos.
- **`_optimizer_step` en `trainer.py` es `@staticmethod`**: no tiene `self`; pasar valores de config (ej. `clip_norm`) como parámetros explícitos al call site en `_train_epoch`. No intentar usar `self.config.*` dentro del método.
- **Subagentes para experimentos de entrenamiento**: requieren permiso Bash explícito. Sin él devuelven "necesito acceso Bash" sin ejecutar nada. Lanzar entrenamientos directamente con `Bash(run_in_background=true)` desde el contexto principal.
- **`last_prices` boundary en `preprocessing.py`**: `df.iloc[cutoff]` es el precio correcto en la frontera train/test — no es leakage. Está implícito en el último retorno de entrenamiento `log(P[cutoff]/P[cutoff-1])`. Test de regresión: `test_last_prices_is_boundary_anchor`.
- **`gh` CLI no está instalado**: crear PRs manualmente desde la URL que imprime `git push` (`https://github.com/.../pull/new/BRANCH`).
- **Smoke test en bash / pickling DataLoader RESUELTO** (`00f119c`, 2026-03-10): `_worker_init_fn` era closure no-picklable con `multiprocessing_context="spawn"`. Reemplazado por `_seed_worker(worker_id, base_seed)` a nivel de módulo + `functools.partial`. Tests y subprocesos ahora funcionan sin `--preset cpu_safe`. ⚠️ Este fix introduce **discontinuidad de reproducibilidad** (ver Benchmarks NVDA arriba).
- **`_make_loader` sin fix de pickling** (follow-up pendiente): el método `_make_loader` en `trainer.py` (usado en paths de rehearsal y fine-tuning) no recibió el `worker_init_fn` de `_seed_worker`. Si `num_workers > 0` en esos paths, los workers no tienen semilla determinista. Bajo carga normal (sin rehearsal activo), esto no afecta. Ticket pendiente para sesión futura.
- **`--base-args-json` acepta JSON inline** (desde `4733160`, 2026-03-10): `run_ablation_matrix.py` detecta si el argumento comienza con `{` para parsear JSON inline vs ruta de archivo. Ejemplo: `--base-args-json '{"seq_len": 96, "preset": "cpu_safe"}'`.
- **Worktrees no tienen `data/`**: usar rutas absolutas al dataset en smoke tests lanzados desde un worktree (`/home/tmp/PROYECTOS PYTHON/FEDformer-Probabilistic-Time-Series-Forecasting/data/NVDA_features.csv`).
- **`ci.yml` push trigger**: añadir manualmente cada rama nueva a `on: push: branches` en `.github/workflows/ci.yml`, y los nuevos archivos de test a los shards correspondientes (`Run simulations and utils fast shard` / `Run feature branch regression slice`).
- **`scripts/__init__.py` es obligatorio**: sin él, `from scripts.run_ablation_matrix import ...` en tests lanza `ModuleNotFoundError`. Siempre crear `scripts/__init__.py` vacío al añadir módulos en `scripts/`.
- **`apply_preset()` se llama ANTES de overrides CLI**: en `_create_config`, el orden es `crear_config_base → apply_preset(config, args.preset) → aplicar_flags_CLI`. Esto garantiza prioridad: defaults < preset < CLI. No reordenar.
- **`monitor_metric` NO está en CLI**: solo acceso programático vía `config.sections.loop.monitor_metric`. Decisión de diseño intencional — no exponer como flag CLI de momento.
- **Bash heredoc `python3 - << 'PYEOF'` para scripts con backslashes**: usar este patrón al manipular archivos con backslashes de continuación de línea (ej. `ci.yml`). `sed` y `awk` tienen escaping problemático con `\` en heredocs.
- **Agentes paralelos en worktrees siempre tocan `ci.yml`**: cuando ≥2 agentes modifican el mismo bloque de `ci.yml`, el segundo merge genera conflicto. Resolver con `python3 - << 'PYEOF'` reemplazando el bloque conflictivo con ambas adiciones.
- **Worktrees de agentes pueden partir de base antigua**: si el agente se despacha inmediatamente después de un cherry-pick o merge complejo, el worktree puede arrancar desde un HEAD desactualizado. Síntoma: archivos existentes en `main` aparecen como "nuevos" en el worktree → conflicto `add/add` al cherry-pick. Solución: `git cherry-pick --abort`, copiar manualmente las funciones nuevas del worktree y hacer commit directo en `main`.
