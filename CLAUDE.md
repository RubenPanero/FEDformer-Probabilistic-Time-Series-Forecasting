# FEDformer – Probabilistic Time Series Forecasting

Implementación de **FEDformer** (Frequency Enhanced Decomposed Transformer) con **Normalizing Flows** para predicción probabilística de series temporales, orientada a datos financieros.

## Project Overview

Este proyecto usa Python con `ruff` para linting (respetar límites de longitud de línea: 88 chars), `pytest` para testing (283+ tests), y YAML/Markdown para CI y documentación. Proyecto principal: FEDformer forecasting.

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
tune_hyperparams.py            # CLI Optuna: --csv, --n-trials, --n-splits, --storage-path,
                               #   --study-objective {sharpe,composite,multi-objective}
                               #   --composite-score-profile {balanced}, --best-save-canonical
                               #   (NO tiene --targets ni --study-name)
scripts/
  __init__.py                  # módulo vacío — necesario para importar scripts en tests
  run_ablation_matrix.py       # AblationJob, build_ablation_jobs, job_to_argv, run_ablation_job
  run_multi_seed.py            # run_single_seed, run_multi_seed_experiment — CLI: --seeds, --dry-run
tests/                         # pytest (conftest.py, 23 archivos de test — 283 fast + 7 @slow)
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
# NOTA: tune_hyperparams.py NO tiene --targets ni --study-name (no existen)
python3 tune_hyperparams.py --csv data/NVDA_features.csv --n-trials 30

# Optuna con score compuesto probabilístico
python3 tune_hyperparams.py --csv data/NVDA_features.csv \
  --n-trials 30 --study-objective composite

# Con persistencia SQLite (permite reanudar)
python3 tune_hyperparams.py --csv data/NVDA_features.csv \
  --n-trials 30 --study-objective composite \
  --storage-path optuna_studies/nvda_composite.db

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

## Flujo de trabajo

**Definition of Done:** tests pasan · linting verde · sin regresiones · CLAUDE.md actualizado si hay nuevos archivos/APIs.
- TDD: test primero, luego implementar.
- Commits atómicos con prefijo convencional en inglés.
- Features en rama corta → PR → merge a `main`. Sin commits directos a `main`.

## MLOps

- **Reproducibilidad**: `config.seed` propagado a PyTorch, NumPy y DataLoader workers. `FEDformerConfig` serializable junto al checkpoint.
- **Checkpoints**: `checkpoints/best_model_fold_N.pt` (`model_state_dict` + `config`). `weights_only=True` en `torch.load()`. Fine-tuning siempre desde checkpoint explícito (`--finetune-from`).
- **Anti-leakage**: walk-forward obligatorio; scaler ajustado solo en train de cada fold.
- **`--save-results`** exporta 7 artefactos: `predictions_*.csv`, `risk_metrics_*.csv`, `portfolio_metrics_*.csv`, `training_history_*.csv`, `run_manifest_*.json`, `probabilistic_metrics_*.csv`, `fold_metrics_*.csv`.
- **Experiment tracking**: W&B opcional (falla silenciosamente si no instalado). Tracking local completo vía `results/` + `utils/experiment_registry.py` (`load_run_manifests`, `rank_runs`, `aggregate_seed_metrics`).

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

- Ejecutar `pytest -q -m "not slow"` tras cualquier cambio en Python antes de commit.
- `pyproject.toml`: `[tool.pytest.ini_options]` — `-q --ignore=reports`, markers: `slow`
- Fixtures globales en `tests/conftest.py`: `config`, `model_factory`, `synthetic_batch`
- Tests no obvios: `test_architecture_and_leakage.py` (data leakage walk-forward), `test_flows.py` (invertibilidad NF), `test_trainer_integration.py` (@slow, run_backtest e2e), `test_calibration.py` (CP conformal), `test_reproducibility.py` (determinismo seed)
- Al añadir flags a `main.py`: actualizar `argparse.Namespace` en `tests/test_finetune.py` con el nuevo atributo a `None` para evitar `AttributeError` en `_create_config()`.

## CI/CD (`.github/workflows/`)

- `ci.yml`: Ubuntu + Windows, Python 3.10–3.11 → smoke test + pytest sin `@slow`
- `pylint.yml`: errores E/F únicamente · `ruff.yml`, `security.yml`, `compatibility.yml`
- Al añadir rama nueva: actualizar `on: push: branches` en `ci.yml`. Al añadir test: añadirlo al shard correspondiente.

## Hoja de ruta activa

- **Épicas 1–9 COMPLETADAS** (2026-03-09) — commits `9810be1`→`623c5b1` en `main`.
- Plan histórico: `archivos auxiliares/plan_ejecucion_pipeline_probabilistico.md`
- Backlog fuente: `archivos auxiliares/backlog_pipeline_investigacion_probabilistica.md`
- **Sesión 2026-03-11** (commits `e91b66e`, `49592af`):
  1. ✅ Patch `_make_loader`: workers de rehearsal/fine-tuning ahora tienen semilla determinista (`e91b66e`).
  2. ✅ Análisis Pareto Optuna: zona coverage≥0.75 + Sharpe>0 **NO existe** en 14 trials. Trade-off estructural confirmado. Punto más cercano: trial #9 (Sharpe=+0.161, coverage=0.739).
  3. ✅ Diagnóstico épocas: early stopping dispara en épocas 10–16/20. Aumentar epochs **no ayudaría** con la varianza. La causa es sensibilidad de inicialización NF.
  4. ✅ `--conformal-calibration` flag implementado (CP Enfoque 2 prototype, `49592af`): post-hoc sobre predicciones agregadas, reporta `cp_coverage_80` y `cp_q_hat` junto a métricas NF.
- **Próximos pasos inmediatos** (Fase 1 — sesión futura):
  1. Investigar fold 0 ausente en `training_history_*.csv` (MetricsTracker indexing)
  2. Implementar CP Enfoque 1 (walk-forward fold-aware) sobre el prototype existente
  3. Verificar cp_coverage_80≥0.80 en run real con `--conformal-calibration`
  4. Multi-seed NVDA limpio (seeds 42,123,7,456) con `--conformal-calibration`
  5. Especialistas multi-ticker (MSFT, AAPL, AMZN, META, TSLA) — siempre NVDA primero
- **Plan de validación corto plazo**: `docs/plans/2026-03-09-validacion-corto-plazo.md`

## Entrenamiento headless

- `plt.show()` bloquea indefinidamente en subprocesos aunque `DISPLAY` esté definido (socket X11 inaccesible desde background).
- Siempre usar `MPLBACKEND=Agg python3 main.py ... --no-show` para ejecuciones no interactivas.
- Comando NVDA validado: `MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" --seq-len 96 --pred-len 20 --batch-size 64 --save-results --no-show`
- Sin `--epochs`: usa el default de `LoopSettings` (20 épocas). Pasar `--epochs N` solo para override explícito.
- Dataset NVDA permanente: `data/NVDA_features.csv` (1725 filas × 11 features, 2019–2026).
- **Segmentación canónica**: `seq_len=96, n_splits=4` → split_size=431. `split_min=342` con defaults; `n_splits = total_filas // split_min` para tickers heterogéneos.
- **Baseline canónico NVDA** (post-fix `00f119c`, seed=42, 2026-03-10): Sharpe **+0.609** · Sortino **+0.993** · VaR 5.35% · CVaR 7.07% · MaxDD −72.69% · Vol 2.47%. Referencia válida para todas las comparaciones. `model_registry.json` actualizado.
- **Alta varianza inter-seed NF**: multi-seed (42,123,7,456) → mean=+0.369, std=0.546; 2/4 seeds con Sharpe>0.5. **Causa: sensibilidad inicialización NF, no falta de épocas** (early stopping dispara en 10–16/20).
- **Optuna NVDA** (14 trials, DBs persistidos en `optuna_studies/`): mejor config `{seq=96, pred=20, batch=32, clip=0.5}` → Sharpe +0.750, coverage_80=0.626. Zona Pareto coverage≥0.75 + Sharpe>0 **NO existe** en datos actuales (trade-off estructural). Candidato más cercano: trial #9 (Sharpe=+0.161, cov=0.739).
- `metric_space="returns"` con `return_transform="log_return"`: VaR/CVaR en unidades de retorno (5–7%). Con `metric_space="prices"` se reconstruyen precios vía `_cumulative_returns_to_prices(last_prices, log_return)`.

## Gotchas

- `FEDformerConfig` usa `enc_in`/`dec_in` para dimensiones (no `num_features`). Defaults: `seq_len=10, pred_len=5` (genera warnings); usar `seq_len=96, pred_len=24` para experimentos reales. Config anidado: `config.sections.preprocessing.return_transform`.
- `pytest -q` no muestra línea de resumen — usar `pytest -v`. `build_financial_dataset()` es función, no clase.
- `label_len` default = **5** (no 63). `len(TimeSeriesDataset)` = ventanas, no filas.
- `torch.fft.rfft` no soporta bfloat16 (AMP Ada Lovelace) → `_apply_rfft` castea a float32, restaura `orig_dtype` tras irfft. `torch.compile(mode="max-autotune")` genera NaN en GPUs <40 SMs → trainer degrada a eager.
- `torch.quantile()` se mueve a CPU en `_evaluate_model` (determinismo CUDA). Forward/backward GPU intacto.
- **`lr_lambda` usa `math.cos`/`math.pi`** (nunca `np`): `np.cos()` → `numpy.float64` → `torch.load(weights_only=True)` falla al deserializar checkpoints.
- **Retrocompatibilidad checkpoints numpy**: `torch.serialization.safe_globals([numpy._core.multiarray.scalar, np.float64, ...])` para cargar checkpoints antiguos. `numpy._core` es privado → `# pylint: disable=no-member`.
- **`_optimizer_step` es `@staticmethod`**: pasar `clip_norm` como parámetro explícito; no usar `self.config.*` dentro.
- **`drop_last=False` en train_loader**: ya aplicado; evita 0 batches en folds pequeños.
- **Tests `model_factory` + `_prepare_batch`**: usar `model_factory().to(get_device())` — tensores van a CUDA, modelo se crea en CPU por defecto.
- **`_set_split_view`**: solo para tests standalone. Walk-forward usa `flag="all"` + `Subset`.
- **`sequential_finetuner.py` no tiene `--save-results`**: usar `main.py --finetune-from ... --finetune-lr 1e-5`.
- **`apply_preset()` antes de CLI overrides**: orden `config_base → apply_preset → flags_CLI`. No reordenar.
- **`monitor_metric` no en CLI**: solo programático vía `config.sections.loop.monitor_metric`.
- **`scripts/__init__.py` obligatorio**: sin él, imports de tests lanzan `ModuleNotFoundError`.
- **`--base-args-json` acepta JSON inline**: `run_ablation_matrix.py` detecta `{` para distinguir JSON vs ruta.
- **`gh` CLI no instalado**: crear PRs manualmente desde la URL que imprime `git push`.
- **Subagentes para entrenamientos**: requieren permiso Bash explícito. Usar `Bash(run_in_background=true)`.
- **`cpu_safe` NO fuerza CPU**: deshabilita AMP/compile/workers pero modelo sigue en CUDA.
- **`last_prices` boundary**: `df.iloc[cutoff]` no es leakage — implícito en último retorno de entrenamiento.
- **`df.index.tz_localize(None)`**: lanza `TypeError` con yfinance moderno. Usar `if df.index.tz is not None: df.index = df.index.tz_localize(None)`.
- **`logging.basicConfig()` en `vix_data.py`/`alpha_vantage_client.py`**: anti-patrón pre-existente; corregir a `logger = logging.getLogger(__name__)` si se modifican.
- **Worktrees sin `data/`**: usar rutas absolutas al dataset.
- **Worktrees desde base antigua**: síntoma `add/add` conflict en cherry-pick. Solución: `--abort`, copiar manualmente, commit directo en `main`.
- **Bash heredoc `python3 - << 'PYEOF'`**: para scripts con backslashes (`ci.yml`). `sed`/`awk` tienen escaping problemático.
- **`tune_hyperparams.py`**: `sharpe` NO existe en `user_attrs` — en modo sharpe, `best_trial.value` = Sharpe. `composite_score = 0.5·sharpe + 0.3·(1-pinball_norm) + 0.2·coverage_score`. Nombre estudio auto: `tune_{ticker_stem}`. Kaggle: `kaggle_optuna_nvda.ipynb`; T4 tiene 40 SMs → compile puede activarse, añadir `cpu_safe` si NaN.
- **`--conformal-calibration`** (`49592af`): CP Enfoque 2 post-hoc; reporta `cp_coverage_80` y `cp_q_hat` junto a métricas NF. Default: off. CP Enfoque 1 (walk-forward) pendiente Fase 1.
- **fold 0 ausente en training_history**: n_splits=4 → solo folds 1,2,3 en CSV. No afecta entrenamiento. Pendiente investigar en Fase 1 (MetricsTracker indexing).
