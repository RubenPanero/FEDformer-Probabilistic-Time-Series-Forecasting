# FEDformer – Probabilistic Time Series Forecasting

## Navegación de código (jcodemunch / tree-sitter)

El repositorio está indexado con jcodemunch. **NUNCA usar `Read`/`Grep`/`Bash grep` para navegar código Python — usar siempre jcodemunch:**

| Tarea | Herramienta preferida |
|-------|-----------------------|
| Buscar una función/clase por nombre | `mcp__jcodemunch__search_symbols` |
| Ver métodos de un archivo | `mcp__jcodemunch__get_file_outline` |
| Leer implementación de un símbolo | `mcp__jcodemunch__get_symbol` |
| Buscar texto en todo el repo | `mcp__jcodemunch__search_text` |
| Vista general de directorios | `mcp__jcodemunch__get_repo_outline` |
| Leer archivo completo `.py` | `mcp__jcodemunch__get_file_content` |
| `Read` | Solo para archivos de configuración/markdown (CLAUDE.md, MEMORY.md, etc.) |

**Repo ID**: `local/FEDformer-Probabilistic-Time-Series-Forecasting-ed518b7c`
**Re-indexar si hay commits nuevos**: `/start` lo hace automáticamente (paso 0).

## Skills de desarrollo activas

Invocar estas skills **proactivamente** según la situación — sin esperar que el usuario las pida:

| Situación | Skill a invocar |
|-----------|----------------|
| Bug, test fallando, comportamiento inesperado | `systematic-debugging` — **antes** de proponer cualquier fix |
| Feature nueva o cambio arquitectónico complejo | `feature-dev` — planificación guiada 7 fases antes de tocar código |
| Tarea multi-paso con spec o requisitos claros | `writing-plans` — **antes** de tocar código |
| Implementar feature o bugfix (escribir código) | `test-driven-development` — **antes** de la implementación |
| Plan con tareas independientes paralelizables | `subagent-driven-development` — al ejecutar el plan |
| A punto de decir "listo", commitear o crear PR | `verification-before-completion` — **siempre** antes de terminar |
| El usuario pide commitear o hay cambios listos | `commit-commands:commit` — commits estandarizados con prefijo convencional |
| Al final de sesión con cambios en archivos/APIs | `claude-md-management:revise-claude-md` — actualiza CLAUDE.md con lo aprendido |

**Nota sobre `feature-dev`**: usar especialmente para Fase 2 del roadmap multi-ticker (MarketEncoder compartido, VIX/SPY conditioning cross-asset) y cualquier refactor estructural grande.



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
- **Salidas del entrenador**: `run_backtest()` retorna `ForecastOutput` (no tuplas). Usar `.preds_for_metrics`/`.gt_for_metrics` — siempre devuelven `*_real`. `window_fold_ids`: shape `(n_windows,)` int32, genera columna `fold` en CSV. Ver `memory/forecast_output.md` para API completa.
- **`return_transform` + `metric_space`**: dimensiones independientes. `return_transform` controla qué ve el modelo; `metric_space` controla qué contiene `preds_real`. Recomendado: `--return-transform log_return --metric-space returns`. `metric_space="prices"` reconstruye precios vía `_cumulative_returns_to_prices(last_prices, log_return)`.

## Code Quality

- Siempre ejecutar `ruff check --fix` y `ruff format` después de editar archivos Python para evitar conflictos de linting/formatting.
- El pre-commit completo es: `ruff check . --fix && ruff format . && pylint --errors-only models/ training/ data/ utils/ && pytest -q -m "not slow"`

## Git Workflow

- **Sin co-autoría de Claude**: no agregar trailers `Co-Authored-By` en commits.
- **PRs manuales**: para crear PRs, proporcionar la URL de GitHub generada por `git push` para creación manual — no usar `gh` CLI salvo que se pida explícitamente.
- Commits atómicos con prefijo convencional en inglés: `fix:`, `feat:`, `chore:`.
- Features en rama corta → PR pequeño → merge a `main`. Sin commits directos a `main`.

## Estándares de desarrollo Python

- Type hints 3.10+: `X | Y`, no `Optional[X]`. Docstrings en español (Google-style).
- Toda función pública nueva: al menos un test en `tests/` con fixtures de `conftest.py`.
- Mocks para llamadas externas — nunca red real en tests. Patrón yfinance: `patch("yfinance.download", return_value=ohlcv_df)` + `patch("data.vix_data.VixDataFetcher.get_vix_data", return_value=vix_df)`.
- `security.yml` ejecuta bandit; no usar `eval()`, `pickle` sin validación, ni `shell=True`.

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

- **Épicas 1–9 COMPLETADAS** (2026-03-09). Ver: `archivos auxiliares/plan_ejecucion_pipeline_probabilistico.md`
- **Estado actual** (2026-03-14): 293 tests fast · per-fold reseed fix activo (trainer.py) · seed 7 nuevo canónico NVDA (Sharpe +1.060).
- **Próximos pasos**:
  1. Push a origin/main (2 commits locales: `12f91df`, `b1b35ef`)
  2. PR: per-fold reseed fix + cp-walkforward-fold-fix → main
  3. Especialistas multi-ticker (MSFT, AAPL, AMZN, META, TSLA) — Kaggle P100
  4. Ampliar multi-seed NVDA (≥10 seeds) para distribución robusta
- **Plan de validación corto plazo**: `docs/plans/2026-03-09-validacion-corto-plazo.md`

## Entrenamiento headless

- `plt.show()` bloquea indefinidamente en subprocesos aunque `DISPLAY` esté definido (socket X11 inaccesible desde background).
- Siempre usar `MPLBACKEND=Agg python3 main.py ... --no-show` para ejecuciones no interactivas.
- Comando NVDA validado: `MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" --seq-len 96 --pred-len 20 --batch-size 64 --save-results --no-show`
- Sin `--epochs`: usa el default de `LoopSettings` (20 épocas). Pasar `--epochs N` solo para override explícito.
- Dataset NVDA permanente: `data/NVDA_features.csv` (1725 filas × 11 features, 2019–2026).
- **Segmentación canónica**: `seq_len=96, n_splits=4` → split_size=431. `split_min=342` con defaults; `n_splits = total_filas // split_min` para tickers heterogéneos.
- **Baseline canónico NVDA** (per-fold reseed fix, seed=7, 2026-03-14): Sharpe **+1.060** · Sortino **+1.940** · MaxDD −55.9% · Vol 2.40%. Referencia válida para comparaciones. `model_registry.json` actualizado.
- **Per-fold reseed fix** (`_run_single_fold`: `torch.manual_seed(seed + fold_idx)`): resuelve regresión seed=42 (+0.609→−1.127). Multi-seed post-fix (seeds 7,123,456,999, Kaggle 2×T4): mean=**+0.534**, std=0.361; **4/4 seeds Sharpe>0**. Seed 7 nuevo canónico.
- **Varianza inter-seed NF**: post-fix mean=+0.534, std=0.361 (vs pre-fix mean=+0.369, std=0.546; 2/4 positivos). **Causa raíz resuelta**: RNG global acumulado entre folds → per-fold reseed hace init independiente por fold.
- **Optuna NVDA** (14 trials, DBs persistidos en `optuna_studies/`): mejor config `{seq=96, pred=20, batch=32, clip=0.5}` → Sharpe +0.750, coverage_80=0.626. Zona Pareto coverage≥0.75 + Sharpe>0 **NO existe** en datos actuales (trade-off estructural). Candidato más cercano: trial #9 (Sharpe=+0.161, cov=0.739).
- `metric_space="returns"` con `return_transform="log_return"`: VaR/CVaR en unidades de retorno (5–7%). Con `metric_space="prices"` se reconstruyen precios vía `_cumulative_returns_to_prices(last_prices, log_return)`.

## Kaggle / notebooks

- **`data/` gitignoreado**: los CSVs (`NVDA_features.csv`, etc.) NO están en el repo — subirlos como dataset Kaggle separado y copiarlos en el notebook con `glob('/kaggle/input/**/*.csv', recursive=True)`.
- **Generar `.ipynb`**: usar `Bash` + `python3 << 'PYEOF' ... PYEOF` (hooks bloquean `Write`/`Read` sobre `.ipynb`).
- **`os.chdir(WORKDIR)` en Kaggle**: llamar UNA sola vez en celda 1; no repetir. Usar rutas absolutas `/kaggle/working/` para DBs y outputs.
- **GPU Kaggle**: activar en Settings → Accelerator → GPU T4 x1 + reiniciar kernel. Sin GPU: ~20 min/trial (vs ~3 min con T4).
- **Docstrings en celdas generadas por PYEOF**: NO usar `"""..."""` — se renderizan como `\"\"\"` en el source → `SyntaxError`. Usar `#` comentarios en su lugar.
- **Kaggle environment**: seleccionar "Latest environment" (no "Pin to original") — asegura PyTorch 2.0+, CUDA drivers actualizados y Python 3.10+.
- **2×T4 parallelismo**: `threading.Thread` + `env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)` para seeds en paralelo. Referencia: `archivos auxiliares/kaggle_multiseed_nvda_2gpu.ipynb`.

## Gotchas

- **`lr_lambda` usa `math.cos`/`math.pi`** (nunca `np`): `numpy.float64` rompe `torch.load(weights_only=True)` al deserializar checkpoints.
- **`functools.partial` para `worker_init_fn`**: NO usar — falla en Python 3.12 spawn. Patrón: clase callable módulo-level con `__slots__` (`_SeedWorker`).
- **`pred_len` debe ser par**: requisito del affine coupling split.
- **`run_manifest_*.json`**: métricas en `manifest["metrics"]`, NO en raíz. Usar `manifest.get("metrics", manifest)` para retrocompatibilidad.
- **`seed=42` intrínsecamente pobre con per-fold reseed**: fórmula `seed + fold_idx` asigna fold-seeds 43/44/45 → Sharpe −0.525 determinista. Usar **seed=7** (fold-seeds 8/9/10) como canónico NVDA.
- **`.ipynb` en CI**: nunca commitear a main — ruff/pylint los parsean como Python y rompen GitHub Actions. Guardar en `archivos auxiliares/` (gitignored).
- **Al añadir flags a `main.py`**: actualizar `argparse.Namespace` en `tests/test_finetune.py` con el nuevo atributo a `None`.
- **`cpu_safe` NO fuerza CPU**: deshabilita AMP/compile/workers pero modelo sigue en CUDA.
- **`tune_hyperparams.py`**: NO tiene `--targets` ni `--study-name`. `sharpe` NO existe en `user_attrs` — en modo sharpe, `best_trial.value` = Sharpe. Nombre estudio auto: `tune_{ticker_stem}`.
- **`--cp-walkforward` coverage 0.706** (seed=42): hallazgo de investigación, no bug — violación exchangeability CP bajo no-estacionariedad. Fold 0 excluido (sin datos previos).
- **`scripts/__init__.py` obligatorio**: sin él, imports de tests lanzan `ModuleNotFoundError`.
- **`FEDformerConfig`**: usa `enc_in`/`dec_in` (no `num_features`). Config anidado: `config.sections.preprocessing.return_transform`. Defaults `seq_len=10, pred_len=5` generan warnings — usar 96/20 en experimentos.
- **Worktrees desde base antigua**: síntoma `add/add` conflict en cherry-pick. Solución: `--abort`, portear con `git diff HEAD` manual. Usar rutas absolutas al dataset.
- **`apply_preset()` antes de CLI overrides**: orden `config_base → apply_preset → flags_CLI`. No reordenar.

→ Gotchas técnicos detallados: `memory/gotchas.md`
