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

## Context7 MCP — documentación técnica en tiempo real

Usar Context7 **proactivamente** siempre que la tarea dependa de documentación externa autoritativa, actualizada o específica de versión que no esté en el workspace. No esperar a que el usuario lo pida explícitamente.

```
resolve-library-id → query-docs → responder con ejemplos y versión citada
```

Casos típicos: PyTorch, pandas, scikit-learn, GitHub Actions, Optuna, W&B, yfinance, cualquier API o SDK externo.

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

`feature-dev`: usar especialmente para Fase 2 del roadmap multi-ticker (MarketEncoder compartido, VIX/SPY conditioning cross-asset) y cualquier refactor estructural grande.

## Project Overview

Implementación de **FEDformer** (Frequency Enhanced Decomposed Transformer) con **Normalizing Flows** para predicción probabilística de series temporales financieras.

- **Stack**: Python 3.10+ · PyTorch 2.0+ · pandas · scikit-learn · W&B · Optuna
- **Linting**: `ruff` (88 chars) · **Tests**: `pytest` (293 fast + 7 @slow)
- **Virtual env**: `.venv/` (`source .venv/bin/activate`) · **OS**: Linux Mint 22.3 → usar `python3`

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
  NVDA_features.csv            # Dataset permanente (1725 filas × 11 features, 2019–2026)
  GOOGL/MSFT/AAPL/AMZN/META/TSLA_features.csv  # Tickers adicionales (mismo formato)
training/
  trainer.py                   # WalkForwardTrainer → devuelve ForecastOutput
  forecast_output.py           # ForecastOutput: dataclass dual-space (scaled + real)
  train_base_model.py          # Entrenamiento base con GOOGL
  sequential_finetuner.py      # Fine-tuning secuencial multi-ticker
                               # Rehearsal: training/rehearsal_buffer.py + --rehearsal-k/epochs/lr-mult
  utils.py                     # mc_dropout_inference, utilidades de entrenamiento
simulations/
  risk.py                      # RiskSimulator: VaR, CVaR, Expected Shortfall
  portfolio.py                 # PortfolioSimulator: Sharpe, drawdown, Sortino
utils/
  helpers.py                   # get_device, setup_cuda_optimizations, AMP dtype selection
  metrics.py                   # MetricsTracker (seguimiento in-memory por época)
  calibration.py               # conformal_quantile (Conformal Prediction)
  probabilistic_metrics.py     # pinball_loss, crps_from_samples, empirical_coverage
  io_experiment.py             # serialize_config, build_run_manifest, save_probabilistic_metrics
  experiment_registry.py       # load_run_manifests, rank_runs, aggregate_seed_metrics
tune_hyperparams.py            # CLI Optuna: --csv, --n-trials, --n-splits, --storage-path,
                               #   --study-objective {sharpe,composite,multi-objective}
                               #   --composite-score-profile {balanced}, --best-save-canonical
                               #   (NO tiene --targets ni --study-name)
scripts/
  __init__.py                  # módulo vacío — necesario para importar scripts en tests
  run_ablation_matrix.py       # AblationJob, build_ablation_jobs, job_to_argv, run_ablation_job
  run_multi_seed.py            # run_single_seed, run_multi_seed_experiment — CLI: --seeds, --dry-run
  verify_cp_walkforward.py     # Verificación CP walk-forward coverage
tests/                         # pytest (conftest.py, 26 archivos — 293 fast + 7 @slow)
results/                       # CSVs exportados por --save-results (gitignored)
checkpoints/                   # Modelos entrenados .pt (gitignored) + model_registry.json
optuna_studies/                # DBs SQLite de Optuna (gitignored — no versionar binarios)
archive/                       # Artefactos históricos: fix-reports, scripts temporales (gitignored)
docs/
  plans/
    2026-03-09-validacion-corto-plazo.md  # Plan ablaciones + multi-seed + Optuna composite
variants.json                  # Variantes de ablación: dropout {0.05,0.1,0.2} × {log_return,none}
```

## Comandos

```bash
# Comando canónico NVDA (seed=7, configuración validada)
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" \
  --seq-len 96 --pred-len 20 --batch-size 64 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.5 --seed 7 \
  --save-results --no-show

# Multi-ticker NVDA+GOOGL
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv data/GOOGL_features.csv \
  --targets "Close" --seq-len 96 --pred-len 20 --batch-size 64 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.5 --save-results --no-show

# Tests rápidos (sin @slow)
pytest -q -m "not slow"

# Verificación pre-commit (paridad con CI) — obligatorio antes de cada commit
ruff check . --fix && ruff format . && pylint --errors-only models/ training/ data/ utils/ && pytest -q -m "not slow"

# Optuna — score compuesto probabilístico con persistencia SQLite
python3 tune_hyperparams.py --csv data/NVDA_features.csv \
  --n-trials 30 --study-objective composite \
  --storage-path optuna_studies/nvda_composite.db

# Ablación reproducible
python3 scripts/run_ablation_matrix.py --csv data/NVDA_features.csv \
  --targets Close --variants-json variants.json --dry-run

# Experimento multi-seed
python3 scripts/run_multi_seed.py --csv data/NVDA_features.csv \
  --seeds 7 42 123 --extra-args --seq-len 96 --pred-len 20 --batch-size 64

# Dataset financiero
python3 data/financial_dataset_builder.py --symbol GOOGL --use_mock
```

## Convenciones

- **Commits**: sin co-autoría de Claude. Mensajes en inglés (`fix:`, `feat:`, `chore:`).
- **Comentarios en código**: en español.
- **Configuración**: todo en `FEDformerConfig` (dataclass); no hardcodear parámetros.
  - Para subsecciones tipadas: `config.sections.model.transformer.dropout = 0.2`
- **Preprocesamiento**: siempre leakage-safe (scaler se ajusta solo en datos de entrenamiento).
- **Salidas del modelo**: objetos `Distribution` (`.mean`, `.log_prob`, `.sample()`), nunca escalares.
- **`pred_len` debe ser par** (requisito del affine coupling split).
- **Early stopping**: monitorea `val_loss`. `val_fraction=0.15` en `LoopSettings`. Defaults: `patience=5`, `min_delta=5e-3`, `gradient_accumulation_steps=2`. Checkpoint de mejor `val_loss` siempre se recarga antes de inferencia.
- **Salidas del entrenador**: `run_backtest()` retorna `ForecastOutput`. Usar `.preds_for_metrics`/`.gt_for_metrics` (siempre `*_real`). Ver `memory/forecast_output.md` para API completa.
- **`return_transform` + `metric_space`**: independientes. `return_transform` controla qué ve el modelo; `metric_space` controla `preds_real`. Canónico: `--return-transform log_return --metric-space returns`.

## Git Workflow

- **Sin co-autoría de Claude**: no agregar trailers `Co-Authored-By` en commits.
- **PRs**: proporcionar URL de GitHub generada por `git push` para creación manual.
- Commits atómicos con prefijo convencional en inglés: `fix:`, `feat:`, `chore:`.
- Features en rama corta → PR pequeño → merge a `main`. Sin commits directos a `main`.
- **Ramas huérfanas**: limpiar `worktree-agent-*` y `claude/*` periódicamente con `git branch -D`.

## Estándares de desarrollo Python

- Type hints 3.10+: `X | Y`, no `Optional[X]`. Docstrings en español (Google-style).
- Toda función pública nueva: al menos un test en `tests/` con fixtures de `conftest.py`.
- Mocks para llamadas externas — nunca red real en tests. Patrón yfinance: `patch("yfinance.download", return_value=ohlcv_df)` + `patch("data.vix_data.VixDataFetcher.get_vix_data", return_value=vix_df)`.
- `security.yml` ejecuta bandit; no usar `eval()`, `pickle` sin validación, ni `shell=True`.
- Al añadir flags a `main.py`: actualizar `argparse.Namespace` en `tests/test_finetune.py` con el nuevo atributo a `None`.

## MLOps

- **Reproducibilidad**: `config.seed` propagado a PyTorch, NumPy y DataLoader workers. `FEDformerConfig` serializable junto al checkpoint.
- **Checkpoints**: `checkpoints/best_model_fold_N.pt` (`model_state_dict` + `config`). `weights_only=True` en `torch.load()`.
- **Anti-leakage**: walk-forward obligatorio; scaler ajustado solo en train de cada fold.
- **`--save-results`** exporta 7 artefactos: `predictions_*.csv`, `risk_metrics_*.csv`, `portfolio_metrics_*.csv`, `training_history_*.csv`, `run_manifest_*.json`, `probabilistic_metrics_*.csv`, `fold_metrics_*.csv`.
- **Experiment tracking**: W&B opcional. Tracking local vía `results/` + `utils/experiment_registry.py`.

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
- Fixtures globales en `tests/conftest.py`: `config`, `model_factory`, `synthetic_batch`
- Tests no obvios: `test_architecture_and_leakage.py` (data leakage), `test_flows.py` (invertibilidad NF), `test_trainer_integration.py` (@slow, e2e), `test_calibration.py` (CP conformal), `test_reproducibility.py` (determinismo seed).

## CI/CD (`.github/workflows/`)

- `ci.yml`: Ubuntu + Windows, Python 3.10–3.11 → smoke test + pytest sin `@slow`
- `pylint.yml`: errores E/F únicamente · `ruff.yml`, `security.yml`, `compatibility.yml`
- **Actions (Node.js 24)**: `actions/checkout@v6` · `actions/setup-python@v6` — actualizado 2026-03-15.
- Al añadir rama nueva: actualizar `on: push: branches` en `ci.yml`.

## Hoja de ruta activa

- **Épicas 1–9 COMPLETADAS** (2026-03-09).
- **Estado actual** (2026-03-15): 293 tests fast · CI verde (6/6 workflows) · per-fold reseed fix activo · seed 7 canónico NVDA (Sharpe +1.060) · repo limpio (archive/).
- **Próximos pasos**:
  1. PR: `feat/cp-walkforward-fold-fix` → main
  2. Especialistas multi-ticker (MSFT, AAPL, AMZN, META, TSLA) — Kaggle P100 con seed=7
  3. Ampliar multi-seed NVDA (≥10 seeds) para distribución robusta
- **Plan de validación**: `docs/plans/2026-03-09-validacion-corto-plazo.md`

## Entrenamiento headless

- Siempre usar `MPLBACKEND=Agg python3 main.py ... --no-show` para ejecuciones no interactivas (`plt.show()` bloquea sin display).
- Sin `--epochs`: usa el default de `LoopSettings` (20 épocas).
- **Segmentación canónica**: `seq_len=96, n_splits=4` → split_size=431.
- **Baseline canónico NVDA** (seed=7, per-fold reseed fix, 2026-03-14): Sharpe **+1.060** · Sortino **+1.940** · MaxDD −55.9% · Vol 2.40%. Referencia en `checkpoints/model_registry.json`.
- **Per-fold reseed fix** (`_run_single_fold`: `torch.manual_seed(seed + fold_idx)`): resuelve regresión RNG acumulado entre folds. Multi-seed post-fix (4 seeds): mean=+0.534, std=0.361, 4/4 Sharpe>0.
- **Optuna NVDA** (14 trials): mejor `{seq=96, pred=20, batch=32, clip=0.5}` → Sharpe +0.750. Zona Pareto coverage≥0.75 + Sharpe>0 **no existe** (trade-off estructural). DBs en `optuna_studies/` (gitignored).

## Kaggle / notebooks

- **`data/` gitignoreado**: los CSVs NO están en el repo — subir como dataset Kaggle + copiar con `glob('/kaggle/input/**/*.csv', recursive=True)`.
- **Generar `.ipynb`**: usar `Bash` + `python3 << 'PYEOF' ... PYEOF` (hooks bloquean `Write`/`Read` sobre `.ipynb`).
- **`os.chdir(WORKDIR)`**: llamar UNA sola vez en celda 1. Rutas absolutas `/kaggle/working/` para DBs y outputs.
- **GPU Kaggle**: Settings → Accelerator → GPU T4 x1 + reiniciar kernel. Sin GPU: ~20 min/trial.
- **Docstrings en celdas PYEOF**: usar `#` comentarios, nunca `"""..."""` (→ `SyntaxError`).
- **Kaggle environment**: "Latest environment" (PyTorch 2.0+, Python 3.10+).
- **2×T4**: `threading.Thread` + `env["CUDA_VISIBLE_DEVICES"]` por grupo. Ver `archive/` para notebooks de referencia.
- **`.ipynb` en CI**: nunca commitear a `main` — ruff los parsea como Python y rompe Actions.

## Gotchas

- **`lr_lambda` usa `math.cos`/`math.pi`** (nunca `np`): `numpy.float64` rompe `torch.load(weights_only=True)`.
- **`functools.partial` para `worker_init_fn`**: NO usar — falla Python 3.12 spawn. Usar clase callable módulo-level con `__slots__` (`_SeedWorker`).
- **`pred_len` debe ser par**: requisito del affine coupling split.
- **`run_manifest_*.json`**: métricas en `manifest["metrics"]`, NO en raíz. Usar `manifest.get("metrics", manifest)` para retrocompatibilidad.
- **`seed=42` intrínsecamente pobre con per-fold reseed**: fold-seeds 43/44/45 → Sharpe −0.525 determinista. Usar **seed=7** como canónico NVDA.
- **`scripts/__init__.py` obligatorio**: sin él, imports de tests lanzan `ModuleNotFoundError`.
- **`FEDformerConfig`**: usa `enc_in`/`dec_in` (no `num_features`). Defaults `seq_len=10, pred_len=5` generan warnings — usar 96/20 en experimentos.
- **`tune_hyperparams.py`**: NO tiene `--targets` ni `--study-name`. En modo sharpe, `best_trial.value` = Sharpe. Nombre estudio auto: `tune_{ticker_stem}`.
- **`optuna_studies/`**: gitignoreado — las DBs SQLite no se versionan. Guardar en local o subir a Kaggle.
- **`apply_preset()` antes de CLI overrides**: orden `config_base → apply_preset → flags_CLI`. No reordenar.
- **`cpu_safe` NO fuerza CPU**: deshabilita AMP/compile/workers pero modelo sigue en CUDA.
- **`--cp-walkforward` coverage 0.706** (seed=42): hallazgo de investigación — violación exchangeability CP bajo no-estacionariedad. Fold 0 excluido (sin datos previos).

→ Gotchas técnicos detallados: `memory/gotchas.md`
