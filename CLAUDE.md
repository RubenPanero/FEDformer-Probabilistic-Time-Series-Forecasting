# FEDformer – Probabilistic Time Series Forecasting

FEDformer (Frequency Enhanced Decomposed Transformer) + Normalizing Flows para predicción probabilística de series temporales financieras.

## Stack

Python 3.10+ · PyTorch 2.0+ · pandas · scikit-learn · Optuna · W&B (opcional)
Linting: `ruff` (88 chars) · Tests: `pytest` (299 fast + 7 @slow)
Entorno: `.venv/` · Linux Mint 22.3 → **usar `python3`** (nunca `python`)

## Navegación de código

Repo indexado con jcodemunch. **NUNCA usar `Read`/`Grep` para código Python — usar jcodemunch:**

- `search_symbols` → función/clase por nombre
- `get_file_outline` → métodos de un archivo
- `get_symbol` → implementación de un símbolo
- `search_text` → buscar texto en todo el repo
- `get_file_content` → archivo `.py` completo
- `Read` → solo para config/markdown (CLAUDE.md, MEMORY.md, .yml)

**Repo ID**: `local/FEDformer-Probabilistic-Time-Series-Forecasting-ed518b7c`
Context7 MCP: usar proactivamente para docs externas (PyTorch, Optuna, yfinance, etc.).

## Arquitectura

```
CSV → TimeSeriesDataset (scale + regimes)
    → WalkForwardTrainer (n folds)
        → Flow_FEDformer: x_enc, x_dec, x_regime → Distribution
        → Loss: -log_prob(y)
    → ForecastOutput (preds, ground_truth, samples)
    → RiskSimulator + PortfolioSimulator → métricas
```

Directorios gitignoreados: `results/`, `checkpoints/`, `optuna_studies/`, `archive/`
Scripts de experimentos: `archivos auxiliares/` — scripts bash para runs multi-seed, grids y limpieza

## Comandos esenciales

```bash
# Run canónico NVDA (seed=7)
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" \
  --seq-len 96 --pred-len 20 --batch-size 64 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.5 --seed 7 --save-results --save-canonical --no-show
# --save-canonical guarda el checkpoint en checkpoints/*_canonical.pt y actualiza model_registry.json

# Tests rápidos
pytest -q -m "not slow"

# Pre-commit (paridad CI) — obligatorio antes de cada commit
ruff check . --fix && ruff format . && pylint --errors-only models/ training/ data/ utils/ && pytest -q -m "not slow"

# Optuna
python3 tune_hyperparams.py --csv data/NVDA_features.csv \
  --n-trials 30 --study-objective sharpe \
  --storage-path optuna_studies/nvda_sharpe.db \
  --enqueue-canonical --clean-results

# Multi-seed
python3 scripts/run_multi_seed.py --csv data/NVDA_features.csv \
  --seeds 7 42 123 --extra-args --seq-len 96 --pred-len 20 --batch-size 64
```

Siempre `MPLBACKEND=Agg` + `--no-show` en ejecuciones headless (plt.show() bloquea).

## Convenciones críticas

- **Commits**: sin co-autoría de Claude. Prefijo convencional en inglés (`fix:`, `feat:`, `chore:`).
- **PRs**: el usuario los crea manualmente — proporcionar URL de `git push`.
- **Comentarios en código**: en español. Docstrings Google-style.
- **Config**: todo en `FEDformerConfig` (dataclass). Orden: `config_base → apply_preset() → flags_CLI`.
- **`pred_len` debe ser par** (requisito affine coupling split).
- **Salidas del modelo**: objetos `Distribution` (`.mean`, `.log_prob`, `.sample()`), nunca escalares.
- **ForecastOutput**: usar `.preds_for_metrics`/`.gt_for_metrics` (siempre `*_real`).
- **Anti-leakage**: walk-forward obligatorio; scaler ajustado solo en train de cada fold.
- **Seed canónico**: 7. seed=42 descartado (Sharpe −0.525 con per-fold reseed).
- **Per-fold reseed**: `_run_single_fold` usa `torch.manual_seed(seed + fold_idx)`.
- Al añadir flags a `main.py`: actualizar `argparse.Namespace` en `tests/test_finetune.py`.
- **`main.py` flags de optimizador**: `--learning-rate` (float, default: config 1e-4), `--weight-decay` (default: config 1e-5). Early stopping en época 6-7/20 es convergencia normal con lr=1e-4, no indica lr incorrecto.
- Type hints 3.10+: `X | Y`, no `Optional[X]`. Tests con fixtures de `conftest.py`.
- Mocks para llamadas externas — nunca red real en tests.
- Ramas huérfanas `worktree-agent-*` y `claude/*`: limpiar periódicamente.

## CI/CD

`ci.yml` (Ubuntu+Windows, Py 3.10–3.11), `pylint.yml` (E/F only), `ruff.yml`, `security.yml`, `compatibility.yml`
Actions: `checkout@v6` · `setup-python@v6`. Al añadir rama: actualizar `on: push: branches` en `ci.yml`.

## Modelos canónicos (seed=7)

| Ticker | Sharpe | Sortino | MaxDD  | Estado |
|--------|--------|---------|--------|--------|
| NVDA   | +1.060 | +1.940  | −55.9% | canónico |
| GOOGL  | +1.196 | +2.306  | −27.7% | canónico |

**Multi-ticker optimization cerrado sesión 14** (2026-03-19). MSFT/AAPL/META/AMZN/TSLA no superaron Sharpe > 0.50 de forma reproducible tras Optuna + Grid2D + multi-seed. Checkpoints archivados en `archive/`. Resultados históricos → `MEMORY.md`.
Foco actual: inference API + visualización probabilística sobre NVDA+GOOGL.

## Gotchas

- **`lr_lambda` usa `math.cos`/`math.pi`** (nunca `np`): `numpy.float64` rompe `torch.load(weights_only=True)`.
- **`functools.partial` para `worker_init_fn`**: NO usar — falla Python 3.12 spawn. Usar `_SeedWorker` (clase callable módulo-level).
- **`FEDformerConfig`**: usa `enc_in`/`dec_in` (no `num_features`). Defaults `seq_len=10, pred_len=5` — usar 96/20 en experimentos.
- **`tune_hyperparams.py`**: NO tiene `--targets` ni `--study-name`. Pasa `--seed 7` y **siempre** `--compile-mode ""` al subprocess (obligatorio — config.py default es `"max-autotune"`, omitir el flag activa compilación en todos los trials). Nuevos flags: `--n-startup-trials`, `--sampler-seed`, `--clean-results`, `--enqueue-canonical/--no-enqueue-canonical`. Trials fallidos retornan `-1.0` — **no contaminan TPE**. **Espacio actual**: 76 combinaciones válidas. Con ≤12 trials, TPE = random search (9% cobertura).
- **`torch.compile` guard** (`trainer.py`): degrada `max-autotune→""` en GPUs <40 SMs. RTX 4050 (20 SMs) → sin compile. T4 (40 SMs) → NO degrada → timeouts.
- **`run_manifest_*.json`**: métricas en `manifest["metrics"]`, NO en raíz.
- **`scripts/__init__.py` obligatorio**: sin él, tests lanzan `ModuleNotFoundError`.
- **Scripts bash**: NUNCA usar `Write` tool → produce CRLF. Usar `Bash` + `cat > file << 'EOF'`.
- **Bash paths con espacios**: `ls "path con espacio"/glob*` falla silenciosamente. Usar `python3 -c "glob.glob(...)"` para listar/leer archivos con paths con espacios. En scripts, usar `PROJDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"` en vez de hardcodear el path.
- **Bash scripts — `set -u`**: obligatorio para detectar variables vacías (e.g. PROJDIR="" → `mkdir /logs` en vez de error claro).
- **`.ipynb` en CI**: nunca commitear a `main` — ruff los parsea y rompe Actions.
- **Kaggle notebooks**: generar con `Bash` + `python3 << 'PYEOF'` (hooks bloquean Write/Read sobre .ipynb). Usar `#` comentarios, nunca `"""..."""` en celdas PYEOF.
- **Timing RTX 4050**: ~10 min/run canónico, ~7-8 min/trial Optuna.

→ Lista completa con contexto: `memory/gotchas.md`

## Skills proactivas

| Situación | Skill |
|-----------|-------|
| Bug / test fallando | `systematic-debugging` |
| Feature nueva / refactor grande | `feature-dev` |
| Tarea multi-paso con spec | `writing-plans` |
| Escribir código | `test-driven-development` |
| Tareas paralelas independientes | `subagent-driven-development` |
| Antes de decir "listo" / commit / PR | `verification-before-completion` |
| Commit | `commit-commands:commit` |
| Fin de sesión | `claude-md-management:revise-claude-md` |
