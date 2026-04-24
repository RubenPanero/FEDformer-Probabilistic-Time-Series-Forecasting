# Repository Technical Status

This document is the tracked technical reference for the current optimization
state of the repository.

## Validation Surface

Use explicit commands. Do not assume `make ci-check`; this repository does not
ship a tracked `Makefile`.

Core local validation:

```bash
python -m pytest -q --collect-only
python -m pytest -q -m "not slow"
python -m pytest -q -m benchmark tests/test_critical_bottlenecks_benchmarks.py
ruff check .
ruff format --check .
python main.py --help
python tune_hyperparams.py --help
python -m inference --help
```

Windows hook entrypoints:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\hooks\pre-commit.ps1
powershell -ExecutionPolicy Bypass -File scripts\hooks\pre-push.ps1
```

Fast regression coverage is aligned with:

- `scripts/hooks/pre-push.ps1`
- `.github/workflows/ci.yml`
- `.github/workflows/compatibility.yml`
- `.github/workflows/critical-fixes.yml`

`tests/test_critical_bottlenecks_benchmarks.py` remains opt-in benchmark
coverage rather than a default fast gate.

## Contractual Budgets

The only contractual benchmark budgets currently enforced are the synthetic
guardrails in `tests/test_critical_bottlenecks_benchmarks.py`:

- `mc_dropout`: `time_delta_pct <= +5%`
- `fourier_modes`: `time_delta_pct <= +15%`

These are regression budgets for the benchmark harness itself. They are not a
promise about absolute runtime on every machine.

Additional contractual behavior:

- `flow_checkpointing` is not a speed budget. Its contract is equivalence and
  training-memory intent.
- Optuna must continue to propagate `--compile-mode` explicitly.
- Inference must not re-fit preprocessing.
- Structural optimization paths must stay explicitly reversible.

## Exploratory Results

The following results are exploratory rather than contractual:

- synthetic benchmark deltas from `docs/optimization/` era scripts
- canonical runtime expectations like "30-50% faster"
- CPU-only `tracemalloc` observations for GPU-oriented features
- local benchmark numbers collected on Python 3.13.6 in `.venv-win`
- hardware-specific speedups from `torch.compile`

These results are useful for investigation and comparison, but they must not be
treated as acceptance criteria unless converted into tests or documented
budgets.

## Current Structural Optimization State

Task 4 is closed with the following controls:

- `torch.compile`
  - explicit disable sentinels such as `none`, `off`, `false`, and `disabled`
    normalize to disabled compile behavior
  - low-SM defensive disablement remains active
  - incompatible paths fall back safely to an uncompiled model
- preprocessing fold refit reduction
  - `allow_reuse_fitted_fold_preprocessor=False` by default
  - reuse is opt-in only
  - reuse is allowed only for matching fold cutoff and `fit_scope ==
    "fold_train_only"`
- Optuna path cleanup
  - subprocess/public-contract behavior is preserved
  - disabled compile mode is propagated as explicit sentinel `none`
  - canonical trial enqueue avoids duplicate insertion on resumed studies

## CPU, CUDA, Windows, Linux Expectations

Expected differences by environment:

- CPU vs CUDA
  - CUDA can change the value of `torch.compile` and `pin_memory`
    optimizations materially.
  - CPU benchmark results are useful for relative harness regressions, but they
    do not predict CUDA throughput or activation-memory savings.
  - `flow_checkpointing` may appear slower on CPU and tiny tensors while still
    being valid for training-memory-oriented use on GPU.

- Windows vs Linux
  - use `.venv-win` on Windows PowerShell and `.venv-linux` on Linux
  - hook execution on Windows should use the PowerShell entrypoints directly
  - line-ending and file-mode noise can differ, but quality gates should stay
    aligned with the same explicit commands

- Python 3.13.6 local vs CI 3.10/3.11
  - local profiling in `.venv-win` may use Python 3.13.6
  - compatibility and acceptance still target CI on Python 3.10 and 3.11
  - avoid optimization choices that depend on 3.12+ or 3.13-only behavior

## Recommended Publication Check

Before publishing a branch with runtime changes:

1. Run the explicit lint, smoke, fast pytest, and benchmark commands above.
2. Confirm the changed tests are covered by existing hooks/workflows.
3. Confirm `git status --short` is clean except for intended tracked changes.
4. Treat benchmark output as relative evidence unless the branch also updates a
   contractual guardrail.
