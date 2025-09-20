# Repository Guidelines

This repository contains advanced time‑series forecasting with FEDformer and probabilistic evaluation. Use this guide to contribute efficiently and consistently.

## Project Structure & Modules
- `main.py`: entry point for training/eval and CLI args.
- `config.py`: configuration defaults and helpers.
- `models/`: FEDformer and related model components.
- `training/`: training loops, schedulers, checkpoints.
- `utils/`: common utilities (metrics, I/O, seeds).
- `data/` and `reports/`: datasets and generated outputs (git‑ignored as appropriate).
- `tests/`: pytest suite; see fixtures in `tests/conftest.py`.

## Build, Test, Run
- `pip install -r requirements.txt`: install dependencies.
- `pytest -q`: run unit tests; honors `pytest.ini`.
- `pytest -q --cov`: run tests with coverage (requires `pytest-cov`).
- `python main.py --help`: inspect available commands/flags.
- Windows helpers: `.\n+verify_local.ps1` and `.
verify_and_smoke.ps1` create a venv, install deps, and run pytest/smoke tests.

## Coding Style & Naming
- Python 3.10+; follow PEP8 (4‑space indent, 88–120 col soft limit).
- Names: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Type hints required on public functions; prefer `typing` and dataclasses where suitable.
- Formatting/lint: run `pytest` locally; if you use a formatter, prefer `black` and `isort` (optional).

## Testing Guidelines
- Framework: `pytest` with fixtures in `tests/conftest.py`.
- Place tests in `tests/`, name `test_*.py`, functions `test_*`.
- GPU tests are conditionally skipped if CUDA is unavailable; keep them fast and guarded.
- Aim for coverage on core logic (models/, training/, utils/). Use small tensors and seeds for determinism.

## Commit & Pull Requests
- Commits: concise imperative subject (<=72 chars), body explaining what/why. Example: `train: fix LR scheduler warmup edge case`.
- PRs: include summary, motivation, key changes, test evidence (pytest output or screenshots), and link issues.
- CI expectations: PRs should pass `pytest -q` and avoid introducing large files to git.

## CI Workflow
- GitHub Actions: pushes/PRs run dependency install and `pytest -q`.
- Speed tips: mark long GPU tests with `@pytest.mark.slow` and guard with CUDA checks; provide CPU fallbacks where possible.
- Artifacts: upload concise logs only (avoid datasets/models). Ensure tests are deterministic with fixed seeds.
- Adding jobs: prefer matrix over duplication (e.g., `os: [ubuntu-latest, windows-latest]`, `python: [3.10, 3.11]`). Keep runtime < 10 min.

## CUDA Local Dev
- Requirements: latest NVIDIA driver, CUDA toolkit matching your PyTorch build, and `torch.cuda.is_available()` should be true.
- Install: follow PyTorch selector (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu121`). Then `pip install -r requirements.txt`.
- Quick check: `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- Repro tips: set `CUDA_VISIBLE_DEVICES`, use small batch sizes first, and seed RNGs. GPU tests in `tests/` are skipped automatically if CUDA is absent.

## Security & Configuration
- Do not commit secrets or large datasets; prefer env vars and `.gitignore`d paths under `data/`.
- Document any new flags in `main.py --help` and update `README.md` when adding user‑facing features.
