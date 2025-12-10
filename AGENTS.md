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

## FEDformer Model Updates (Sep 2025)
- `Flow_FEDformer` now centralizes submodules inside `components` and `sequence_layers`, reducing attribute noise and sharing dropout logic (`models/fedformer.py:27`, `models/fedformer.py:67`). Helpers `_prepare_decoder_input`, `_attach_regime_vectors`, and `_embed_with_dropout` consolidate seasonal/trend preparation and regime conditioning (`models/fedformer.py:86`, `models/fedformer.py:106`, `models/fedformer.py:124`).
- Gradient checkpointing passes registered modules directly to `torch.utils.checkpoint.checkpoint`, avoiding lambda wrappers and better aligning with PyTorch 2.5+ (`models/fedformer.py:144`). Flow conditioning reshapes once and reuses the shared projection stored in `components["flow_conditioner_proj"]` (`models/fedformer.py:166`).
- `NormalizingFlowDistribution` documents its API, keeps per-feature contexts, and exposes a sampling helper for downstream evaluation (`models/fedformer.py:180`).
- Encoder/decoder layers mirror the refactor by switching to `nn.ModuleDict` collections and the new `AttentionConfig` to reuse typed hyperparameters (`models/encoder_decoder.py:35`, `models/encoder_decoder.py:96`; `models/layers.py:36`).
- Fourier attention now wraps FFT/conv calls for static analysis safety while preserving cross-length interpolation behaviour (`models/layers.py:16`, `models/layers.py:151`).
- `FEDformerConfig` is split into typed sections (`config.sections`) but preserves the legacy keyword/attribute surface; validation still infers encoder/decoder dimensions and clamps Fourier modes as needed (`config.py:20`, `config.py:195`, `config.py:210`).

### Usage Refresher
```python
from config import FEDformerConfig
from models.fedformer import Flow_FEDformer

config = FEDformerConfig(
    target_features=["load"],
    file_path="data/train.csv",
    use_gradient_checkpointing=True,
    moving_avg=[24, 48, 96],
)

# Optional: tweak grouped settings via the new sections container.
config.sections.model.transformer.dropout = 0.2

model = Flow_FEDformer(config)
distribution = model(x_enc, x_dec, x_regime)
loss = -distribution.log_prob(y_true).mean()
```

### Migration Notes
- Access decoder/encoder and embeddings through `model.components[...]` and `model.sequence_layers[...]` (e.g., `model.components["dropout"]`) instead of the old attributes (`model.dropout`, `model.encoder`, etc.).
- `use_gradient_checkpointing` is read from `model.config.use_gradient_checkpointing`; direct reads of `model.use_gradient_checkpointing` will fail.
- Custom config overrides can continue using attribute syntax, but nested edits should prefer the grouped sections to stay Pylint-compliant (e.g., `config.sections.training.runtime.compile_mode = "reduce-overhead"`).

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
