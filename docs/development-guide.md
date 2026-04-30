# Development Guide

## Setup

1. **Virtual Environment**:
   - Windows: `python -m venv .venv-win && .venv-win\Scripts\Activate.ps1`
   - Linux: `python -m venv .venv-linux && source .venv-linux/bin/activate`
2. **Dependencies**: `pip install -r requirements.txt`
3. **Optional Dev Tools**: `pip install ruff pylint pytest`

## Main Workflows

### 1. Training a Model
Use `main.py` as the primary entry point:
```bash
python main.py --csv data/NVDA_features.csv --targets close --seq_len 96 --label_len 48 --pred_len 24
```

### 2. Hyperparameter Tuning
Run the Optuna search:
```bash
python tune_hyperparams.py --csv data/NVDA_features.csv --n_trials 100
```

### 3. Inference
Run the inference CLI:
```bash
python -m inference --model_path checkpoints/my_specialist.pt --input_csv data/NVDA_features.csv
```

## Testing
Run the comprehensive test suite:
```bash
pytest tests/
```

## Quality Standards
- **Linter**: `ruff check .`
- **Type Checking**: `mypy .` (if configured)
- **Hooks**: See `.github/workflows` for CI pipelines.
