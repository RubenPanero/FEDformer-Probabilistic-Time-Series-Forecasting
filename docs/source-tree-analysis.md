# Source Tree Analysis

## Directory Structure

```
FEDformer-Probabilistic-Time-Series-Forecasting/
├── data/                # Data loading and preprocessing
│   ├── dataset.py       # PyTorch Dataset implementation
│   ├── preprocessing.py # Feature engineering and scaling
│   └── *_features.csv   # Raw/processed data files
├── models/              # Model architecture definitions
│   ├── fedformer.py     # Main FEDformer implementation
│   ├── layers.py        # Fourier blocks and attention layers
│   └── flows.py         # Normalizing flows for probabilistic output
├── training/            # Training and fine-tuning logic
│   ├── trainer.py       # Core Training loop
│   └── sequential_finetuner.py # Meta-learning / sequential training
├── inference/           # Prediction and model loading
│   ├── predictor.py     # Inference interface
│   └── loader.py        # Model serialization utilities
├── utils/               # Metrics and helper functions
├── simulations/         # Portfolio and risk simulations
├── tests/               # Comprehensive test suite
├── docs/                # Project documentation and plans
├── main.py              # Main entry point for training
└── tune_hyperparams.py  # Hyperparameter optimization (Optuna)
```

## Critical Folders

- **data/**: Handles the "Probabilistic" part by preparing features and managing datasets for different tickers (NVDA, GOOGL).
- **models/**: Contains the core FEDformer architecture which uses frequency domain decomposition.
- **training/**: Manages the lifecycle of model training, including validation and fine-tuning.
- **inference/**: Provides tools for using trained models to generate future forecasts.
- **simulations/**: Bridges the gap between forecasts and financial decision-making (portfolio/risk).
