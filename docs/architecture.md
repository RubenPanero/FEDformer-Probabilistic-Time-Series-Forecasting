# Architecture Documentation

## Overview

The project follows a modular Machine Learning pipeline architecture designed for high reproducibility and production readiness.

## Core Components

### 1. Data Layer (`data/`)
Responsible for ingesting raw CSV or financial data (`yfinance`), applying technical indicators (`pandas-ta`), and preparing PyTorch-compatible datasets. It emphasizes temporal consistency to prevent data leakage.

### 2. Model Layer (`models/`)
Implements the hybrid model:
- **FEDformer**: Frequency-domain Transformer that decomposes time series into trend and seasonal components.
- **Normalizing Flows**: Post-processing head that transforms a simple distribution into a complex one representing forecast uncertainty.

### 3. Training Engine (`training/`)
Orchestrates the training process using a `WalkForwardTrainer`. This ensures the model is evaluated on data it hasn't seen yet, mimicking real-world trading conditions.

### 4. Inference Engine (`inference/`)
Provides a standalone predictor interface to load "Specialist" models (trained on specific tickers) and generate future-looking forecasts.

### 5. Simulation & Risk (`simulations/`)
Converts probabilistic forecasts into actionable financial insights, simulating portfolio performance and calculating risk metrics (VaR, CVaR).

## Design Patterns

- **Specialist Model Pattern**: Instead of one general model, the system encourages "specialists" trained and calibrated for specific assets.
- **Conformal Prediction**: Used to ensure that uncertainty bands are statistically valid and well-calibrated.
- **Experiment Registry**: All runs are tracked with unique IDs and manifests for full traceability.
