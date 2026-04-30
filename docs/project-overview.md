# Project Overview

## FEDformer Probabilistic Time-Series Forecasting

This project implements a state-of-the-art probabilistic forecasting system for financial time series. It combines the frequency-domain decomposition strengths of **FEDformer** with **Normalizing Flows** to provide not just point forecasts, but full probability distributions (quantiles, uncertainty bands).

## Key Features

- **Advanced Backbone**: Uses FEDformer (Frequency Enhanced Decomposed Transformer).
- **Probabilistic Output**: Integrated Normalizing Flows for stochastic path generation.
- **Robust Validation**: Walk-forward temporal validation and conformal calibration.
- **Financial Focus**: Built-in metrics like Sharpe, Sortino, and Drawdown.
- **Automation**: Hyperparameter optimization via Optuna and experiment tracking via WandB.

## Tech Stack Summary

- **Primary Language**: Python
- **ML Framework**: PyTorch
- **Numerical Ops**: NumPy, Pandas, SciPy, Scikit-Learn
- **Financial Analysis**: yfinance, pandas-ta-classic
- **Visualization**: Matplotlib, Seaborn
- **Optimization**: Optuna
- **Experiment Tracking**: WandB
