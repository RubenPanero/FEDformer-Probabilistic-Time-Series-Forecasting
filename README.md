# FEDformer — Probabilistic Time-Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg)]()
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ci.yml)
[![Ruff Lint](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ruff.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ruff.yml)
[![Pylint](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/pylint.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/pylint.yml)
[![Security](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/security.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/security.yml)
[![Compatibility](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/compatibility.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/compatibility.yml)
[![Regression Guards](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/critical-fixes.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/critical-fixes.yml)

A production-ready, optimized implementation of FEDformer (Frequency Enhanced Decomposed Transformer) with **Normalizing Flows** for probabilistic time series forecasting. This system goes beyond point predictions to model the full probability distribution of future outcomes, making it ideal for financial markets, supply chain optimization, and any domain where uncertainty quantification is critical.

## 🌟 Key Features

### 🎯 **Probabilistic Forecasting**
- **Full Distribution Modeling**: Uses Normalizing Flows to learn complex, non-Gaussian distributions
- **Uncertainty Quantification**: Provides confidence intervals, VaR, CVaR, and Expected Shortfall
- **Risk-Aware Predictions**: Essential for financial applications and decision-making under uncertainty

### 🧠 **Advanced Architecture**
- **Frequency Domain Attention**: Fourier-based attention mechanism for efficient long-sequence modeling
- **Series Decomposition**: Automatic trend-seasonal decomposition with multiple kernel sizes
- **Regime-Adaptive Learning**: Detects and adapts to different market volatility regimes
- **Flow-Based Distributions**: RealNVP-style normalizing flows for empirical distribution learning

### ⚡ **Performance Optimizations**
- **Memory Efficient**: Gradient checkpointing, optimized tensor operations, and smart caching
- **GPU Accelerated**: Mixed precision training (AMP), CUDA optimization, model compilation
- **Scalable Architecture**: Supports distributed training and large-scale datasets

### 🔬 **Production Ready**
- **Walk-Forward Backtesting**: Realistic evaluation with temporal splits
- **Comprehensive Metrics**: Sharpe ratio, maximum drawdown, Sortino ratio, risk metrics
- **Robust Error Handling**: Graceful degradation, extensive logging, validation checks
- **Monitoring Integration**: Weights & Biases logging, real-time metrics tracking

## 🏗️ Architecture Overview

```text
CSV -> TimeSeriesDataset / PreprocessingPipeline (scale + regimes)
    -> WalkForwardTrainer (walk-forward, anti-leakage)
        -> Flow_FEDformer: x_enc, x_dec, x_regime -> Distribution
            -> FEDformer Encoder/Decoder (Fourier attention)
            -> Normalizing Flows -> probabilistic forecasts
    -> ForecastOutput (preds, ground_truth, quantiles, samples)
    -> RiskSimulator + PortfolioSimulator -> Sharpe, Sortino, MaxDD
```

### Core Components

1. **FEDformer Backbone**
   - Frequency Enhanced Decomposed Transformer
   - Fourier attention for efficient long-range dependencies
   - Multi-scale series decomposition

2. **Regime Detection System**
   - Automatic volatility regime identification
   - Adaptive embedding for different market conditions
   - Context-aware feature conditioning

3. **Normalizing Flow Network**
   - Affine coupling layers for invertible transformations
   - Context-conditioned flow parameters
   - Base distribution modeling with proper device handling

4. **Risk Simulation Engine**
   - Monte Carlo sampling for uncertainty quantification
   - Advanced risk metrics (VaR, CVaR, Expected Shortfall)
   - Portfolio simulation with realistic trading strategies

## 📈 Key Capabilities

### Probabilistic Outputs
- **Point Estimates**: Mean predictions for traditional forecasting
- **Prediction Intervals**: Confidence bands at any desired level
- **Risk Metrics**: VaR, CVaR, Expected Shortfall calculations
- **Sample Generation**: Draw multiple scenarios from learned distribution

### Advanced Evaluation
- **Walk-Forward Backtesting**: Time-aware evaluation with realistic constraints
- **Portfolio Simulation**: Strategy backtesting with comprehensive performance metrics
- **Risk Assessment**: Comprehensive risk analysis including drawdown, volatility
- **Regime Analysis**: Performance across different market conditions

### Flexible Configuration
- **Multi-Asset Support**: Handle multiple correlated time series
- **Configurable Horizons**: Short-term to long-term forecasting
- **Scalable Architecture**: From single GPU to distributed training
- **Extensive Hyperparameters**: Fine-tune every aspect of the model

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting.git
cd FEDformer-Probabilistic-Time-Series-Forecasting

# Create venv (Linux)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Canonical headless NVDA run (seed=7)
MPLBACKEND=Agg python3 main.py \
    --csv data/NVDA_features.csv \
    --targets "Close" \
    --seq-len 96 \
    --pred-len 20 \
    --batch-size 64 \
    --splits 4 \
    --return-transform log_return \
    --metric-space returns \
    --gradient-clip-norm 0.5 \
    --seed 7 \
    --save-results \
    --save-canonical \
    --no-show

# Canonical headless GOOGL run (seed=7)
MPLBACKEND=Agg python3 main.py \
    --csv data/GOOGL_features.csv \
    --targets "Close" \
    --seq-len 96 \
    --pred-len 20 \
    --batch-size 64 \
    --splits 4 \
    --return-transform log_return \
    --metric-space returns \
    --gradient-clip-norm 0.5 \
    --seed 7 \
    --save-results \
    --save-canonical \
    --no-show
```

### Advanced Configuration

```bash
MPLBACKEND=Agg python3 main.py \
    --csv data/financial_data.csv \
    --targets "close_price" \
    --date-col "date" \
    --pred-len 24 \
    --seq-len 96 \
    --label-len 48 \
    --epochs 15 \
    --batch-size 64 \
    --splits 5 \
    --use-checkpointing \
    --grad-accum-steps 2 \
    --wandb-project "fedformer-experiment" \
    --wandb-entity "your-team" \
    --seed 123 \
    --deterministic \
    --save-fig results/portfolio.png \
    --no-show
```

## Resultados canonicos (seed=7)

| Ticker | Sharpe | Sortino | MaxDD  | Config |
|--------|--------|---------|--------|--------|
| NVDA   | +0.990 | +1.857  | -54.2% | seq=96, pred=20, batch=64, splits=4 |
| GOOGL  | +0.737 | +1.009  | -40.2% | seq=96, pred=20, batch=64, splits=4 |

Config comun: `log_return`, `metric_space=returns`, `gradient_clip_norm=0.5`, `seed=7`

## Inference CLI

Canonical inference requires checkpoints trained with `--save-canonical`, which stores both the model checkpoint and preprocessing artifacts. The repository currently ships canonical specialists for `NVDA` and `GOOGL`.

```bash
# Canonical model inference
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv
python3 -m inference --ticker GOOGL --csv data/GOOGL_features.csv

# Export predictions to CSV
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --output results/preds.csv

# Fan chart + calibration plot
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --plot --output-dir results/

# List registered canonical specialists
python3 -m inference --list-models
```

## Visualizacion probabilistica

The inference CLI can generate:

- Fan chart: p10-p90 band, p50 median, and ground truth.
- Calibration plot: reliability view and PIT histogram for probabilistic quality checks.
- Output files: `results/fan_chart_{ticker}.png` and `results/calibration_{ticker}.png`.

In headless environments, prefer `MPLBACKEND=Agg` and `--no-show` for training runs.

## 🗃️ Data Format

Your CSV should contain:
- **Target columns**: Variables to predict (e.g., 'price', 'volume')
- **Feature columns**: Additional predictors (e.g., 'volume', 'volatility')
- **Date column** (optional): Timestamp column to exclude from features

Example:
```csv
date,close_price,volume,volatility,rsi
2023-01-01,100.5,1000000,0.15,45.2
2023-01-02,101.2,1200000,0.18,47.8
...
```

## 🎛️ Configuration Parameters

### Model Architecture
- `d_model`: Hidden dimension (default: 512)
- `n_heads`: Number of attention heads (default: 8)
- `e_layers`: Encoder layers (default: 2)
- `modes`: Fourier modes for frequency attention (default: 64)
- `activation`: Activation function ('gelu' or 'relu')

### Sequence Configuration
- `seq_len`: Input sequence length (default: 96)
- `label_len`: Decoder start tokens (default: 48)
- `pred_len`: Prediction horizon (default: 24, must be even)

### Normalizing Flow
- `n_flow_layers`: Number of coupling layers (default: 4)
- `flow_hidden_dim`: Hidden dimension in flows (default: 64)

### Training
- `learning_rate`: Learning rate (default: 1e-4)
- `batch_size`: Batch size (default: 32)
- `n_epochs_per_fold`: Epochs per fold (default: 5)
- `use_amp`: Mixed precision training (default: True)
- `use_gradient_checkpointing`: Memory optimization (default: False)
- `gradient_accumulation_steps`: Effective larger batch size (default: 1)

## 📊 Performance Metrics

### Forecasting Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Normalized metrics** for cross-series comparison

### Probabilistic Metrics
- **Negative Log-Likelihood**: Distribution fitting quality
- **Coverage**: Prediction interval accuracy
- **Calibration**: Reliability of uncertainty estimates

### Financial Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst-case loss
- **Sortino Ratio**: Downside risk-adjusted returns
- **Volatility**: Return variability

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Expected Shortfall**: Average of worst-case scenarios

## 🧰 Advanced Features

### Memory Optimization
```python
config = FEDformerConfig(
    use_gradient_checkpointing=True,
    batch_size=16,
    compile_mode='max-autotune'
)
```

### Multi-GPU Training (current status and alternatives)

- Status: distributed multi-GPU training is not enabled yet in this repository.
- Alternatives today:
  - Use gradient checkpointing with `--use-checkpointing` to reduce memory.
  - Increase effective batch size with `--grad-accum-steps <N>`.
  - Run multiple independent processes (e.g., different seeds/datasets) to parallelize experiments.

Note: When DDP support is added, usage of `torchrun` and the corresponding flags will be documented.

### Custom Risk Analysis
```python
# After training
risk_sim = RiskSimulator(samples_oos)
var_95 = risk_sim.calculate_var()
cvar_95 = risk_sim.calculate_cvar()
expected_shortfall = risk_sim.calculate_expected_shortfall()
```

## 🧪 Reproducibility

- Set seeds and deterministic mode:
  - `--seed 123 --deterministic`
- DataLoader workers are seeded for consistent shuffling.
- cuDNN deterministic may reduce speed; disable with `--deterministic` off.
- Canonical seed for repository benchmarks is `7`.

## 📈 Visualization

- Use `--save-fig path.png` to save plots instead of showing them.
- Use `--no-show` in headless environments.
- For canonical Linux runs, prefer `MPLBACKEND=Agg`.

## 🧪 Testing

```bash
# Fast local CI shard
pytest -q -m "not slow"

# Full project validation
pytest -q

# Lint + format parity with CI
ruff check .
ruff format --check .

# Pre-commit shorthand
make ci-check
# Equivalent to:
# ruff check . --fix && ruff format . && \
# pylint --errors-only models/ training/ data/ utils/ inference/ && \
# pytest -q -m "not slow"
```

### Local smoke test

```bash
MPLBACKEND=Agg python3 main.py \
    --csv data/your_data.csv \
    --targets "price" \
    --pred-len 8 \
    --seq-len 32 \
    --label-len 16 \
    --epochs 1 \
    --batch-size 8 \
    --splits 2 \
    --seed 123 \
    --no-show
```

## ⚠️ Notes & Limitations

- Current Fourier attention uses a simplified formulation that does not incorporate `V` explicitly; empirically effective but differs from standard attention.
- `pred_len` must be even (affine coupling split requirement).
- Distributed training (DDP) not implemented yet.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FEDformer Paper**: Zhou et al. "FEDformer: Frequency Enhanced Decomposed Transformer"
- **Normalizing Flows**: Dinh et al. "Density estimation using Real NVP"
- **PyTorch Team**: For the excellent deep learning framework

**⭐ If you find this project useful, please consider starring it on GitHub!**



