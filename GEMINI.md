
# GEMINI.md

## Project Overview

This project is a production-ready implementation of FEDformer (Frequency Enhanced Decomposed Transformer) with Normalizing Flows for probabilistic time series forecasting. It is designed for financial markets, supply chain optimization, and other domains where uncertainty quantification is critical.

The model uses a FEDformer backbone for time series forecasting and a Normalizing Flow network to model the full probability distribution of future outcomes. This allows for not only point predictions but also for generating prediction intervals, calculating risk metrics (VaR, CVaR), and running portfolio simulations.

The project is written in Python and uses PyTorch as the deep learning framework. It includes features like walk-forward backtesting, comprehensive performance metrics, and integration with Weights & Biases for experiment tracking.

## Building and Running

### 1. Installation

To set up the environment and install the required dependencies, run the following commands:

```powershell
# Clone the repository
git clone https://github.com/RbnGlz/Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting.git
cd Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting

# Create venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Model

The main entry point for the project is `main.py`. You can run the model with your own CSV data using the following command:

```powershell
python main.py \
    --csv data/your_data.csv \
    --targets "price,volume" \
    --date-col "timestamp" \
    --pred-len 24 \
    --seq-len 96 \
    --epochs 10 \
    --batch-size 32
```

For a full list of available command-line arguments, you can run:

```powershell
python main.py --help
```

### 3. Testing

The project includes a smoke test to verify that the training and backtesting pipeline is working correctly. To run the smoke test, use the following command:

```powershell
python main.py \
    --csv data/nvidia_stock_2024-08-20_to_2025-08-20.csv \
    --targets "Close" \
    --pred-len 8 \
    --seq-len 32 \
    --label-len 16 \
    --epochs 1 \
    --batch-size 8 \
    --splits 2 \
    --seed 123 \
    --no-show
```

## Development Conventions

*   **Configuration:** The project uses a dataclass-based configuration system (`config.py`) to manage all the parameters for the model, training, and data.
*   **Training:** The training process is implemented using a walk-forward cross-validation strategy (`training/trainer.py`) to ensure robust evaluation of the time series model.
*   **Modularity:** The code is organized into modules for data loading, model definition, training, and simulation, which makes it easier to understand and maintain.
*   **Error Handling:** The code includes comprehensive error handling to gracefully handle potential issues during execution.
*   **Logging:** The project uses the `logging` module to provide informative output during execution.
*   **Experiment Tracking:** The project is integrated with Weights & Biases for experiment tracking and visualization.
