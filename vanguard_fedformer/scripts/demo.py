#!/usr/bin/env python3
"""
Demo script for Vanguard-FEDformer.

This script demonstrates the basic usage of the model
and can be converted to a Jupyter notebook.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vanguard_fedformer.core.models.fedformer import VanguardFEDformer
from vanguard_fedformer.core.data.dataset import TimeSeriesDataset
from vanguard_fedformer.core.training.trainer import VanguardTrainer
from vanguard_fedformer.utils.config import ConfigManager
from vanguard_fedformer.utils.visualization import plot_forecasts, plot_regimes

def generate_sample_data(n_samples=1000, n_features=5):
    """Generate synthetic time series data for demonstration."""
    np.random.seed(42)
    
    # Generate trend and seasonality
    t = np.linspace(0, 10, n_samples)
    trend = 0.1 * t
    seasonality = 0.5 * np.sin(2 * np.pi * t) + 0.3 * np.sin(4 * np.pi * t)
    
    # Generate features
    data = []
    for i in range(n_features):
        noise = np.random.normal(0, 0.1, n_samples)
        feature = trend + seasonality + noise
        data.append(feature)
    
    # Add some regime changes
    regime_change = np.where(t > 5, 1.5, 1.0)
    data = [d * regime_change for d in data]
    
    return np.column_stack(data)

def main():
    print("=== Vanguard-FEDformer Demo ===\n")
    
    # Load configuration
    config = ConfigManager("configs/default.yaml")
    print("✓ Configuration loaded")
    
    # Generate sample data
    print("Generating sample data...")
    sample_data = generate_sample_data()
    
    # Save sample data
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(data_dir / "sp500_sample.csv", sample_data, delimiter=",")
    print("✓ Sample data generated and saved")
    
    # Create dataset
    dataset = TimeSeriesDataset(
        data_path=str(data_dir / "sp500_sample.csv"),
        sequence_length=config.data.sequence_length,
        prediction_length=config.data.prediction_length,
        batch_size=config.data.batch_size
    )
    print("✓ Dataset created")
    
    # Create model
    model = VanguardFEDformer(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        activation=config.model.activation
    )
    print("✓ Model created")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Quick forward pass test
    print("\nTesting forward pass...")
    try:
        # Get a sample batch
        sample_batch = next(iter(dataset))
        x, y = sample_batch
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Target shape: {y.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    # Demonstrate regime detection
    print("\nDemonstrating regime detection...")
    try:
        # Simple regime detection based on volatility
        returns = np.diff(sample_data[:, 0])  # First feature returns
        volatility = np.abs(returns)
        
        # Simple threshold-based regime detection
        regime_threshold = np.percentile(volatility, 75)
        regimes = (volatility > regime_threshold).astype(int)
        
        print(f"✓ Regime detection completed")
        print(f"  High volatility periods: {np.sum(regimes)}")
        print(f"  Regime change points: {np.sum(np.diff(regimes) != 0)}")
        
    except Exception as e:
        print(f"✗ Regime detection failed: {e}")
    
    # Demonstrate probabilistic forecasting
    print("\nDemonstrating probabilistic forecasting...")
    try:
        # Generate multiple forecasts with noise
        n_forecasts = 100
        forecasts = []
        
        for _ in range(n_forecasts):
            # Add noise to input for ensemble
            noisy_x = x + torch.randn_like(x) * 0.01
            with torch.no_grad():
                forecast = model(noisy_x)
            forecasts.append(forecast.numpy())
        
        forecasts = np.array(forecasts)
        
        # Calculate confidence intervals
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)
        
        print(f"✓ Probabilistic forecasting completed")
        print(f"  Forecast ensemble size: {n_forecasts}")
        print(f"  Mean forecast shape: {mean_forecast.shape}")
        print(f"  Forecast uncertainty: {np.mean(std_forecast):.4f}")
        
    except Exception as e:
        print(f"✗ Probabilistic forecasting failed: {e}")
    
    print("\n=== Demo Completed Successfully! ===")
    print("\nNext steps:")
    print("1. Run training: python scripts/train.py --data_path data/sample/sp500_sample.csv")
    print("2. Run evaluation: python scripts/evaluate.py --model models/vanguard_fedformer.pt --data data/sample/sp500_sample.csv")
    print("3. Check out the notebooks in the notebooks/ directory")
    print("4. Explore the configuration files in configs/")

if __name__ == "__main__":
    main()