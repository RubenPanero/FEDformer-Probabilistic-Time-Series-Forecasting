#!/usr/bin/env python3
"""
Main training script for Vanguard-FEDformer.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/financial.yaml --data_path data/sample/sp500_sample.csv
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vanguard_fedformer.core.training.trainer import VanguardTrainer
from vanguard_fedformer.core.data.dataset import TimeSeriesDataset
from vanguard_fedformer.core.models.fedformer import VanguardFEDformer
from vanguard_fedformer.utils.config import ConfigManager
from vanguard_fedformer.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Train Vanguard-FEDformer model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="models/",
                       help="Output directory for saved models")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Vanguard-FEDformer training")
    
    # Load configuration
    config = ConfigManager(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = TimeSeriesDataset(
        data_path=args.data_path,
        sequence_length=config.data.sequence_length,
        prediction_length=config.data.prediction_length,
        batch_size=config.data.batch_size
    )
    
    # Create model
    model = VanguardFEDformer(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        activation=config.model.activation
    ).to(device)
    
    # Create trainer
    trainer = VanguardTrainer(
        model=model,
        config=config,
        device=device
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(dataset)
    
    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path / "vanguard_fedformer.pt")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()