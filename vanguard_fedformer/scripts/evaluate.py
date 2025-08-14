#!/usr/bin/env python3
"""
Evaluation script for Vanguard-FEDformer.

Usage:
    python evaluate.py --model models/vanguard_fedformer.pt --data data/sample/sp500_sample.csv
    python evaluate.py --model models/vanguard_fedformer.pt --config configs/financial.yaml
"""

import argparse
import torch
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vanguard_fedformer.core.evaluation.metrics import ProbabilisticMetrics, RegimeMetrics
from vanguard_fedformer.core.evaluation.backtesting import WalkForwardBacktester
from vanguard_fedformer.core.evaluation.risk_analysis import RiskSimulator
from vanguard_fedformer.core.models.fedformer import VanguardFEDformer
from vanguard_fedformer.core.data.dataset import TimeSeriesDataset
from vanguard_fedformer.utils.config import ConfigManager
from vanguard_fedformer.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Evaluate Vanguard-FEDformer model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to test data")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="results/",
                       help="Output directory for evaluation results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Vanguard-FEDformer evaluation")
    
    # Load configuration
    config = ConfigManager(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model = VanguardFEDformer.load(args.model, device=device)
    logger.info(f"Loaded model from {args.model}")
    
    # Create dataset
    dataset = TimeSeriesDataset(
        data_path=args.data,
        sequence_length=config.data.sequence_length,
        prediction_length=config.data.prediction_length,
        batch_size=1  # Evaluation with batch size 1
    )
    
    # Initialize evaluators
    prob_metrics = ProbabilisticMetrics()
    regime_metrics = RegimeMetrics()
    backtester = WalkForwardBacktester(
        model=model,
        dataset=dataset,
        config=config
    )
    risk_simulator = RiskSimulator(config=config)
    
    # Run evaluation
    logger.info("Running probabilistic evaluation...")
    prob_results = prob_metrics.evaluate(model, dataset)
    
    logger.info("Running regime detection evaluation...")
    regime_results = regime_metrics.evaluate(model, dataset)
    
    logger.info("Running walk-forward backtesting...")
    backtest_results = backtester.run()
    
    logger.info("Running risk analysis...")
    risk_results = risk_simulator.analyze(backtest_results)
    
    # Combine results
    evaluation_results = {
        "probabilistic_metrics": prob_results,
        "regime_metrics": regime_results,
        "backtest_results": backtest_results,
        "risk_analysis": risk_results
    }
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Print summary
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to {output_path / 'evaluation_results.json'}")
    
    # Print key metrics
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Probabilistic CRPS: {prob_results.get('crps', 'N/A'):.4f}")
    print(f"Regime Accuracy: {regime_results.get('accuracy', 'N/A'):.4f}")
    print(f"Backtest Sharpe: {backtest_results.get('sharpe_ratio', 'N/A'):.4f}")
    print(f"VaR (95%): {risk_results.get('var_95', 'N/A'):.4f}")

if __name__ == "__main__":
    main()