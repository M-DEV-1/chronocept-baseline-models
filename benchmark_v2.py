"""
Comprehensive benchmark script for improved Chronocept baselines.

This script implements all the baselines mentioned in the review feedback:
1. SBERT + FFNN & SBERT + BiLSTM (SoTA embedding baselines)
2. RoBERTa-base + linear head (MSE)
3. DeBERTa-V3-base + linear head (MSE)
4. DeBERTa-V3-base + skew-normal NLL head
5. DistilBERT + skew-normal NLL head
6. MT-DNN style multi-task learning
7. Legacy 9-pass concatenation (negative ablation)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Import our new models
from models_v2 import (
    SBERTFFNN, SBERTBiLSTM,
    RoBERTaRegression, DeBERTaRegression, DistilBERTRegression,
    MTDNNModel, LegacyBERTRegression
)
from utils_v2 import ImprovedDataLoader, TrainingManager, ExperimentLogger
from utils_v2.metrics import evaluate_model_comprehensive

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs comprehensive benchmarks for all baseline models."""
    
    def __init__(self, 
                 benchmark: str = "benchmark_1",
                 save_dir: str = "benchmark_results",
                 device: str = "auto"):
        
        self.benchmark = benchmark
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Running benchmark on {self.benchmark} using device: {self.device}")
        
        # Results storage
        self.results = {}
        
        # Define model configurations
        self.model_configs = self._get_model_configs()
    
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all baseline models."""
        return {
            # SBERT baselines (SoTA embedding baselines)
            "sbert_ffnn": {
                "model_class": SBERTFFNN,
                "params": {
                    "sbert_model": "all-MiniLM-L6-v2",
                    "hidden_dim": 512,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "axis_encoding": "single_sequence_markers",
                    "loss_type": "skew_normal"
                },
                "training": {
                    "epochs": 50,
                    "lr": 1e-3,
                    "warm_start_epochs": 0
                }
            },
            
            "sbert_bilstm": {
                "model_class": SBERTBiLSTM,
                "params": {
                    "sbert_model": "all-MiniLM-L6-v2",
                    "hidden_dim": 128,
                    "num_layers": 1,
                    "dropout": 0.1,
                    "axis_encoding": "single_sequence_markers",
                    "loss_type": "skew_normal"
                },
                "training": {
                    "epochs": 50,
                    "lr": 1e-3,
                    "warm_start_epochs": 0
                }
            },
            
            # RoBERTa baseline
            "roberta_mse": {
                "model_class": RoBERTaRegression,
                "params": {
                    "model_name": "roberta-base",
                    "pooling_type": "mean",
                    "head_type": "linear",
                    "axis_encoding": "single_sequence_markers",
                    "loss_type": "mse",
                    "dropout": 0.1
                },
                "training": {
                    "epochs": 10,
                    "lr": 1e-5,
                    "warm_start_epochs": 3
                }
            },
            
            # DeBERTa baselines
            "deberta_mse": {
                "model_class": DeBERTaRegression,
                "params": {
                    "model_name": "microsoft/deberta-v3-base",
                    "pooling_type": "mean",
                    "head_type": "linear",
                    "axis_encoding": "single_sequence_markers",
                    "loss_type": "mse",
                    "dropout": 0.1
                },
                "training": {
                    "epochs": 10,
                    "lr": 1e-5,
                    "warm_start_epochs": 3
                }
            },
            
            "deberta_skew": {
                "model_class": DeBERTaRegression,
                "params": {
                    "model_name": "microsoft/deberta-v3-base",
                    "pooling_type": "mean",
                    "head_type": "linear",
                    "axis_encoding": "single_sequence_markers",
                    "loss_type": "skew_normal",
                    "dropout": 0.1
                },
                "training": {
                    "epochs": 10,
                    "lr": 1e-5,
                    "warm_start_epochs": 3
                }
            },
            
            # DistilBERT baseline
            "distilbert_skew": {
                "model_class": DistilBERTRegression,
                "params": {
                    "model_name": "distilbert-base-uncased",
                    "pooling_type": "mean",
                    "head_type": "linear",
                    "axis_encoding": "single_sequence_markers",
                    "loss_type": "skew_normal",
                    "dropout": 0.1
                },
                "training": {
                    "epochs": 15,
                    "lr": 2e-5,
                    "warm_start_epochs": 3
                }
            },
            
            # MT-DNN baseline
            "mtdnn": {
                "model_class": MTDNNModel,
                "params": {
                    "model_name": "roberta-base",
                    "pooling_type": "mean",
                    "axis_encoding": "single_sequence_markers",
                    "loss_type": "skew_normal",
                    "dropout": 0.1,
                    "hidden_dim": 256
                },
                "training": {
                    "epochs": 15,
                    "lr": 1e-5,
                    "warm_start_epochs": 3
                }
            },
            
            # Legacy ablation
            "legacy_bert": {
                "model_class": LegacyBERTRegression,
                "params": {
                    "model_name": "bert-base-uncased",
                    "dropout": 0.1
                },
                "training": {
                    "epochs": 5,
                    "lr": 1e-4,
                    "warm_start_epochs": 0
                }
            }
        }
    
    def run_single_model(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single model and return results."""
        logger.info(f"Running {model_name}...")
        
        try:
            # Create data loader
            data_loader = ImprovedDataLoader(
                benchmark=self.benchmark,
                axis_encoding=config["params"].get("axis_encoding", "single_sequence_markers"),
                batch_size=16,
                shuffle=True
            )
            
            train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
            
            # Create model
            model_class = config["model_class"]
            model = model_class(**config["params"])
            model.to(self.device)
            
            # Create optimizer
            if hasattr(model, 'get_optimizer'):
                optimizer = model.get_optimizer(config["training"]["lr"])
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
            
            # Create training manager
            experiment_name = f"{model_name}_{self.benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            training_manager = TrainingManager(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                loss_fn=model.loss_fn,
                device=self.device,
                save_dir=str(self.save_dir),
                experiment_name=experiment_name
            )
            
            # Train model
            training_history = training_manager.train(
                epochs=config["training"]["epochs"],
                warm_start_epochs=config["training"]["warm_start_epochs"]
            )
            
            # Evaluate model
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    texts = batch['texts']
                    axes_data = batch.get('axes_data', None)
                    targets = batch['targets']
                    
                    predictions = model(texts, axes_data)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets)
            
            predictions = torch.cat(all_predictions, dim=0).numpy()
            targets = torch.cat(all_targets, dim=0).numpy()
            
            # Compute comprehensive metrics
            metrics = evaluate_model_comprehensive(
                predictions, targets, 
                loss_type=config["params"].get("loss_type", "mse")
            )
            
            # Log results
            experiment_logger = ExperimentLogger(training_manager.experiment_dir)
            experiment_logger.log_experiment_config({
                "model_name": model_name,
                "benchmark": self.benchmark,
                "model_params": config["params"],
                "training_params": config["training"],
                "device": str(self.device)
            })
            experiment_logger.log_training_results(training_history)
            experiment_logger.log_evaluation_results(metrics)
            experiment_logger.log_model_info({
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            })
            experiment_logger.create_summary_report()
            
            logger.info(f"Completed {model_name}")
            logger.info(f"Best validation loss: {training_history.get('best_val_loss', 'N/A')}")
            logger.info(f"Test metrics: {metrics}")
            
            return {
                "model_name": model_name,
                "training_history": training_history,
                "metrics": metrics,
                "experiment_dir": str(training_manager.experiment_dir)
            }
            
        except Exception as e:
            logger.error(f"Error running {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "error": str(e),
                "training_history": None,
                "metrics": None,
                "experiment_dir": None
            }
    
    def run_all_models(self, model_names: List[str] = None) -> Dict[str, Any]:
        """Run all models or specified subset."""
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        logger.info(f"Running {len(model_names)} models: {model_names}")
        
        results = {}
        for model_name in model_names:
            if model_name not in self.model_configs:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            config = self.model_configs[model_name]
            result = self.run_single_model(model_name, config)
            results[model_name] = result
        
        self.results = results
        self._save_benchmark_results()
        return results
    
    def _save_benchmark_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.save_dir / f"benchmark_results_{self.benchmark}_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                "model_name": result["model_name"],
                "training_history": result.get("training_history"),
                "metrics": result.get("metrics"),
                "experiment_dir": result.get("experiment_dir"),
                "error": result.get("error")
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def create_comparison_table(self) -> str:
        """Create a comparison table of all results."""
        if not self.results:
            return "No results available"
        
        # Create table header
        table = "\\n" + "="*100 + "\\n"
        table += f"BENCHMARK RESULTS - {self.benchmark.upper()}\\n"
        table += "="*100 + "\\n"
        table += f"{'Model':<20} {'MSE':<10} {'MAE':<10} {'R²':<10} {'NLL':<10} {'CRPS':<10} {'Spearman':<10}\\n"
        table += "-"*100 + "\\n"
        
        # Add results for each model
        for model_name, result in self.results.items():
            if result.get("error"):
                table += f"{model_name:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}\\n"
                continue
            
            metrics = result.get("metrics", {})
            table += f"{model_name:<20} "
            table += f"{metrics.get('mse', 'N/A'):<10.4f} "
            table += f"{metrics.get('mae', 'N/A'):<10.4f} "
            table += f"{metrics.get('r2', 'N/A'):<10.4f} "
            table += f"{metrics.get('nll_skew', metrics.get('nll_gauss', metrics.get('nll_simple', 'N/A'))):<10.4f} "
            table += f"{metrics.get('crps_skew', metrics.get('crps_gauss', metrics.get('crps_simple', 'N/A'))):<10.4f} "
            table += f"{metrics.get('spearman_xi', 'N/A'):<10.4f}\\n"
        
        table += "="*100 + "\\n"
        
        return table


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run Chronocept baseline benchmarks")
    parser.add_argument("--benchmark", choices=["benchmark_1", "benchmark_2"], 
                       default="benchmark_1", help="Which benchmark to run")
    parser.add_argument("--models", nargs="+", help="Specific models to run (default: all)")
    parser.add_argument("--save_dir", default="benchmark_results", help="Directory to save results")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        benchmark=args.benchmark,
        save_dir=args.save_dir,
        device=args.device
    )
    
    # Run benchmarks
    results = runner.run_all_models(args.models)
    
    # Print comparison table
    print(runner.create_comparison_table())
    
    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main()
