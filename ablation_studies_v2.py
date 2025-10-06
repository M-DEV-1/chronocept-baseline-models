"""
Ablation studies for Chronocept baselines.

This script implements the ablation studies mentioned in the review:
- Axis Encoding: NoAxes, SingleSeq_Markers, SBERT_Concat, 9-pass Concat
- Objectives: MSE, NLL_skew, NLL_gauss
- Heads: LinearShared, FFNN_shared, Shared trunk + 3 heads
- Pooling: pooler_output, mean_pool, attention_pool
- Training stability: Warm-start, Layer-wise LR decay
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import itertools

# Import our models
from models_v2 import (
    RoBERTaRegression, DeBERTaRegression,
    SBERTFFNN, SBERTBiLSTM
)
from utils_v2 import ImprovedDataLoader, TrainingManager, ExperimentLogger
from utils_v2.dataloader import ChronoceptDataset
from utils_v2.metrics import evaluate_model_comprehensive

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AblationRunner:
    """Runs ablation studies for different model components."""
    
    def __init__(self, 
                 benchmark: str = "benchmark_1",
                 save_dir: str = "ablation_results",
                 device: str = "auto"):
        
        self.benchmark = benchmark
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Running ablation studies on {self.benchmark} using device: {self.device}")
        
        # Results storage
        self.results = {}
    
    def run_axis_encoding_ablation(self) -> Dict[str, Any]:
        """Run ablation study on axis encoding strategies."""
        logger.info("Running axis encoding ablation study...")
        
        axis_encodings = [
            "no_axes",
            "single_sequence_markers", 
            "sbert_concat"
        ]
        
        results = {}
        
        for axis_encoding in axis_encodings:
            logger.info(f"Testing axis encoding: {axis_encoding}")
            
            try:
                # Create data loader
                data_loader = ImprovedDataLoader(
                    benchmark=self.benchmark,
                    axis_encoding=axis_encoding,
                    batch_size=16,
                    shuffle=True
                )
                
                train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
                
                # Use RoBERTa as base model
                model = RoBERTaRegression(
                    model_name="roberta-base",
                    pooling_type="mean",
                    head_type="linear",
                    axis_encoding=axis_encoding,
                    loss_type="skew_normal",
                    dropout=0.1
                )
                model.to(self.device)
                
                # Train model
                optimizer = model.get_optimizer(1e-5)
                experiment_name = f"axis_encoding_{axis_encoding}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
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
                
                training_history = training_manager.train(epochs=10, warm_start_epochs=3)
                
                # Evaluate
                metrics = self._evaluate_model(model, test_loader)
                
                results[axis_encoding] = {
                    "training_history": training_history,
                    "metrics": metrics,
                    "experiment_dir": str(training_manager.experiment_dir)
                }
                
                logger.info(f"Completed axis encoding: {axis_encoding}")
                
            except Exception as e:
                logger.error(f"Error in axis encoding {axis_encoding}: {str(e)}")
                results[axis_encoding] = {"error": str(e)}
        
        return results
    
    def run_objective_ablation(self) -> Dict[str, Any]:
        """Run ablation study on loss objectives."""
        logger.info("Running objective ablation study...")
        
        objectives = [
            ("mse", "MSE Loss"),
            ("gaussian", "Gaussian NLL"),
            ("skew_normal", "Skew-Normal NLL")
        ]
        
        results = {}
        
        for loss_type, loss_name in objectives:
            logger.info(f"Testing objective: {loss_name}")
            
            try:
                # Create data loader
                data_loader = ImprovedDataLoader(
                    benchmark=self.benchmark,
                    axis_encoding="single_sequence_markers",
                    batch_size=16,
                    shuffle=True
                )
                
                train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
                
                # Use RoBERTa as base model
                model = RoBERTaRegression(
                    model_name="roberta-base",
                    pooling_type="mean",
                    head_type="linear",
                    axis_encoding="single_sequence_markers",
                    loss_type=loss_type,
                    dropout=0.1
                )
                model.to(self.device)
                
                # Train model
                optimizer = model.get_optimizer(1e-5)
                experiment_name = f"objective_{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
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
                
                training_history = training_manager.train(epochs=10, warm_start_epochs=3)
                
                # Evaluate
                metrics = self._evaluate_model(model, test_loader)
                
                results[loss_type] = {
                    "loss_name": loss_name,
                    "training_history": training_history,
                    "metrics": metrics,
                    "experiment_dir": str(training_manager.experiment_dir)
                }
                
                logger.info(f"Completed objective: {loss_name}")
                
            except Exception as e:
                logger.error(f"Error in objective {loss_name}: {str(e)}")
                results[loss_type] = {"error": str(e)}
        
        return results
    
    def run_head_ablation(self) -> Dict[str, Any]:
        """Run ablation study on head architectures."""
        logger.info("Running head architecture ablation study...")
        
        head_configs = [
            ("linear", "Linear Head"),
            ("ffnn", "FFNN Head"),
            ("multitask", "Multi-Task Head")
        ]
        
        results = {}
        
        for head_type, head_name in head_configs:
            logger.info(f"Testing head: {head_name}")
            
            try:
                # Create data loader
                data_loader = ImprovedDataLoader(
                    benchmark=self.benchmark,
                    axis_encoding="single_sequence_markers",
                    batch_size=16,
                    shuffle=True
                )
                
                train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
                
                # Use RoBERTa as base model
                model = RoBERTaRegression(
                    model_name="roberta-base",
                    pooling_type="mean",
                    head_type=head_type,
                    axis_encoding="single_sequence_markers",
                    loss_type="skew_normal",
                    dropout=0.1
                )
                model.to(self.device)
                
                # Train model
                optimizer = model.get_optimizer(1e-5)
                experiment_name = f"head_{head_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
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
                
                training_history = training_manager.train(epochs=10, warm_start_epochs=3)
                
                # Evaluate
                metrics = self._evaluate_model(model, test_loader)
                
                results[head_type] = {
                    "head_name": head_name,
                    "training_history": training_history,
                    "metrics": metrics,
                    "experiment_dir": str(training_manager.experiment_dir)
                }
                
                logger.info(f"Completed head: {head_name}")
                
            except Exception as e:
                logger.error(f"Error in head {head_name}: {str(e)}")
                results[head_type] = {"error": str(e)}
        
        return results
    
    def run_pooling_ablation(self) -> Dict[str, Any]:
        """Run ablation study on pooling strategies."""
        logger.info("Running pooling strategy ablation study...")
        
        pooling_configs = [
            ("pooler", "Pooler Output"),
            ("mean", "Mean Pooling"),
            ("attention", "Attention Pooling")
        ]
        
        results = {}
        
        for pooling_type, pooling_name in pooling_configs:
            logger.info(f"Testing pooling: {pooling_name}")
            
            try:
                # Create data loader
                data_loader = ImprovedDataLoader(
                    benchmark=self.benchmark,
                    axis_encoding="single_sequence_markers",
                    batch_size=16,
                    shuffle=True
                )
                
                train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
                
                # Use RoBERTa as base model
                model = RoBERTaRegression(
                    model_name="roberta-base",
                    pooling_type=pooling_type,
                    head_type="linear",
                    axis_encoding="single_sequence_markers",
                    loss_type="skew_normal",
                    dropout=0.1
                )
                model.to(self.device)
                
                # Train model
                optimizer = model.get_optimizer(1e-5)
                experiment_name = f"pooling_{pooling_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
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
                
                training_history = training_manager.train(epochs=10, warm_start_epochs=3)
                
                # Evaluate
                metrics = self._evaluate_model(model, test_loader)
                
                results[pooling_type] = {
                    "pooling_name": pooling_name,
                    "training_history": training_history,
                    "metrics": metrics,
                    "experiment_dir": str(training_manager.experiment_dir)
                }
                
                logger.info(f"Completed pooling: {pooling_name}")
                
            except Exception as e:
                logger.error(f"Error in pooling {pooling_name}: {str(e)}")
                results[pooling_type] = {"error": str(e)}
        
        return results
    
    def run_training_stability_ablation(self) -> Dict[str, Any]:
        """Run ablation study on training stability techniques."""
        logger.info("Running training stability ablation study...")
        
        stability_configs = [
            (0, "No Warm Start"),
            (3, "3 Epoch Warm Start"),
            (5, "5 Epoch Warm Start")
        ]
        
        results = {}
        
        for warm_start_epochs, config_name in stability_configs:
            logger.info(f"Testing training stability: {config_name}")
            
            try:
                # Create data loader
                data_loader = ImprovedDataLoader(
                    benchmark=self.benchmark,
                    axis_encoding="single_sequence_markers",
                    batch_size=16,
                    shuffle=True
                )
                
                train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
                
                # Use RoBERTa as base model
                model = RoBERTaRegression(
                    model_name="roberta-base",
                    pooling_type="mean",
                    head_type="linear",
                    axis_encoding="single_sequence_markers",
                    loss_type="skew_normal",
                    dropout=0.1
                )
                model.to(self.device)
                
                # Train model
                optimizer = model.get_optimizer(1e-5)
                experiment_name = f"stability_{warm_start_epochs}warm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
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
                
                training_history = training_manager.train(
                    epochs=10, 
                    warm_start_epochs=warm_start_epochs
                )
                
                # Evaluate
                metrics = self._evaluate_model(model, test_loader)
                
                results[f"warm_{warm_start_epochs}"] = {
                    "config_name": config_name,
                    "warm_start_epochs": warm_start_epochs,
                    "training_history": training_history,
                    "metrics": metrics,
                    "experiment_dir": str(training_manager.experiment_dir)
                }
                
                logger.info(f"Completed training stability: {config_name}")
                
            except Exception as e:
                logger.error(f"Error in training stability {config_name}: {str(e)}")
                results[f"warm_{warm_start_epochs}"] = {"error": str(e)}
        
        return results
    
    def run_axis_shuffling_ablation(self) -> Dict[str, Any]:
        """Run ablation study on axis shuffling strategies."""
        logger.info("Running axis shuffling ablation study...")
        
        shuffling_configs = [
            (False, "No Shuffling"),
            (True, "Random Axis Shuffling")
        ]
        
        results = {}
        
        for shuffle_axes, config_name in shuffling_configs:
            logger.info(f"Testing axis shuffling: {config_name}")
            
            try:
                # Create data loader with axis shuffling
                data_loader = ImprovedDataLoader(
                    benchmark=self.benchmark,
                    axis_encoding="single_sequence_markers",
                    batch_size=16,
                    shuffle=True
                )
                
                # Apply axis shuffling to the dataset if enabled
                if shuffle_axes:
                    data_loader = self._apply_axis_shuffling(data_loader)
                
                train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
                
                # Use RoBERTa as base model
                model = RoBERTaRegression(
                    model_name="roberta-base",
                    pooling_type="mean",
                    head_type="linear",
                    axis_encoding="single_sequence_markers",
                    loss_type="skew_normal",
                    dropout=0.1
                )
                model.to(self.device)
                
                # Train model
                optimizer = model.get_optimizer(1e-5)
                experiment_name = f"axis_shuffling_{shuffle_axes}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
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
                
                training_history = training_manager.train(epochs=10, warm_start_epochs=3)
                
                # Evaluate
                metrics = self._evaluate_model(model, test_loader)
                
                results[f"shuffle_{shuffle_axes}"] = {
                    "config_name": config_name,
                    "shuffle_axes": shuffle_axes,
                    "training_history": training_history,
                    "metrics": metrics,
                    "experiment_dir": str(training_manager.experiment_dir)
                }
                
                logger.info(f"Completed axis shuffling: {config_name}")
                
            except Exception as e:
                logger.error(f"Error in axis shuffling {config_name}: {str(e)}")
                results[f"shuffle_{shuffle_axes}"] = {"error": str(e)}
        
        return results
    
    def _apply_axis_shuffling(self, data_loader: ImprovedDataLoader) -> ImprovedDataLoader:
        """Apply random shuffling to axis order in the dataset."""
        # Create a new dataset with shuffled axes
        shuffled_train_dataset = self._shuffle_axes_in_dataset(data_loader.train_dataset)
        shuffled_valid_dataset = self._shuffle_axes_in_dataset(data_loader.valid_dataset)
        shuffled_test_dataset = self._shuffle_axes_in_dataset(data_loader.test_dataset)
        
        # Create new data loaders with shuffled datasets
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            shuffled_train_dataset,
            batch_size=data_loader.batch_size,
            shuffle=data_loader.shuffle,
            collate_fn=data_loader._collate_fn
        )
        
        valid_loader = DataLoader(
            shuffled_valid_dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            collate_fn=data_loader._collate_fn
        )
        
        test_loader = DataLoader(
            shuffled_test_dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            collate_fn=data_loader._collate_fn
        )
        
        # Create a new data loader object
        new_data_loader = ImprovedDataLoader(
            benchmark=data_loader.benchmark,
            axis_encoding=data_loader.axis_encoding,
            max_length=data_loader.max_length,
            batch_size=data_loader.batch_size,
            shuffle=data_loader.shuffle,
            normalization=data_loader.normalization,
            log_scale=data_loader.log_scale
        )
        
        # Replace the data loaders
        new_data_loader.train_loader = train_loader
        new_data_loader.valid_loader = valid_loader
        new_data_loader.test_loader = test_loader
        
        return new_data_loader
    
    def _shuffle_axes_in_dataset(self, dataset: ChronoceptDataset) -> ChronoceptDataset:
        """Create a new dataset with shuffled axis order."""
        import random
        
        class ShuffledChronoceptDataset(ChronoceptDataset):
            def __getitem__(self, idx):
                sample = self.data[idx]
                
                # Extract parent text
                parent_text = sample["parent_text"]
                
                # Extract axes data and shuffle the order
                axes_data = sample.get("axes", {})
                if axes_data:
                    # Convert to list, shuffle, and convert back to dict
                    axes_items = list(axes_data.items())
                    random.shuffle(axes_items)
                    axes_data = dict(axes_items)
                
                # Extract targets - ensure the expected keys: xi, omega, alpha
                target_values = sample.get("target_values", {})
                xi = target_values.get("xi", target_values.get("location", 0.0))
                omega = target_values.get("omega", target_values.get("scale", 1.0))
                alpha = target_values.get("alpha", target_values.get("skewness", 0.0))
                targets = np.array([xi, omega, alpha], dtype=np.float32)
                
                return {
                    'texts': parent_text,
                    'axes_data': axes_data,
                    'targets': torch.tensor(targets, dtype=torch.float32)
                }
        
        return ShuffledChronoceptDataset(dataset.data, dataset.axis_encoding)
    
    def _evaluate_model(self, model, test_loader) -> Dict[str, float]:
        """Evaluate model and return metrics."""
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
        
        return evaluate_model_comprehensive(predictions, targets, "skew_normal")
    
    def run_all_ablations(self) -> Dict[str, Any]:
        """Run all ablation studies."""
        logger.info("Running all ablation studies...")
        
        results = {
            "axis_encoding": self.run_axis_encoding_ablation(),
            "objectives": self.run_objective_ablation(),
            "heads": self.run_head_ablation(),
            "pooling": self.run_pooling_ablation(),
            "training_stability": self.run_training_stability_ablation(),
            "axis_shuffling": self.run_axis_shuffling_ablation()
        }
        
        self.results = results
        self._save_ablation_results()
        return results
    
    def _save_ablation_results(self):
        """Save ablation results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.save_dir / f"ablation_results_{self.benchmark}_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for ablation_type, ablation_results in self.results.items():
            serializable_results[ablation_type] = {}
            for config_name, result in ablation_results.items():
                serializable_results[ablation_type][config_name] = {
                    "training_history": result.get("training_history"),
                    "metrics": result.get("metrics"),
                    "experiment_dir": result.get("experiment_dir"),
                    "error": result.get("error")
                }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Ablation results saved to {results_file}")
    
    def create_ablation_summary(self) -> str:
        """Create a summary of ablation results."""
        if not self.results:
            return "No ablation results available"
        
        summary = "\\n" + "="*120 + "\\n"
        summary += f"ABLATION STUDY RESULTS - {self.benchmark.upper()}\\n"
        summary += "="*120 + "\\n"
        
        for ablation_type, ablation_results in self.results.items():
            summary += f"\\n{ablation_type.upper().replace('_', ' ')} ABLATION:\\n"
            summary += "-" * 60 + "\\n"
            
            if ablation_type == "axis_encoding":
                summary += f"{'Encoding':<25} {'MSE':<10} {'MAE':<10} {'R²':<10} {'NLL':<10} {'CRPS':<10}\\n"
            elif ablation_type == "objectives":
                summary += f"{'Objective':<25} {'MSE':<10} {'MAE':<10} {'R²':<10} {'NLL':<10} {'CRPS':<10}\\n"
            elif ablation_type == "heads":
                summary += f"{'Head Type':<25} {'MSE':<10} {'MAE':<10} {'R²':<10} {'NLL':<10} {'CRPS':<10}\\n"
            elif ablation_type == "pooling":
                summary += f"{'Pooling':<25} {'MSE':<10} {'MAE':<10} {'R²':<10} {'NLL':<10} {'CRPS':<10}\\n"
            elif ablation_type == "training_stability":
                summary += f"{'Warm Start':<25} {'MSE':<10} {'MAE':<10} {'R²':<10} {'NLL':<10} {'CRPS':<10}\\n"
            elif ablation_type == "axis_shuffling":
                summary += f"{'Shuffling':<25} {'MSE':<10} {'MAE':<10} {'R²':<10} {'NLL':<10} {'CRPS':<10}\\n"
            
            summary += "-" * 60 + "\\n"
            
            for config_name, result in ablation_results.items():
                if result.get("error"):
                    summary += f"{config_name:<25} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}\\n"
                    continue
                
                metrics = result.get("metrics", {})
                summary += f"{config_name:<25} "
                summary += f"{metrics.get('mse', 'N/A'):<10.4f} "
                summary += f"{metrics.get('mae', 'N/A'):<10.4f} "
                summary += f"{metrics.get('r2', 'N/A'):<10.4f} "
                summary += f"{metrics.get('nll_skew', 'N/A'):<10.4f} "
                summary += f"{metrics.get('crps_skew', 'N/A'):<10.4f}\\n"
        
        summary += "="*120 + "\\n"
        return summary


def main():
    """Main function to run ablation studies."""
    parser = argparse.ArgumentParser(description="Run Chronocept ablation studies")
    parser.add_argument("--benchmark", choices=["benchmark_1", "benchmark_2"], 
                       default="benchmark_1", help="Which benchmark to run")
    parser.add_argument("--ablation", choices=["axis_encoding", "objectives", "heads", "pooling", "training_stability", "axis_shuffling", "all"],
                       default="all", help="Which single ablation study to run (use --ablations for multiple)")
    parser.add_argument("--ablations", nargs="+", choices=["axis_encoding", "objectives", "heads", "pooling", "training_stability", "axis_shuffling"],
                       help="Run multiple ablations in one invocation (space-separated)")
    parser.add_argument("--save_dir", default="ablation_results", help="Directory to save results")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Create ablation runner
    runner = AblationRunner(
        benchmark=args.benchmark,
        save_dir=args.save_dir,
        device=args.device
    )
    
    # Run ablations
    if args.ablations:
        results = {}
        for ab in args.ablations:
            if ab == "axis_encoding":
                results["axis_encoding"] = runner.run_axis_encoding_ablation()
            elif ab == "objectives":
                results["objectives"] = runner.run_objective_ablation()
            elif ab == "heads":
                results["heads"] = runner.run_head_ablation()
            elif ab == "pooling":
                results["pooling"] = runner.run_pooling_ablation()
            elif ab == "training_stability":
                results["training_stability"] = runner.run_training_stability_ablation()
            elif ab == "axis_shuffling":
                results["axis_shuffling"] = runner.run_axis_shuffling_ablation()
    else:
        if args.ablation == "all":
            results = runner.run_all_ablations()
        elif args.ablation == "axis_encoding":
            results = {"axis_encoding": runner.run_axis_encoding_ablation()}
        elif args.ablation == "objectives":
            results = {"objectives": runner.run_objective_ablation()}
        elif args.ablation == "heads":
            results = {"heads": runner.run_head_ablation()}
        elif args.ablation == "pooling":
            results = {"pooling": runner.run_pooling_ablation()}
        elif args.ablation == "training_stability":
            results = {"training_stability": runner.run_training_stability_ablation()}
        elif args.ablation == "axis_shuffling":
            results = {"axis_shuffling": runner.run_axis_shuffling_ablation()}
    
    runner.results = results
    runner._save_ablation_results()
    
    # Print summary
    print(runner.create_ablation_summary())
    
    logger.info("Ablation studies completed!")


if __name__ == "__main__":
    main()
