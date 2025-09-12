"""
Training script for parameter-space Gaussian NLL models.
"""

import torch
import logging
from models_v2 import RoBERTaRegression, DeBERTaRegression, DistilBERTRegression
from utils_v2 import ImprovedDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(model_name: str, loss_type: str = "param_gauss", epochs: int = 5):
    """Train a model with the specified configuration."""
    
    logger.info(f"Training {model_name} with {loss_type} loss...")
    
    # Create data loader
    data_loader = ImprovedDataLoader(
        benchmark="benchmark_1",
        axis_encoding="single_sequence_markers",
        batch_size=8,  # Small batch for testing
        shuffle=True
    )
    
    train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
    
    # Create model
    if model_name == "roberta":
        model = RoBERTaRegression(
            model_name="roberta-base",
            pooling_type="mean",
            head_type="linear",
            axis_encoding="single_sequence_markers",
            loss_type=loss_type,
            dropout=0.1
        )
    elif model_name == "deberta":
        model = DeBERTaRegression(
            model_name="microsoft/deberta-v3-base",
            pooling_type="mean",
            head_type="linear",
            axis_encoding="single_sequence_markers",
            loss_type=loss_type,
            dropout=0.1
        )
    elif model_name == "distilbert":
        model = DistilBERTRegression(
            model_name="distilbert-base-uncased",
            pooling_type="mean",
            head_type="linear",
            axis_encoding="single_sequence_markers",
            loss_type=loss_type,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train model
    history = model.train_model(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=epochs,
        lr=2e-5,
        warm_start_epochs=2  # Warm-start for 2 epochs
    )
    
    # Evaluate model
    metrics = model.evaluate_model(test_loader)
    
    logger.info(f"Training completed for {model_name}")
    logger.info(f"Final metrics: {metrics}")
    
    return model, history, metrics

def main():
    """Run training for different models and loss types."""
    
    # Test different configurations
    configurations = [
        ("roberta", "param_gauss"),
        ("roberta", "mse"),  # Compare with MSE
        ("deberta", "param_gauss"),
        ("distilbert", "param_gauss")
    ]
    
    results = {}
    
    for model_name, loss_type in configurations:
        try:
            model, history, metrics = train_model(model_name, loss_type, epochs=3)
            results[f"{model_name}_{loss_type}"] = {
                "history": history,
                "metrics": metrics
            }
            logger.info(f"✅ {model_name}_{loss_type} completed successfully")
        except Exception as e:
            logger.error(f"❌ {model_name}_{loss_type} failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("-" * 60)
    
    # Print summary
    logger.info("Training Summary:")
    for config, result in results.items():
        if result:
            metrics = result["metrics"]
            logger.info(f"{config}:")
            logger.info(f"  MSE: {metrics.get('mse', 'N/A'):.4f}")
            logger.info(f"  Spearman xi: {metrics.get('spearman_xi', 'N/A'):.4f}")
            logger.info(f"  Spearman omega: {metrics.get('spearman_omega', 'N/A'):.4f}")
            logger.info(f"  Spearman alpha: {metrics.get('spearman_alpha', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
