# Chronocept Improved Baseline Models

> **Implementation of review feedback for Chronocept baseline models**  
> **Authors:** Implementation based on review feedback from the paper submission

This repository contains the improved baseline implementations for Chronocept, addressing the critical issues identified in the review feedback. The models now implement proper distributional regression, modern encoders, and state-of-the-art training techniques.

## What Was Fixed

### ❌ Previous Issues (Addressed)

1. **Naïve BERT regression with 9-pass concatenation**
   - **Problem**: 9 separate forward passes, concatenated pooled outputs (9×768 → 6912)
   - **Issues**: Compute waste (9× slower), over-parameterized head, no axis interaction, weak pooling
   - **Solution**: Single-pass encoding with axis markers, proper pooling strategies

2. **Wrong loss function (MSE on ξ, ω, α)**
   - **Problem**: MSE is not a proper scoring rule for distributional regression
   - **Issues**: Meaningless Euclidean distance for ω (scale) and α (shape), model learns ξ and collapses ω, α
   - **Solution**: Parameter-space Gaussian NLL with proper transformations

3. **Outdated baselines**
   - **Problem**: Simple linear models, vanilla BERT with MSE
   - **Solution**: Modern encoders (RoBERTa, DeBERTa, DistilBERT), proper distributional heads

### ✅ New Implementation

- **Single-pass encoding** with axis markers instead of 9-pass concatenation
- **Proper pooling** (mean, attention) instead of pooler_output
- **Parameter-space regression** with per-parameter Gaussian NLL loss
- **Modern encoders** (RoBERTa, DeBERTa, DistilBERT)
- **Training stability** with warm-start and layer-wise LR decay
- **Comprehensive metrics** (NLL, CRPS, per-param RMSE, Spearman ρ)

## New Baseline Suite

### 1. **SBERT + FFNN** & **SBERT + BiLSTM** (SoTA Embedding Baselines)
- **Description**: Cheap embedding baseline using Sentence-BERT
- **Architecture**: SBERT encoder + FFNN/BiLSTM head
- **Loss**: Skew-normal NLL
- **Use case**: Lowest compute requiring baseline

### 2. **RoBERTa-base + Linear Head (MSE)**
- **Description**: Standard regression baseline
- **Architecture**: RoBERTa encoder + linear head
- **Loss**: MSE (legacy comparison)
- **Pooling**: Mean pooling

### 3. **DeBERTa-V3-base + Linear Head (MSE)**
- **Description**: Strong encoder baseline
- **Architecture**: DeBERTa-v3 encoder + linear head
- **Loss**: MSE
- **Pooling**: Mean pooling

### 4. **DeBERTa-V3-base + Parameter-space Gaussian NLL Head**
- **Description**: Principled baseline with proper parameter-space regression
- **Architecture**: DeBERTa-v3 encoder + 6-output head (means + log-stds)
- **Loss**: Per-parameter Gaussian NLL with transformations
- **Pooling**: Mean pooling

### 5. **DistilBERT + Parameter-space Gaussian NLL Head**
- **Description**: Efficiency baseline
- **Architecture**: DistilBERT encoder + 6-output head (means + log-stds)
- **Loss**: Per-parameter Gaussian NLL with transformations
- **Pooling**: Mean pooling

### 6. **MT-DNN Style Multi-task Learning**
- **Description**: Multi-task learning with shared encoder + 3 heads
- **Architecture**: RoBERTa encoder + multi-task head
- **Loss**: Skew-normal NLL + auxiliary losses
- **Pooling**: Mean pooling

### 7. **Legacy 9-pass Concatenation (Negative Ablation)**
- **Description**: Original problematic approach for comparison
- **Architecture**: BERT with 9 separate forward passes
- **Loss**: MSE
- **Pooling**: Pooler output only

## Ablation Studies

### Axis Encoding
- `NoAxes`: Parent text only
- `SingleSeq_Markers`: One sequence with `[AXIS_i]` markers (default)
- `SBERT_Concat`: SBERT embeddings per axis, concatenated
- `9-pass_Concat`: Old method (negative ablation)

### Objectives
- `MSE`: Legacy loss function
- `param_gauss`: Per-parameter Gaussian NLL with transformations (preferred)
- `NLL_skew`: Skew-normal negative log-likelihood (legacy, problematic)
- `NLL_gauss`: Gaussian negative log-likelihood (baseline)

### Heads
- `LinearShared`: Single linear layer → 3 parameters
- `FFNN_shared`: Hidden layer before outputs
- `Shared_trunk_+_3_heads`: Small trunk → 3 separate heads

### Pooling
- `pooler_output`: BERT's pooler output (baseline)
- `mean_pool`: Mean pooling over sequence (recommended)
- `attention_pool`: Learned attention pooling

### Training Stability
- **Warm-start**: Train only ξ for 2-3 epochs, then full NLL
- **Layer-wise LR decay**: Head LR = 5× encoder LR

## Loss Function: Why Parameter-Space Gaussian NLL?

### The Problem with Traditional Approaches

#### ❌ **MSE Loss Issues**
- **Scale Mismatch**: Parameters have vastly different scales and meanings
  - ξ (location): Small changes are often acceptable
  - ω (scale): Must be positive, small changes have huge distributional impact
  - α (shape): Bounded, small changes dramatically affect distribution shape
- **Result**: Model learns ξ, collapses ω→0, ignores α

#### ❌ **Skew-Normal NLL on Labels (Previous Approach)**
- **Fundamental Issue**: Treats parameter labels as if they were observed data samples
- **What it does**: Computes `log p_θ(labels)` where `labels = [ξ_true, ω_true, α_true]`
- **Why it's wrong**: Labels are the parameters themselves, not samples from the distribution
- **Result**: Encourages ω→0 (to maximize density at ξ_true), poor parameter correlations

### ✅ **Parameter-Space Gaussian NLL Solution**

#### **Core Idea**
Treat each parameter label as a noisy measurement of the true parameter, model the measurement noise explicitly.

#### **Transformations for Proper Geometry**
- **ξ (location)**: Direct prediction in R
- **ω (scale)**: Predict in log-space (log ω) to handle multiplicative errors
- **α (shape)**: Predict via bounded transform (artanh(α/A)) to avoid extreme values

#### **Model Output**
6-dimensional head: `[μ_ξ, σ_ξ, μ_logω, σ_logω, μ_α̃, σ_α̃]`
- Each parameter gets a mean and uncertainty estimate
- Heteroscedastic weighting: residuals normalized by learned uncertainty

#### **Loss Function**
```
L = NLL(ξ_true | N(μ_ξ, σ_ξ²)) + NLL(log ω_true | N(μ_logω, σ_logω²)) + NLL(α̃_true | N(μ_α̃, σ_α̃²))
```

#### **Why This Works**
1. **Proper likelihood**: Maximizes p(labels | inputs) in parameter space
2. **Scale-aware**: Log-space for ω handles multiplicative errors correctly
3. **Bounded α**: Tanh transform prevents extreme shape parameters
4. **Uncertainty learning**: Model learns how uncertain each parameter is
5. **No collapse**: Each parameter optimized on its natural scale

#### **Expected Results**
- Strong positive correlations for all parameters (ξ, ω, α)
- No parameter collapse or scale issues
- Meaningful uncertainty estimates
- Stable training dynamics

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd chronocept-baseline-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running All Baselines

```bash
# Run all baselines on benchmark_1
python benchmark_v2.py --benchmark benchmark_1

# Run specific models
python benchmark_v2.py --benchmark benchmark_1 --models sbert_ffnn roberta_mse deberta_skew

# Run on benchmark_2
python benchmark_v2.py --benchmark benchmark_2
```

### Running Ablation Studies

```bash
# Run all ablation studies
python ablation_studies_v2.py --benchmark benchmark_1

# Run specific ablation
python ablation_studies_v2.py --benchmark benchmark_1 --ablation axis_encoding
python ablation_studies_v2.py --benchmark benchmark_1 --ablation objectives
python ablation_studies_v2.py --benchmark benchmark_1 --ablation pooling
```

### Using Individual Models

#### **Parameter-Space Gaussian NLL (Recommended)**
```python
from models_v2 import RoBERTaRegression, DeBERTaRegression, DistilBERTRegression
from utils_v2 import ImprovedDataLoader

# Create data loader
data_loader = ImprovedDataLoader(
    benchmark="benchmark_1",
    axis_encoding="single_sequence_markers",
    batch_size=16
)

train_loader, valid_loader, test_loader = data_loader.get_data_loaders()

# Create model with parameter-space Gaussian NLL
model = RoBERTaRegression(
    model_name="roberta-base",
    pooling_type="mean",
    head_type="linear",
    axis_encoding="single_sequence_markers",
    loss_type="param_gauss"  # Use new parameter-space Gaussian NLL
)

# Train model with warm-start
history = model.train_model(
    train_loader=train_loader, 
    valid_loader=valid_loader, 
    epochs=10, 
    lr=2e-5,
    warm_start_epochs=2  # Train xi-only for 2 epochs
)

# Evaluate model
metrics = model.evaluate_model(test_loader)
print(f"Spearman correlations: ξ={metrics['spearman_xi']:.3f}, ω={metrics['spearman_omega']:.3f}, α={metrics['spearman_alpha']:.3f}")
```

#### **Legacy Approaches (For Comparison)**
```python
# MSE baseline
model_mse = RoBERTaRegression(
    model_name="roberta-base",
    loss_type="mse"  # Legacy MSE
)

# Skew-normal NLL (problematic)
model_skew = RoBERTaRegression(
    model_name="roberta-base", 
    loss_type="skew_normal"  # Misapplied NLL
)
```

## Metrics

### Primary Metrics
- **NLL**: Negative Log-Likelihood (parameter-space Gaussian)
- **CRPS**: Continuous Ranked Probability Score
- **Per-param RMSE**: RMSE for each parameter (ξ, ω, α)
- **Spearman ρ**: Spearman correlation for each parameter
- **Parameter correlations**: Individual correlations for ξ, ω, α

### Legacy Metrics (for comparison)
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: R-squared

## Architecture Details

### Model Structure
```
Input Text + Axes → Encoder → Pooling → Head → Predictions (ξ, ω, α)
```

### Key Improvements
1. **Single-pass encoding**: All text processed in one forward pass
2. **Proper pooling**: Mean or attention pooling instead of pooler_output
3. **Parameter-space regression**: Predict parameter means and uncertainties
4. **Proper loss functions**: Parameter-space Gaussian NLL with transformations
5. **Training stability**: Warm-start and layer-wise learning rates
6. **Scale-aware optimization**: Handles different parameter scales correctly

## Results Structure

Results are saved in organized directories:
```
benchmark_results/
├── sbert_ffnn_benchmark_1_20241201_120000/
│   ├── best_model.pt
│   ├── training_plots.png
│   ├── config.json
│   ├── training_results.json
│   └── evaluation_results.json
├── roberta_mse_benchmark_1_20241201_120500/
└── ...
```

## Comparison with Original Baselines

| Aspect | Original | Improved |
|--------|----------|----------|
| **Encoding** | 9-pass concatenation | Single-pass with markers |
| **Pooling** | Pooler output only | Mean/attention pooling |
| **Loss** | MSE on all parameters | Parameter-space Gaussian NLL |
| **Encoders** | BERT only | RoBERTa, DeBERTa, DistilBERT |
| **Training** | Basic training | Warm-start + layer-wise LR |
| **Metrics** | Basic regression | Parameter-space + correlation |
| **Scale handling** | Uniform MSE | Transform-aware optimization |

## Citation

If you use these improved baselines in your work, please cite:

```bibtex
@misc{goel2025chronocept,
    title={Chronocept: Instilling a Sense of Time in Machines}, 
    author={Krish Goel and Sanskar Pandey and KS Mahadevan and Harsh Kumar and Vish Khadaria},
    year={2025},
    eprint={2505.07637},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.07637}, 
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
