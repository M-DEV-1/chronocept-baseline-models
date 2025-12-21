# Chronocept Baseline Training Report

**Generated:** 2025-10-06 12:24:46  
**Benchmark:** benchmark_2  
**Models Trained:** 11

---

## Executive Summary

This report presents the training results for 11 baseline models on the Chronocept benchmark_2 dataset.

### Models Evaluated

1. **deberta_mse**
   - Architecture: microsoft/deberta-v3-base
   - Loss: mse
   - Epochs: 100
   - Learning Rate: 1e-05

2. **deberta_param**
   - Architecture: microsoft/deberta-v3-base
   - Loss: param_gauss
   - Epochs: 100
   - Learning Rate: 1e-05

3. **deberta_skew**
   - Architecture: microsoft/deberta-v3-base
   - Loss: skew_normal
   - Epochs: 100
   - Learning Rate: 1e-05

4. **distilbert_param**
   - Architecture: distilbert-base-uncased
   - Loss: param_gauss
   - Epochs: 100
   - Learning Rate: 2e-05

5. **distilbert_skew**
   - Architecture: distilbert-base-uncased
   - Loss: skew_normal
   - Epochs: 100
   - Learning Rate: 2e-05

6. **mtdnn**
   - Architecture: roberta-base
   - Loss: skew_normal
   - Epochs: 100
   - Learning Rate: 1e-05

7. **mtdnn_param**
   - Architecture: roberta-base
   - Loss: param_gauss
   - Epochs: 100
   - Learning Rate: 1e-05

8. **roberta_mse**
   - Architecture: roberta-base
   - Loss: mse
   - Epochs: 100
   - Learning Rate: 1e-05

9. **roberta_param**
   - Architecture: roberta-base
   - Loss: param_gauss
   - Epochs: 100
   - Learning Rate: 1e-05

10. **sbert_bilstm**
   - Architecture: N/A
   - Loss: skew_normal
   - Epochs: 100
   - Learning Rate: 0.001

11. **sbert_ffnn**
   - Architecture: N/A
   - Loss: skew_normal
   - Epochs: 100
   - Learning Rate: 0.001


---

## Performance Comparison

### Test Set Metrics

| Model | MSE | MAE | R² | Spearman (ξ) | Spearman (ω) | Spearman (α) | NLL |
|-------|-----|-----|----|--------------|--------------|--------------|-----|
| deberta_mse | 182.3851 | 7.4118 | -0.6490 | -0.2178 | 0.1617 | 0.0159 | 270.9766 |
| deberta_param | 544.5860 | 13.7358 | -2.7730 | -0.2608 | 0.3994 | 0.0037 | 11.6107 |
| deberta_skew | 674.3253 | 15.7013 | -5.0860 | -0.1996 | 0.0321 | 0.0090 | 16.5712 |
| distilbert_param | 640.2664 | 14.7484 | -3.0546 | 0.0754 | 0.2373 | 0.2271 | 13.6647 |
| distilbert_skew | 659.3134 | 15.6195 | -4.1172 | 0.0591 | -0.0659 | 0.0643 | 32.4152 |
| mtdnn | 375.5049 | 15.9036 | -32.0211 | 0.0603 | -0.0872 | 0.1009 | 4.1012 |
| mtdnn_param | 64.5708 | 4.5253 | -0.0183 | 0.1807 | -0.0488 | 0.0042 | 4.1865 |
| roberta_mse | 658.4019 | 15.2268 | -3.5012 | 0.0750 | -0.0874 | 0.1210 | 981.8790 |
| roberta_param | 748.8589 | 16.0097 | -3.5666 | 0.0882 | 0.2670 | 0.1383 | 15.2253 |
| sbert_bilstm | 107.4771 | 6.0472 | -1.1621 | 0.1337 | -0.2033 | -0.0327 | 4.1373 |
| sbert_ffnn | 153.2178 | 8.5744 | -6.3268 | 0.3690 | -0.0511 | -0.1867 | 3.9842 |

### Per-Parameter RMSE

| Model | RMSE (ξ) | RMSE (ω) | RMSE (α) |
|-------|----------|----------|----------|
| deberta_mse | 23.2404 | 2.4431 | 1.0349 |
| deberta_param | 40.2941 | 2.9834 | 1.1133 |
| deberta_skew | 44.8280 | 2.5648 | 2.6169 |
| distilbert_param | 43.7430 | 2.5137 | 1.0161 |
| distilbert_skew | 44.2291 | 4.5040 | 1.2021 |
| mtdnn | 26.8896 | 19.3944 | 5.2273 |
| mtdnn_param | 13.6704 | 2.4017 | 1.0319 |
| roberta_mse | 44.2936 | 3.5016 | 1.0122 |
| roberta_param | 47.3300 | 2.3293 | 1.0111 |
| sbert_bilstm | 17.3793 | 4.3467 | 1.2238 |
| sbert_ffnn | 19.5139 | 8.4704 | 2.6672 |

---

## Detailed Model Results

### deberta_mse

**Configuration:**
- Model: microsoft/deberta-v3-base
- Pooling: mean
- Head: linear
- Loss: mse
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 1e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 175.2798
- Final Valid Loss: 118.1889
- Best Valid Loss: 118.1889 (epoch 100)

**Test Set Performance:**
- MSE: 182.3851
- MAE: 7.4118
- R²: -0.6490
- RMSE (ξ): 23.2404
- RMSE (ω): 2.4431
- RMSE (α): 1.0349
- Spearman (ξ): -0.2178
- Spearman (ω): 0.1617
- Spearman (α): 0.0159

**Training Curves:** See `deberta_mse_benchmark_2_20251005_172759\training_plots.pdf`

---

### deberta_param

**Configuration:**
- Model: microsoft/deberta-v3-base
- Pooling: mean
- Head: linear
- Loss: param_gauss
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 1e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 56.3431
- Final Valid Loss: 41.9915
- Best Valid Loss: 41.9915 (epoch 100)

**Test Set Performance:**
- MSE: 544.5860
- MAE: 13.7358
- R²: -2.7730
- RMSE (ξ): 40.2941
- RMSE (ω): 2.9834
- RMSE (α): 1.1133
- Spearman (ξ): -0.2608
- Spearman (ω): 0.3994
- Spearman (α): 0.0037

**Training Curves:** See `deberta_param_benchmark_2_20251006_120826\training_plots.pdf`

---

### deberta_skew

**Configuration:**
- Model: microsoft/deberta-v3-base
- Pooling: mean
- Head: linear
- Loss: skew_normal
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 1e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 17.4308
- Final Valid Loss: 13.6530
- Best Valid Loss: 13.6530 (epoch 100)

**Test Set Performance:**
- MSE: 674.3253
- MAE: 15.7013
- R²: -5.0860
- RMSE (ξ): 44.8280
- RMSE (ω): 2.5648
- RMSE (α): 2.6169
- Spearman (ξ): -0.1996
- Spearman (ω): 0.0321
- Spearman (α): 0.0090

**Training Curves:** See `deberta_skew_benchmark_2_20251005_173743\training_plots.pdf`

---

### distilbert_param

**Configuration:**
- Model: distilbert-base-uncased
- Pooling: mean
- Head: linear
- Loss: param_gauss
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 2e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 36.7599
- Final Valid Loss: 30.2636
- Best Valid Loss: 30.2636 (epoch 100)

**Test Set Performance:**
- MSE: 640.2664
- MAE: 14.7484
- R²: -3.0546
- RMSE (ξ): 43.7430
- RMSE (ω): 2.5137
- RMSE (α): 1.0161
- Spearman (ξ): 0.0754
- Spearman (ω): 0.2373
- Spearman (α): 0.2271

**Training Curves:** See `distilbert_param_benchmark_2_20251006_121808\training_plots.pdf`

---

### distilbert_skew

**Configuration:**
- Model: distilbert-base-uncased
- Pooling: mean
- Head: linear
- Loss: skew_normal
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 2e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 32.4434
- Final Valid Loss: 25.6388
- Best Valid Loss: 25.6388 (epoch 100)

**Test Set Performance:**
- MSE: 659.3134
- MAE: 15.6195
- R²: -4.1172
- RMSE (ξ): 44.2291
- RMSE (ω): 4.5040
- RMSE (α): 1.2021
- Spearman (ξ): 0.0591
- Spearman (ω): -0.0659
- Spearman (α): 0.0643

**Training Curves:** See `distilbert_skew_benchmark_2_20251005_174722\training_plots.pdf`

---

### mtdnn

**Configuration:**
- Model: roberta-base
- Pooling: mean
- Head: N/A
- Loss: skew_normal
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 1e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 4.0868
- Final Valid Loss: 4.0348
- Best Valid Loss: 4.0348 (epoch 100)

**Test Set Performance:**
- MSE: 375.5049
- MAE: 15.9036
- R²: -32.0211
- RMSE (ξ): 26.8896
- RMSE (ω): 19.3944
- RMSE (α): 5.2273
- Spearman (ξ): 0.0603
- Spearman (ω): -0.0872
- Spearman (α): 0.1009

**Training Curves:** See `mtdnn_benchmark_2_20251005_175144\training_plots.pdf`

---

### mtdnn_param

**Configuration:**
- Model: roberta-base
- Pooling: mean
- Head: N/A
- Loss: param_gauss
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 1e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 65.5217
- Final Valid Loss: 56.1831
- Best Valid Loss: 51.9539 (epoch 17)

**Test Set Performance:**
- MSE: 64.5708
- MAE: 4.5253
- R²: -0.0183
- RMSE (ξ): 13.6704
- RMSE (ω): 2.4017
- RMSE (α): 1.0319
- Spearman (ξ): 0.1807
- Spearman (ω): -0.0488
- Spearman (α): 0.0042

**Training Curves:** See `mtdnn_param_benchmark_2_20251006_122228\training_plots.pdf`

---

### roberta_mse

**Configuration:**
- Model: roberta-base
- Pooling: mean
- Head: linear
- Loss: mse
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 1e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 606.2531
- Final Valid Loss: 522.2026
- Best Valid Loss: 522.2026 (epoch 100)

**Test Set Performance:**
- MSE: 658.4019
- MAE: 15.2268
- R²: -3.5012
- RMSE (ξ): 44.2936
- RMSE (ω): 3.5016
- RMSE (α): 1.0122
- Spearman (ξ): 0.0750
- Spearman (ω): -0.0874
- Spearman (α): 0.1210

**Training Curves:** See `roberta_mse_benchmark_2_20251005_172115\training_plots.pdf`

---

### roberta_param

**Configuration:**
- Model: roberta-base
- Pooling: mean
- Head: linear
- Loss: param_gauss
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 1e-05
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 101.1957
- Final Valid Loss: 80.9483
- Best Valid Loss: 80.9483 (epoch 100)

**Test Set Performance:**
- MSE: 748.8589
- MAE: 16.0097
- R²: -3.5666
- RMSE (ξ): 47.3300
- RMSE (ω): 2.3293
- RMSE (α): 1.0111
- Spearman (ξ): 0.0882
- Spearman (ω): 0.2670
- Spearman (α): 0.1383

**Training Curves:** See `roberta_param_benchmark_2_20251006_120139\training_plots.pdf`

---

### sbert_bilstm

**Configuration:**
- Model: N/A
- Pooling: N/A
- Head: N/A
- Loss: skew_normal
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 0.001
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 4.0911
- Final Valid Loss: 3.9003
- Best Valid Loss: 3.9003 (epoch 100)

**Test Set Performance:**
- MSE: 107.4771
- MAE: 6.0472
- R²: -1.1621
- RMSE (ξ): 17.3793
- RMSE (ω): 4.3467
- RMSE (α): 1.2238
- Spearman (ξ): 0.1337
- Spearman (ω): -0.2033
- Spearman (α): -0.0327

**Training Curves:** See `sbert_bilstm_benchmark_2_20251005_171950\training_plots.pdf`

---

### sbert_ffnn

**Configuration:**
- Model: N/A
- Pooling: N/A
- Head: N/A
- Loss: skew_normal
- Dropout: 0.1
- Epochs: 100
- Learning Rate: 0.001
- Warm Start: 5 epochs

**Training Results:**
- Final Train Loss: 3.8340
- Final Valid Loss: 3.9668
- Best Valid Loss: 3.9546 (epoch 19)

**Test Set Performance:**
- MSE: 153.2178
- MAE: 8.5744
- R²: -6.3268
- RMSE (ξ): 19.5139
- RMSE (ω): 8.4704
- RMSE (α): 2.6672
- Spearman (ξ): 0.3690
- Spearman (ω): -0.0511
- Spearman (α): -0.1867

**Training Curves:** See `sbert_ffnn_benchmark_2_20251005_234355\training_plots.pdf`

---

