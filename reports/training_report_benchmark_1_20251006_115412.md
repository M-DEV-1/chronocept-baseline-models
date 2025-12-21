# Chronocept Baseline Training Report

**Generated:** 2025-10-06 11:54:12  
**Benchmark:** benchmark_1  
**Models Trained:** 11

---

## Executive Summary

This report presents the training results for 11 baseline models on the Chronocept benchmark_1 dataset.

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
| deberta_mse | 151.0373 | 6.9770 | -0.0329 | -0.1261 | -0.1255 | 0.1766 | 220.4651 |
| deberta_param | 695.7505 | 14.8318 | -1.3404 | -0.1179 | 0.0318 | 0.1953 | 9.9973 |
| deberta_skew | 512.7388 | 13.1731 | -1.9905 | -0.1264 | -0.0763 | -0.1191 | 19.7282 |
| distilbert_param | 737.3157 | 15.3509 | -1.5092 | -0.0545 | 0.0763 | 0.0860 | 11.3089 |
| distilbert_skew | 730.7649 | 15.7800 | -2.8329 | -0.0248 | -0.0099 | -0.0324 | 14.8254 |
| mtdnn | 394.7480 | 14.3401 | -11.2308 | -0.1122 | -0.0578 | 0.0740 | 4.4120 |
| mtdnn_param | 108.3768 | 5.9425 | 0.0314 | 0.5200 | 0.1405 | 0.0115 | 4.5117 |
| roberta_mse | 654.4803 | 14.2539 | -1.2212 | -0.1373 | -0.0809 | 0.1188 | 975.6742 |
| roberta_param | 909.5181 | 17.2330 | -1.8687 | -0.1243 | -0.0371 | 0.0948 | 15.2214 |
| sbert_bilstm | 171.7044 | 8.5698 | -2.3193 | 0.3710 | 0.1396 | -0.0385 | 4.4084 |
| sbert_ffnn | 155.8747 | 7.9780 | -2.0203 | 0.4687 | 0.1670 | -0.0295 | 4.3769 |

### Per-Parameter RMSE

| Model | RMSE (ξ) | RMSE (ω) | RMSE (α) |
|-------|----------|----------|----------|
| deberta_mse | 20.9545 | 3.5140 | 1.2927 |
| deberta_param | 45.5250 | 3.6120 | 1.2950 |
| deberta_skew | 38.7160 | 5.9390 | 2.0034 |
| distilbert_param | 46.8732 | 3.5750 | 1.4375 |
| distilbert_skew | 46.5636 | 3.9724 | 2.8889 |
| mtdnn | 28.9076 | 18.4124 | 3.0951 |
| mtdnn_param | 17.6030 | 3.6852 | 1.2974 |
| roberta_mse | 44.1533 | 3.5003 | 1.2955 |
| roberta_param | 52.0884 | 3.6933 | 1.3079 |
| sbert_bilstm | 20.7577 | 8.9916 | 1.8386 |
| sbert_ffnn | 19.6840 | 8.8119 | 1.5859 |

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
- Final Train Loss: 142.6012
- Final Valid Loss: 159.7992
- Best Valid Loss: 158.6334 (epoch 91)

**Test Set Performance:**
- MSE: 151.0373
- MAE: 6.9770
- R²: -0.0329
- RMSE (ξ): 20.9545
- RMSE (ω): 3.5140
- RMSE (α): 1.2927
- Spearman (ξ): -0.1261
- Spearman (ω): -0.1255
- Spearman (α): 0.1766

**Training Curves:** See `deberta_mse_benchmark_1_20251005_160809\training_plots.pdf`

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
- Final Train Loss: 14.5166
- Final Valid Loss: 12.6492
- Best Valid Loss: 12.6492 (epoch 100)

**Test Set Performance:**
- MSE: 695.7505
- MAE: 14.8318
- R²: -1.3404
- RMSE (ξ): 45.5250
- RMSE (ω): 3.6120
- RMSE (α): 1.2950
- Spearman (ξ): -0.1179
- Spearman (ω): 0.0318
- Spearman (α): 0.1953

**Training Curves:** See `deberta_param_benchmark_1_20251006_014906\training_plots.pdf`

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
- Final Train Loss: 23.2669
- Final Valid Loss: 19.7145
- Best Valid Loss: 19.7145 (epoch 100)

**Test Set Performance:**
- MSE: 512.7388
- MAE: 13.1731
- R²: -1.9905
- RMSE (ξ): 38.7160
- RMSE (ω): 5.9390
- RMSE (α): 2.0034
- Spearman (ξ): -0.1264
- Spearman (ω): -0.0763
- Spearman (α): -0.1191

**Training Curves:** See `deberta_skew_benchmark_1_20251005_161717\training_plots.pdf`

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
- Final Train Loss: 17.1395
- Final Valid Loss: 16.1392
- Best Valid Loss: 16.1392 (epoch 100)

**Test Set Performance:**
- MSE: 737.3157
- MAE: 15.3509
- R²: -1.5092
- RMSE (ξ): 46.8732
- RMSE (ω): 3.5750
- RMSE (α): 1.4375
- Spearman (ξ): -0.0545
- Spearman (ω): 0.0763
- Spearman (α): 0.0860

**Training Curves:** See `distilbert_param_benchmark_1_20251006_031533\training_plots.pdf`

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
- Final Train Loss: 15.9828
- Final Valid Loss: 14.6209
- Best Valid Loss: 14.6209 (epoch 100)

**Test Set Performance:**
- MSE: 730.7649
- MAE: 15.7800
- R²: -2.8329
- RMSE (ξ): 46.5636
- RMSE (ω): 3.9724
- RMSE (α): 2.8889
- Spearman (ξ): -0.0248
- Spearman (ω): -0.0099
- Spearman (α): -0.0324

**Training Curves:** See `distilbert_skew_benchmark_1_20251005_163412\training_plots.pdf`

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
- Final Train Loss: 4.4285
- Final Valid Loss: 4.4939
- Best Valid Loss: 4.4932 (epoch 64)

**Test Set Performance:**
- MSE: 394.7480
- MAE: 14.3401
- R²: -11.2308
- RMSE (ξ): 28.9076
- RMSE (ω): 18.4124
- RMSE (α): 3.0951
- Spearman (ξ): -0.1122
- Spearman (ω): -0.0578
- Spearman (α): 0.0740

**Training Curves:** See `mtdnn_benchmark_1_20251005_164506\training_plots.pdf`

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
- Final Train Loss: 98.5632
- Final Valid Loss: 115.1962
- Best Valid Loss: 115.1962 (epoch 100)

**Test Set Performance:**
- MSE: 108.3768
- MAE: 5.9425
- R²: 0.0314
- RMSE (ξ): 17.6030
- RMSE (ω): 3.6852
- RMSE (α): 1.2974
- Spearman (ξ): 0.5200
- Spearman (ω): 0.1405
- Spearman (α): 0.0115

**Training Curves:** See `mtdnn_param_benchmark_1_20251006_111102\training_plots.pdf`

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
- Final Train Loss: 674.2443
- Final Valid Loss: 668.0521
- Best Valid Loss: 668.0521 (epoch 100)

**Test Set Performance:**
- MSE: 654.4803
- MAE: 14.2539
- R²: -1.2212
- RMSE (ξ): 44.1533
- RMSE (ω): 3.5003
- RMSE (α): 1.2955
- Spearman (ξ): -0.1373
- Spearman (ω): -0.0809
- Spearman (α): 0.1188

**Training Curves:** See `roberta_mse_benchmark_1_20251005_160133\training_plots.pdf`

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
- Final Train Loss: 57.7659
- Final Valid Loss: 53.8469
- Best Valid Loss: 53.8469 (epoch 100)

**Test Set Performance:**
- MSE: 909.5181
- MAE: 17.2330
- R²: -1.8687
- RMSE (ξ): 52.0884
- RMSE (ω): 3.6933
- RMSE (α): 1.3079
- Spearman (ξ): -0.1243
- Spearman (ω): -0.0371
- Spearman (α): 0.0948

**Training Curves:** See `roberta_param_benchmark_1_20251006_005449\training_plots.pdf`

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
- Final Train Loss: 4.1898
- Final Valid Loss: 4.4243
- Best Valid Loss: 4.3513 (epoch 73)

**Test Set Performance:**
- MSE: 171.7044
- MAE: 8.5698
- R²: -2.3193
- RMSE (ξ): 20.7577
- RMSE (ω): 8.9916
- RMSE (α): 1.8386
- Spearman (ξ): 0.3710
- Spearman (ω): 0.1396
- Spearman (α): -0.0385

**Training Curves:** See `sbert_bilstm_benchmark_1_20251005_160004\training_plots.pdf`

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
- Final Train Loss: 4.1978
- Final Valid Loss: 4.4054
- Best Valid Loss: 4.3607 (epoch 17)

**Test Set Performance:**
- MSE: 155.8747
- MAE: 7.9780
- R²: -2.0203
- RMSE (ξ): 19.6840
- RMSE (ω): 8.8119
- RMSE (α): 1.5859
- Spearman (ξ): 0.4687
- Spearman (ω): 0.1670
- Spearman (α): -0.0295

**Training Curves:** See `sbert_ffnn_benchmark_1_20251005_233655\training_plots.pdf`

---

