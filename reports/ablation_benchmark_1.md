### Ablation Study Results - Benchmark 1

#### Axis Encoding Ablation

| Encoding                  | MSE       | MAE     | R²      | NLL      | CRPS    |
|---------------------------|-----------|---------|---------|----------|---------|
| no_axes                   | 1110.8466 | 21.3611 | -5.1725 | 1023.1238 | 52.6468 |
| single_sequence_markers   | 1085.3527 | 21.1018 | -5.0833 | 928.8128  | 51.9435 |
| sbert_concat              | 1044.7926 | 20.0065 | -3.7696 | 95.5716  | 49.6696 |

#### Objectives Ablation

| Objective     | MSE       | MAE     | R²      | NLL      | CRPS    |
|---------------|-----------|---------|---------|----------|---------|
| mse           | 1091.9943 | 20.8734 | -4.5301 | 296.6341 | 52.5248 |
| gaussian      | 1085.7045 | 21.1044 | -5.0763 | 758.6294 | 51.6280 |
| skew_normal   | 1104.5408 | 21.2746 | -5.1060 | 844.8828 | 52.3723 |

#### Pooling Ablation

| Pooling    | MSE       | MAE     | R²      | NLL       | CRPS    |
|------------|-----------|---------|---------|-----------|---------|
| pooler     | 1124.9136 | 21.5665 | -5.3592 | 1469.2367 | 52.8958 |
| mean       | 1083.6992 | 21.0199 | -4.9446 | 655.3037  | 51.8844 |
| attention  | 1090.7451 | 21.1635 | -5.1119 | 868.3504  | 51.8900 |

Notes:
- The `sbert_concat` configuration errored out in this run; inspect logs under `ablation_results/` for details.

