### Ablation Study Results - Benchmark 2

#### Axis Encoding Ablation

| Encoding                | MSE      | MAE     | R²      | NLL         | CRPS    |
|-------------------------|----------|---------|---------|-------------|---------|
| no_axes                 | 844.9597 | 19.0196 | -8.5646 | 1637.6928   | 47.3738 |
| single_sequence_markers | 836.9846 | 18.9870 | -8.7325 | 2944.0318   | 47.1757 |
| sbert_concat            | 853.9321 | 18.9265 | -7.9329 | 568.3081    | 47.6832 |

#### Objectives Ablation

| Objective   | MSE      | MAE     | R²      | NLL       | CRPS    |
|-------------|----------|---------|---------|-----------|---------|
| mse         | 849.8824 | 19.0005 | -8.3296 | 1021.8701 | 47.7491 |
| gaussian    | 865.5526 | 19.3327 | -8.9882 | 3997.7248 | 48.3848 |
| skew_normal | 847.8174 | 19.1539 | -8.9701 | 3867.5659 | 47.3658 |

#### Pooling Ablation

| Pooling   | MSE      | MAE     | R²      | NLL       | CRPS    |
|-----------|----------|---------|---------|-----------|---------|
| pooler    | 862.3647 | 19.2215 | -8.7367 | 2063.9955 | 47.9558 |
| mean      | 857.5925 | 19.2147 | -8.8866 | 3384.4937 | 47.8308 |
| attention | 837.1380 | 18.9568 | -8.6075 | 1977.2097 | 47.1323 |

#### Axis Shuffling Ablation

| Shuffling     | MSE      | MAE     | R²      | NLL       | CRPS    |
|---------------|----------|---------|---------|-----------|---------|
| shuffle_False | 841.7615 | 18.9694 | -8.4962 | 1494.3527 | 47.3578 |
| shuffle_True  | 849.0772 | 19.1761 | -9.0324 | 6846.4384 | 47.5677 |

Notes:
- The `sbert_concat` configuration errored in this run; see `ablation_results/` logs for details.

