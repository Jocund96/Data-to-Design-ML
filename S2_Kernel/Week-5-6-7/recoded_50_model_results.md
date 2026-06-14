## Iteration 1 — StandardScaler on numeric cols only (target-encoded cols left unscaled)

| Model | Best Hyperparameters | CV RMSE | Val RMSE | Val MAE | Val R² | Val Corr | Test RMSE | Test MAE | Test R² | Test Corr |
|-------|----------------------|---------|----------|---------|--------|----------|-----------|----------|---------|-----------|
| KNN   | metric=manhattan, n_neighbors=4, p=2, weights=distance | 17.5251 | 16.9975 | 12.2978 | 0.7970 | 0.8943 | 15.6126 | 11.5335 | 0.7916 | 0.8900 |
| SVR   | C=512, epsilon=1, gamma=0.01 | 15.5465 | 15.8285 | 10.9051 | 0.8240 | 0.9112 | 17.6648 | 11.6142 | 0.7332 | 0.8584 |
| NuSVR | C=1024, gamma=0.01, nu=0.3 | 15.8566 | 15.8388 | 10.7150 | 0.8237 | 0.9106 | 18.2938 | 11.6521 | 0.7139 | 0.8489 |

## Iteration 2 — Global StandardScaler after ColumnTransformer (incl. OHE + target-encoded cols)

| Model | Best Hyperparameters | CV RMSE | Val RMSE | Val MAE | Val R² | Val Corr | Test RMSE | Test MAE | Test R² | Test Corr |
|-------|----------------------|---------|----------|---------|--------|----------|-----------|----------|---------|-----------|
| KNN   | metric=manhattan, n_neighbors=4, p=3, weights=distance | 17.5132 | 16.9201 | 12.3566 | 0.7988 | 0.8945 | 16.1910 | 11.4788 | 0.7759 | 0.8814 |
| SVR   | C=1024, epsilon=1, gamma=0.01 | 15.7543 | 16.9175 | 11.0464 | 0.7989 | 0.8984 | 18.8676 | 11.8200 | 0.6956 | 0.8406 |
| NuSVR | C=1024, gamma=0.01, nu=0.8 | 15.7117 | 17.7640 | 11.8322 | 0.7783 | 0.8938 | 18.5393 | 11.7141 | 0.7061 | 0.8498 |

## Iteration 3 — Targeted scaling: StandardScaler on numeric cols + target-encoded cols only (OHE left as 0/1), gamma re-tuned with extended C grid

| Model | Best Hyperparameters | CV RMSE | Val RMSE | Val MAE | Val R² | Val Corr | Test RMSE | Test MAE | Test R² | Test Corr |
|-------|----------------------|---------|----------|---------|--------|----------|-----------|----------|---------|-----------|
| KNN   | metric=minkowski, n_neighbors=3, p=1, weights=distance | 15.9142 | 15.9252 | 11.3549 | 0.8218 | 0.9105 | 16.0832 | 11.1342 | 0.7788 | 0.8838 |
| SVR   | C=1024, epsilon=3, gamma=scale | 14.7073 | 14.1309 | 9.7122 | 0.8597 | 0.9300 | 13.1096 | 9.2133 | 0.8531 | 0.9247 |
| NuSVR | C=1024, gamma=scale, nu=0.4 | 14.6345 | 14.3086 | 9.7774 | 0.8561 | 0.9278 | 13.2144 | 9.1518 | 0.8507 | 0.9230 |

