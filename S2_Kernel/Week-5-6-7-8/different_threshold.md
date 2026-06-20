## Threshold 5
- pooled summary

-unique values after threshold  5: 123

| Model | RMSE | MAE | R2 | Correlation | Mean Residual | N | N Total |
|-------|------|-----|----|-------------|---------------|---|---------|
| knn | 23.948 | 18.116 | 0.5651 | 0.7587 | -0.6858 | 1947 | 1947 |
| nusvr | 26.554 | 19.944 | 0.4653 | 0.6943 | 0.3685 | 1947 | 1947 |
| svr | 26.898 | 20.322 | 0.4513 | 0.6877 | 0.3829 | 1947 | 1947 |

## Threshold 16 (upper quartile)
 
- unique values after threshold  16: 44

- pooled summary 
| Model | RMSE | MAE | R2 | Correlation | Mean Residual | N | N Total |
|-------|------|-----|----|-------------|---------------|---|---------|
| knn | 23.753 | 18.042 | 0.6004 | 0.7811 | -0.3765 | 1309 | 1309 |
| nusvr | 26.844 | 19.945 | 0.4897 | 0.7130 | 1.5734 | 1309 | 1309 |
| svr | 27.412 | 20.624 | 0.4679 | 0.7005 | 1.7967 | 1309 | 1309 |

## Threshold 40

- unique values after threshold 40: 8

- pooled summary 
| Model | RMSE | MAE | R2 | Correlation | Mean Residual | N | N Total |
|-------|------|-----|----|-------------|---------------|---|---------|
| knn | 26.641 | 20.785 | 0.3584 | 0.6094 | 1.4017 | 538 | 538 |
| nusvr | 30.911 | 22.920 | 0.1362 | 0.4675 | 5.4310 | 538 | 538 |
| svr | 31.317 | 23.422 | 0.1134 | 0.4694 | 6.4136 | 538 | 538 |

## Threshold 13 (mean)
- unique values after threshold 13: 47

- pooled summary 
| Model | RMSE | MAE | R2 | Correlation | Mean Residual | N | N Total |
|-------|------|-----|----|-------------|---------------|---|---------|
| knn | 23.932 | 18.119 | 0.5986 | 0.7808 | -0.8851 | 1350 | 1350 |
| nusvr | 27.313 | 20.412 | 0.4772 | 0.7027 | 0.7119 | 1350 | 1350 |
| svr | 27.411 | 20.683 | 0.4734 | 0.7036 | 1.2919 | 1350 | 1350 |

## Threshold 8 (median)
- unique values after threshold 8: 85


- pooled summary 

| Model | RMSE | MAE | R2 | Correlation | Mean Residual | N | N Total |
|-------|------|-----|----|-------------|---------------|---|---------|
| knn | 23.674 | 17.863 | 0.5928 | 0.7748 | -0.9759 | 1729 | 1729 |
| nusvr | 26.668 | 19.903 | 0.4833 | 0.7048 | 0.5360 | 1729 | 1729 |
| svr | 27.244 | 20.624 | 0.4608 | 0.6923 | 0.6710 | 1729 | 1729 |


N is the number of samples (rows) in the dataset after applying the LOPO (Leave-One-Publication-Out) evaluation — i.e., how many test predictions were made across all folds combined.