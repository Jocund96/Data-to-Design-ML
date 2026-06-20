import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder,StandardScaler


def prepare_data(df, target_col='cs_28d', train_size=0.70, random_state=42):
    """Split a dataframe into train/val/test sets for the given target column."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - train_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def identify_column_types(X):
    """Split columns into numerical, one-hot (low cardinality) and k-fold/target-encoded (high cardinality)."""
    str_cols = X.select_dtypes(include='object').columns
    one_hot_columns = str_cols[X[str_cols].nunique() <= 10].tolist()
    k_fold_columns = str_cols[X[str_cols].nunique() > 10].tolist()
    numerical_cols = X.select_dtypes(include='number').columns.tolist()

    return numerical_cols, one_hot_columns, k_fold_columns


def create_preprocessor(numerical_cols, one_hot_columns, k_fold_columns,
                         handle_unknown='ignore'):
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ohe', OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False), one_hot_columns),
        ('target', TargetEncoder(cv=5), k_fold_columns)
    ])

    n_target = len(k_fold_columns)  # to solve not proper scaling of target encoding columns
    target_scaler = ColumnTransformer([
        ('scale_target', StandardScaler(), list(range(-n_target, 0)))
    ], remainder='passthrough')

    return Pipeline([
        ('preprocessor', preprocessor),
        ('scale_target_encoded', target_scaler)
    ])


def evaluate_model(y_true, y_pred, set_name=''):
    """Compute RMSE, MAE, R2, Pearson correlation, residual direction and n, optionally printing a labeled report."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Correlation': r,
        'Mean_Residual': residuals.mean(),  # +ve = under-prediction, -ve = over-prediction
        'N': len(y_true)
    }

    if set_name:
        print(f"\n{set_name.upper()} SET PERFORMANCE")
        print("=" * 60)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}" if metric_name != 'N' else f"{metric_name}: {value}")

    return metrics


def run_grid_search(pipeline, param_grid, X_train, X_val, X_test, y_train, y_val, y_test, model_label):
    """Run GridSearchCV over a pipeline, printing best params and val/test metrics."""
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"\nGridSearchCV will test {total} combinations for {model_label}...")

    gs = GridSearchCV(pipeline, param_grid, cv=5,
                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    print("\n" + "=" * 80)
    print(f"BEST HYPERPARAMETERS — {model_label}")
    print("=" * 80)
    for param, value in gs.best_params_.items():
        print(f"  {param.replace('model__', '')}: {value}")
    print(f"Best CV RMSE: {np.sqrt(-gs.best_score_):.4f}")

    val_metrics = evaluate_model(y_val, gs.predict(X_val), 'Validation')
    test_metrics = evaluate_model(y_test, gs.predict(X_test), 'Test')

    summary = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2', 'Correlation (R)','Mean Residual','N'],
        'Validation': list(val_metrics.values()),
        'Test': list(test_metrics.values())
    })
    print("\nRESULTS SUMMARY\n", summary.to_string(index=False))

    cv_df = pd.DataFrame(gs.cv_results_).sort_values('mean_test_score', ascending=False)
    param_cols = [c for c in cv_df.columns if c.startswith('param_model__')]
    top10 = cv_df[param_cols + ['mean_test_score']].head(10).copy()
    top10['CV_rmse'] = np.sqrt(-top10['mean_test_score'].values)
    top10.columns = [c.replace('param_model__', '') for c in top10.columns]
    print(f"\nTOP 10 COMBINATIONS\n{top10.to_string(index=False)}")

    return gs
