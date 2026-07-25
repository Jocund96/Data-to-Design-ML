"""Dataset loading, scaling, evaluation, and results-saving helpers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR, NuSVR
import seaborn as sns

def load_data():
    """Load the processed train/val/test feature and target CSVs."""
    X_train = pd.read_csv('../Datasets/processed/X_train.csv')
    X_val   = pd.read_csv('../Datasets/processed/X_val.csv')
    X_test  = pd.read_csv('../Datasets/processed/X_test.csv')
    y_train = pd.read_csv('../Datasets/processed/y_train.csv').squeeze()
    y_val   = pd.read_csv('../Datasets/processed/y_val.csv').squeeze()
    y_test  = pd.read_csv('../Datasets/processed/y_test.csv').squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test

    
    
def scale_data(X_train, X_val, X_test, data_scaler = 'RobustScaler'):
    """Fit a scaler on X_train and apply it to train/val/test sets."""
    if data_scaler == 'RobustScaler':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val   = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    print(X_train.describe().round(2))
    return X_train, X_val, X_test, scaler

def evaluate_model(model_name,y_train,y_train_predicted,y_test,y_test_predicted):
    """Compute and print MAE, RMSE, R, and R² for test predictions, returning them as a dict."""
    mae  = mean_absolute_error(y_test, y_test_predicted)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predicted))
    r2   = r2_score(y_test, y_test_predicted)
    r    = np.sqrt(abs(r2))
    n_samples = len(y_test)
    total_samples = len(y_train)+n_samples

    print(f"\n{model_name} Results:")
    print(f"Total Samples : {total_samples}")
    print(f"Test Samples  : {n_samples}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R        : {r:.4f}")
    print(f"R²       : {r2:.4f}")

    return {'model': model_name, 'Total Samples':total_samples, 'Test Samples': n_samples, 'MAE': mae, 'RMSE': rmse, 'R': r, 'R2': r2}

def save_results(results_dict, filename):
    """Save a results dict as a single-row CSV under ./results/."""
    pd.DataFrame([results_dict]).to_csv(f'./results/{filename}', index=False)
    print(f"Results saved to results/{filename}")







