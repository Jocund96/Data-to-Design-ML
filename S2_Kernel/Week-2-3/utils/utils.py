import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data():
    X_train = pd.read_csv('../Datasets/processed/X_train.csv')
    X_val   = pd.read_csv('../Datasets/processed/X_val.csv')
    X_test  = pd.read_csv('../Datasets/processed/X_test.csv')
    y_train = pd.read_csv('../Datasets/processed/y_train.csv').squeeze()
    y_val   = pd.read_csv('../Datasets/processed/y_val.csv').squeeze()
    y_test  = pd.read_csv('../Datasets/processed/y_test.csv').squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test

    
    
def scale_data(X_train, X_val, X_test, data_scaler = 'RobustScaler'):
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
    pd.DataFrame([results_dict]).to_csv(f'./results/{filename}', index=False)
    print(f"Results saved to results/{filename}")


def plot_residuals(model_name, y_test, y_pred):
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True)

    axes[1].scatter(y_test, y_pred, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 color='red', linestyle='--', label='Perfect fit')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].hist(residuals, bins=30, edgecolor='black')
    axes[2].axvline(x=0, color='red', linestyle='--')
    axes[2].set_xlabel('Residual')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    axes[2].grid(True)

    plt.suptitle(f'{model_name} Residual Analysis')
    plt.tight_layout()
    plt.show()


def load_and_prepare_kfold_data(filepath, test_size=0.2, random_state=42):
    """
    Load data from CSV, clean it, rename columns, and split for k-fold validation.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    X_train_kf, X_test_kf, y_train_kf, y_test_kf : pandas DataFrames/Series
        Split training and testing data
    """
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(filepath)
    df_clean = df.dropna()
    df_clean = df_clean.drop_duplicates()
    
    df_clean.columns = [
        "cement", "slag", "fly_ash", "water",
        "superplasticizer", "coarse_agg",
        "fine_agg", "age", "strength"
    ]
    
    X = df_clean.drop("strength", axis=1)
    y = df_clean["strength"]
    
    X_train_kf, X_test_kf, y_train_kf, y_test_kf = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train_kf, X_test_kf, y_train_kf, y_test_kf