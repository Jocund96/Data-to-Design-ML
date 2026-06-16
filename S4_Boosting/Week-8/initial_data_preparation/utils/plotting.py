import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_rmse_vs_k(k_range : list, kfold_rmse_list: list , best_k_kfold:int):
    
    plt.plot(k_range, kfold_rmse_list,  label='3-Fold CV',        marker='o')

    plt.axvline(x=best_k_kfold, color='red',  linestyle='--', label=f'Best k (KFold)={best_k_kfold}')
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.title('RMSE vs K')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    

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
    


def plot_feature_vs_strength(df, feature_col, strength_col):
    feature = df[feature_col]
    strength = df[strength_col]
    correlation = feature.corr(strength)

    z = np.polyfit(feature, strength, 1)
    p = np.poly1d(z)
    x_line = np.linspace(feature.min(), feature.max(), 200)

    plt.figure(figsize=(8, 5))
    plt.scatter(feature, strength, alpha=0.4, color="steelblue", edgecolors="k", linewidths=0.3)
    plt.plot(x_line, p(x_line), color="red", linewidth=2, label="Trend line")
    plt.xlabel(feature_col)
    plt.ylabel(strength_col)
    plt.title(f"{feature_col} vs {strength_col}\nPearson r = {correlation:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Pearson correlation: {correlation:.4f}")
    
    
    
def plot_correlation_matrix(df, figsize=(10, 8), cmap="coolwarm", title="Feature Correlation Matrix"):
    corr_matrix = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()