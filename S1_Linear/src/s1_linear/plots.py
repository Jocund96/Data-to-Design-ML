from pathlib import Path
import matplotlib.pyplot as plt


def _prepare_save_path(save_path):
    if save_path is None:
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, save_path=None) -> None:
    """Plot actual strength against predicted strength."""
    save_path = _prepare_save_path(save_path)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(y_true, y_pred, model_name: str, save_path=None) -> None:
    """Plot residuals against predicted strength."""
    save_path = _prepare_save_path(save_path)
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Strength (MPa)")
    plt.ylabel("Residual: Actual - Predicted")
    plt.title(f"Residual Plot: {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals_vs_age(X_test, y_true, y_pred, model_name: str, save_path=None) -> None:
    """Plot residuals against concrete age when the Age column exists."""
    if "Age" not in X_test.columns:
        return

    save_path = _prepare_save_path(save_path)
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(X_test["Age"], residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Age (days)")
    plt.ylabel("Residual: Actual - Predicted")
    plt.title(f"Residuals vs Age: {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
