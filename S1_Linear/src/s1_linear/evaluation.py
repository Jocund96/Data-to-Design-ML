import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Calculate MAE, RMSE, R, and R²."""
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R": float(np.sqrt(r2)) if r2 >= 0 else np.nan,
        "R2": r2,
    }


def evaluate_model(model_name: str, model, X_test, y_test) -> dict:
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    return {"Model": model_name, **metrics}


def make_predictions_frame(y_true, y_pred, model_name: str) -> pd.DataFrame:
    """Create a dataframe with actual values, predictions, and residuals."""
    return pd.DataFrame({
        "Model": model_name,
        "Actual": y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true,
        "Predicted": y_pred,
        "Residual": y_true - y_pred,
    })
