from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a local CSV dataset. The dataset itself is not committed to Git."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place local CSV in S1_Linear/data/processed/ or update configs/week03_linear.yaml."
        )
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def split_features_target(df: pd.DataFrame, features: list[str], target: str):
    """Create X and y using only selected features."""
    validate_columns(df, features + [target])
    X = df[features].copy()
    y = df[target].copy()
    return X, y


def make_train_test_split(X, y, test_size: float, random_state: int):
    """Create a reproducible 80/20-style train-test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
