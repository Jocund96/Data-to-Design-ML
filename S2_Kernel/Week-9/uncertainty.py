import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit

from helpers.modeling import evaluate_model, identify_column_types, create_preprocessor


def conformal_quantile(pipeline, X_cal, y_cal, alpha=0.90):
    """Absolute-residual quantile on a calibration set."""
    residuals = np.abs(np.asarray(y_cal) - pipeline.predict(X_cal))
    return float(np.quantile(residuals, alpha))


def build_intervals(pipeline, X_test, y_test, pub_test, q, X_train, n_neighbors=5):
    """Per-row prediction interval DataFrame with distance-to-training signal."""
    test_preds = pipeline.predict(X_test)
    y_true = np.asarray(y_test)

    if hasattr(pipeline.named_steps["model"], "kneighbors"):
        dists, _ = pipeline.named_steps["model"].kneighbors(
            pipeline[:-1].transform(X_test)
        )
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(pipeline[:-1].transform(X_train))
        dists, _ = nn.kneighbors(pipeline[:-1].transform(X_test))

    pub_values = pub_test.values if hasattr(pub_test, "values") else list(pub_test)
    return pd.DataFrame({
        "publication":   pub_values,
        "prediction":    test_preds,
        "true":          y_true,
        "lower":         test_preds - q,
        "upper":         test_preds + q,
        "residual":      np.abs(y_true - test_preds),
        "mean_distance": dists.mean(axis=1),
    })


def calibration_table(intervals):
    """Aggregate per-row intervals into a per-publication calibration table."""
    return intervals.groupby("publication").agg(
        n_rows          = ("true",  "count"),
        mean_true       = ("true",  "mean"),
        mean_prediction = ("prediction", "mean"),
        coverage        = ("true",  lambda x: np.mean(
            (x.values >= intervals.loc[x.index, "lower"].values) &
            (x.values <= intervals.loc[x.index, "upper"].values)
        )),
        interval_width  = ("upper", lambda x: (
            x.values - intervals.loc[x.index, "lower"].values
        ).mean()),
        mean_residual   = ("residual",     "mean"),
        mean_distance   = ("mean_distance","mean"),
    ).reset_index()


def plot_interval_band(intervals, title="Prediction Intervals", ax=None):
    """Sort test points by predicted value; shade the conformal band; colour dots by coverage."""
    import matplotlib.pyplot as plt

    df = intervals.sort_values("prediction").reset_index(drop=True)
    covered = (df["true"] >= df["lower"]) & (df["true"] <= df["upper"])
    coverage = covered.mean()
    x = df["prediction"].values

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(x, df["lower"].values, df["upper"].values,
                    alpha=0.20, color="steelblue", label="Interval [ŷ ± q]")
    ax.plot(x, x, color="steelblue", linewidth=1.2, linestyle="--", label="Predicted")
    ax.scatter(x[covered.values], df.loc[covered, "true"].values,
               color="seagreen", s=20, zorder=3, label="True (in)")
    ax.scatter(x[~covered.values], df.loc[~covered, "true"].values,
               color="tomato", s=20, zorder=3, marker="x", linewidths=1.2, label="True (out)")

    ax.set_xlabel("Predicted compressive strength (MPa)", fontsize=10)
    ax.set_ylabel("Compressive strength (MPa)", fontsize=10)
    ax.set_title(f"{title}  —  coverage {coverage:.1%}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    if own_fig:
        plt.tight_layout()
        plt.show()


def run_conformal(model_cls, model_key, results, param_key, preprocessor,
                  X_train, y_train, X_val, y_val, X_test, y_test, pub_test,
                  kernel_kwargs=None, alpha=0.90, n_neighbors=5):
    """Fit model, compute split-conformal intervals, return pipeline + metrics + calibration table."""
    params = results["best_params"][param_key][model_key]
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model",        model_cls(**(kernel_kwargs or {}))),
    ])
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate_model(y_train, pipeline.predict(X_train))
    test_metrics  = evaluate_model(y_test,  pipeline.predict(X_test))

    q         = conformal_quantile(pipeline, X_val, y_val, alpha)
    intervals = build_intervals(pipeline, X_test, y_test, pub_test, q, X_train, n_neighbors)
    cal_table = calibration_table(intervals)

    return pipeline, train_metrics, test_metrics, q, intervals, cal_table


def prepare_data_group(df, target_col, group_col="paper_reference",
                       test_size=0.15, random_state=42):
    """Publication-aware 70/15/15 split — no publication bleeds across splits."""
    X      = df.drop(columns=[target_col, group_col])
    y      = df[target_col]
    groups = df[group_col]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tv_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tv, y_tv, g_tv = X.iloc[tv_idx], y.iloc[tv_idx], groups.iloc[tv_idx]

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=test_size / (1 - test_size), random_state=random_state
    )
    tr_idx, va_idx = next(gss2.split(X_tv, y_tv, groups=g_tv))

    return (
        X_tv.iloc[tr_idx], X_tv.iloc[va_idx], X.iloc[te_idx],
        y_tv.iloc[tr_idx], y_tv.iloc[va_idx], y.iloc[te_idx],
    )
