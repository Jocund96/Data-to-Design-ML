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
