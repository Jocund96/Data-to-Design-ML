import pandas as pd


def numeric_summary_table(df, float_only=True):
    """Per-numeric-column summary: mean, median, std, min, max, CV%, skewness and high-side outlier count."""
    if float_only:
        cols = df.select_dtypes(include=['float']).columns
    else:
        cols = df.select_dtypes(include='number').columns
    data = df[cols]

    cv = (data.std() / data.mean() * 100).round(1)

    return pd.DataFrame({
        'mean': data.mean().round(2),
        'median': data.median().round(2),
        'std': data.std().round(2),
        'min': data.min().round(2),
        'max': data.max().round(2),
        'cv (%)': cv,
        'skewness': data.skew().round(2),
        'outliers': [(data[c] > data[c].quantile(0.75) + 1.5 * data[c].std()).sum() for c in data.columns]
    })


def iqr_bounds(series):
    """Return (lower, upper) IQR-based outlier bounds for a numeric series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def iqr_outlier_summary(df, num_cols=None):
    """Count and percentage of IQR-based outliers per numeric column (columns with 0 outliers omitted)."""
    if num_cols is None:
        num_cols = df.select_dtypes(include='number').columns

    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    outlier_counts = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).sum()
    outlier_pct = (outlier_counts / len(df) * 100).round(1)

    summary = pd.DataFrame({'outlier_count': outlier_counts, 'outlier_%': outlier_pct})
    return summary[summary['outlier_count'] > 0].sort_values('outlier_%', ascending=False)


def iqr_outliers(series):
    """Return the subset of values in series that fall outside its IQR bounds."""
    lower, upper = iqr_bounds(series)
    return series[(series < lower) | (series > upper)]
