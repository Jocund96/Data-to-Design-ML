import pandas as df

## Data Filtering Utility

def drop_by_threshold(df, threshold_pct):
    """Drop columns where missing values exceed threshold percentage."""
    threshold = threshold_pct * len(df)
    dropped = df.columns[df.isna().sum() > threshold].tolist()
    print(f"Dropped columns ({int(threshold_pct*100)}% threshold): {dropped}")
    return df.loc[:, df.isna().sum() <= threshold]


def recode_semantic_missingness(df, amount_col, type_col):
    """Recode missing values based on material amount.
    
    Rules:
    - amount > 0 & type NaN → 'unknown_type' (material used, type unknown)
    - amount NaN/0 & type NaN → 'not_applicable' (material not used)
    """
    # Material used but type unknown
    mask1 = (df[amount_col] > 0) & df[type_col].isna()
    df.loc[mask1, type_col] = 'unknown_type'

    # Material not used (amount 0 or NaN) and type NaN
    mask2 = (df[amount_col].isna() | (df[amount_col] == 0)) & df[type_col].isna()
    df.loc[mask2, type_col] = 'not_applicable'
    df.loc[mask2, amount_col] = 0

    return df


