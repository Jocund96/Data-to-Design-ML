import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    
    return df.reset_index(drop=True)

def scale_data(X_train, scaler_type="RobustScaler"):
    scaler = RobustScaler() if scaler_type == "RobustScaler" else StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    return X_train_scaled, scaler
    
def load_and_prepare_kfold_data(
    filepath, target, columns=None, test_size=0.2, random_state=42
):
    df = pd.read_csv(filepath)
    df = clean_data(df)

    if columns is not None:
        df.columns = columns

    X = df.drop(target, axis=1)
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
