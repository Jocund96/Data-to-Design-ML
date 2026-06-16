import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


class CuringRegimeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method_col='curing_method', temp_col='curing_temp', humid_col='curing_humidity', press_col='curing_pressure'):
        self.method_col = method_col
        self.temp_col = temp_col
        self.humid_col = humid_col
        self.press_col = press_col
        
        self.group_means_ = {}
        self.global_means_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Learn means only for whichever columns actually exist in the passed dataset
        for col in [self.temp_col, self.humid_col, self.press_col]:
            if col in X_df.columns and self.method_col in X_df.columns:
                self.group_means_[col] = X_df.groupby(self.method_col)[col].mean().to_dict()
                self.global_means_[col] = X_df[col].mean()
        return self

    def transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        has_method = self.method_col in X_copy.columns
        has_temp = self.temp_col in X_copy.columns
        has_humid = self.humid_col in X_copy.columns
        has_press = self.press_col in X_copy.columns

        if has_method:
            # Build the mask checking if any numeric curing data exists for a row
            has_numeric_data = pd.Series(False, index=X_copy.index)
            if has_temp:  has_numeric_data = has_numeric_data | (X_copy[self.temp_col] >= 0)
            if has_humid: has_numeric_data = has_numeric_data | (X_copy[self.humid_col] >= 0)
            if has_press: has_numeric_data = has_numeric_data | (X_copy[self.press_col] >= 0)

            # if there is numeric data but no text method, it is 'unknown_type'
            method_is_null = X_copy[self.method_col].isnull()
            X_copy.loc[method_is_null & has_numeric_data, self.method_col] = "unknown_type"
            
            # Fill remaining completely blank rows
            X_copy[self.method_col] = X_copy[self.method_col].fillna("not_applicable")

            # Conditional average imputation
            for col in [self.temp_col, self.humid_col, self.press_col]:
                if col in X_copy.columns:
                    missing_mask = X_copy[col].isnull()
                    
                    # Map the curing method to the specific mean learned during fit
                    mapped_means = X_copy[self.method_col].map(self.group_means_.get(col, {}))
                    
                    # Apply mapped means, falling back to global mean if a totally new category appears
                    X_copy.loc[missing_mask, col] = mapped_means[missing_mask].fillna(self.global_means_.get(col, np.nan))

        return X_copy


# Pipeline
def preprocessing_pipeline(one_hot_encode_cols, target_encode_cols, numeric_features, estimator=XGBRegressor(random_state=42)):
    
    # Target Encoding Pipeline 
    target_transformer = Pipeline(steps=[
        ('target_enc', TargetEncoder(categories='auto', cv=5, smooth='auto', random_state=42))
    ])

    # One-Hot Encoding Pipeline 
    one_hot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Numeric Pipeline Median + RobustScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Acts as a safety net for other numericals
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ohe', one_hot_transformer, one_hot_encode_cols),
            ('target', target_transformer, target_encode_cols)
        ],
        remainder='passthrough'
    )

    # Master Pipeline
    pipeline = Pipeline(steps=[
        ('curing_logic', CuringRegimeImputer()), # Safely handles temp, humid, and pressure if present
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])
    
    return pipeline