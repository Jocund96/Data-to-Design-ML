from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

def preprocessing_pipeline(one_hot_encode_cols, target_encode_cols, valid_numeric_features, estimator=XGBRegressor(random_state=42)):
    target_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown_type')),
        ('target_enc', TargetEncoder(categories='auto', cv=5, smooth='auto', random_state=42))
    ])

    one_hot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown_type')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler())
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', one_hot_transformer, one_hot_encode_cols),
            ('target', target_transformer, target_encode_cols),
            ('numeric', numeric_transformer, valid_numeric_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])
    return pipeline
