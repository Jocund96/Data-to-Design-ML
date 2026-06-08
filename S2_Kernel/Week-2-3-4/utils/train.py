import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR,NuSVR
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold


def train_knn(X_train_kf, y_train_kf, metric, k_range=range(1, 31), scaler_type="RobustScaler", p_degree = 2 ):
    scalers = {
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
    }
    scaler = scalers.get(scaler_type, RobustScaler())

    kfold_rmse_list = []
    kfold_std_list = []

    for k in k_range:
        pipeline = Pipeline([
            ("scaler", scaler),
            ("knn_kf", KNeighborsRegressor( n_neighbors=k, metric = metric,p = p_degree ))
        ])
        scores = cross_val_score(pipeline, X_train_kf, y_train_kf,
                                 cv=3, scoring="neg_root_mean_squared_error")
        kfold_rmse_list.append(-scores.mean())
        kfold_std_list.append(scores.std())

    best_k = list(k_range)[np.argmin(kfold_rmse_list)]
    print(f"Best k (K-Fold): {best_k}, val RMSE: {min(kfold_rmse_list):.4f}")

    best_pipeline = Pipeline([
    ("scaler", scalers.get(scaler_type, RobustScaler())),
    ("knn_kf", KNeighborsRegressor(n_neighbors=best_k, metric=metric))
    ])
    
    return best_k, min(kfold_rmse_list), kfold_rmse_list, best_pipeline


def train_svr(X_train, y_train, svr_type="SVR", param_grid=None, cv=3, scaler_type="RobustScaler",json_path = "config.json" ):
    scalers = {
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
    }

    models = {
        "SVR":   SVR(),
        "NuSVR": NuSVR()
    }

    with open(json_path) as f:
        config = json.load(f)
    
    default_param_grids = {
    "SVR": {
        "svr__C":       config["svr"]["grid"]["C"],
        "svr__epsilon": config["svr"]["grid"]["epsilon"],
        "svr__kernel":  [config["svr"]["kernel"]]
    },
    "NuSVR": {
        "svr__C":      config["nusvr"]["grid"]["C"],
        "svr__nu":     config["nusvr"]["grid"]["nu"],
        "svr__kernel": [config["nusvr"]["kernel"]]
    }
}

    if param_grid is None:
        param_grid = default_param_grids[svr_type]

    pipeline = Pipeline([
        ("scaler", scalers.get(scaler_type, RobustScaler())),
        ("svr", models[svr_type])
    ])

    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_rmse = -grid_search.best_score_

    y_pred_train = grid_search.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)

    print(f"[{svr_type}] Best params: {best_params}")
    print(f"[{svr_type}] Best CV RMSE: {best_rmse:.4f}")
    print(f"[{svr_type}] Train RMSE:   {train_rmse:.4f}")
    print(f"[{svr_type}] Train R²:     {train_r2:.4f}")

    return grid_search.best_estimator_, best_params, best_rmse