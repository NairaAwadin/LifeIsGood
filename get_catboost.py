import pandas as pd
import pipeline_v
import pipeline_0
import numpy as np
from typing import List, Optional, Tuple
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_catboost_regressor(   
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: List[str],
    model_path: Optional[str] = None,     # if provided -> load instead of train
    save_path: Optional[str] = None,
    test_size: float = 0.2,
    seed: int = 7,
    model_params: Optional[dict] = None,
    early_stopping_rounds: int = 100
):
    #quick preprocess/clean/cast...
    X = X.copy()
    y = pd.to_numeric(y, errors="coerce")
    m = y.notna()
    X, y = X.loc[m], y.loc[m].astype(float)
    for c in cat_cols :
        X[c] = X[c].astype("string")
    #split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    #get cat indexes
    cat_idxs = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    #indicate which are categorical cols
    train_pool = Pool(X_train, y_train, cat_features=cat_idxs)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idxs)

    #build model parameters
    params = dict(
        loss_function="MAE",
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        random_seed=seed,
        verbose=200,
        task_type="GPU"
    )
    #update params if function received params
    if model_params:
        params.update(model_params)

    #get base model and load trained model if function receive model path
    base_model = None
    if model_path:
        base_model = CatBoostRegressor()
        base_model.load_model(model_path)
    
        expected = base_model.feature_names_
        for c in expected:
            if c not in X.columns:
                X[c] = pd.NA
        X = X[expected]  # reorder + drop extras

    #load params to model
    model = CatBoostRegressor(**params)
    
    #train/fit model
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        init_model=base_model,   # <-- continue from loaded model if provided
    )

    mae = float(mean_absolute_error(y_valid, model.predict(valid_pool)))

    if save_path:
        model.save_model(save_path)

    return model, mae


def load_catboost(model_path:str):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

def predict(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    cat_cols: List[str],
):
    X = X.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string")
    cat_idxs = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    return model.predict(Pool(X, cat_features=cat_idxs))
