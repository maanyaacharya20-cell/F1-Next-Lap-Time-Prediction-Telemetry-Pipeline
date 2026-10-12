"""Training and inference for the next-lap duration model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def make_pipeline(random_state=42):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    regressor = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def train_model(dataset, model_path="models/next_lap_model.joblib", test_fraction=0.2):
    """Train chronologically and persist the fitted preprocessing/model pipeline."""
    dataset = dataset.copy().reset_index(drop=True)
    if len(dataset) < 5:
        raise ValueError("At least five valid lap rows are required to train the model.")
    dataset = dataset.dropna(subset=[TARGET_COLUMN])
    if len(dataset) < 5:
        raise ValueError("At least five valid target rows are required to train the model.")

    split_index = max(1, int(len(dataset) * (1 - test_fraction)))
    split_index = min(split_index, len(dataset) - 1)
    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]
    if len(train) < 2 or test.empty:
        raise ValueError("The chronological split did not produce train and test rows.")

    model = make_pipeline()
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    predictions = model.predict(test[FEATURE_COLUMNS])
    rmse = mean_squared_error(test[TARGET_COLUMN], predictions) ** 0.5
    metrics = {
        "mae": float(mean_absolute_error(test[TARGET_COLUMN], predictions)),
        "rmse": float(rmse),
        "r2": float(r2_score(test[TARGET_COLUMN], predictions)) if len(test) > 1 else float("nan"),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model, metrics


def predict_next_lap(model, conditions):
    """Predict one or more next-lap durations from dashboard conditions."""
    frame = conditions if isinstance(conditions, pd.DataFrame) else pd.DataFrame([conditions])
    return model.predict(frame[FEATURE_COLUMNS])
