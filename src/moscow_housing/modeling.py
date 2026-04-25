from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from moscow_housing.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from moscow_housing.features import QuantileClipper

MAX_REASONABLE_PRICE = 10_000_000_000
MAX_LOG_TARGET = np.log1p(MAX_REASONABLE_PRICE)


def safe_expm1_target(y: np.ndarray) -> np.ndarray:
    """Invert log1p target transform without allowing numerical overflow."""

    return np.expm1(np.clip(y, a_min=0, a_max=MAX_LOG_TARGET))


def get_feature_columns(df: pd.DataFrame, use_feature_engineering: bool = True) -> list[str]:
    if use_feature_engineering:
        return [col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES if col in df.columns]

    raw_numeric = [
        "minutes_to_metro",
        "number_of_rooms",
        "area",
        "living_area",
        "kitchen_area",
        "floor",
        "number_of_floors",
    ]
    raw_categorical = ["apartment_type", "metro_station", "region", "renovation"]
    return [col for col in raw_numeric + raw_categorical if col in df.columns]


def build_preprocessor(feature_columns: Iterable[str]) -> ColumnTransformer:
    feature_columns = list(feature_columns)
    numeric_features = [col for col in NUMERIC_FEATURES if col in feature_columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in feature_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(lower_quantile=0.01, upper_quantile=0.99)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator,
    feature_columns: Iterable[str],
    log_target: bool = True,
) -> Pipeline:
    model = estimator
    if log_target:
        model = TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=safe_expm1_target,
        )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_columns)),
            ("model", model),
        ]
    )


def get_experiments(df: pd.DataFrame) -> dict[str, tuple[Pipeline, list[str]]]:
    raw_features = get_feature_columns(df, use_feature_engineering=False)
    full_features = get_feature_columns(df, use_feature_engineering=True)

    return {
        "dummy_median_raw": (
            build_pipeline(DummyRegressor(strategy="median"), raw_features, log_target=False),
            raw_features,
        ),
        "knn_raw_baseline": (
            build_pipeline(KNeighborsRegressor(n_neighbors=10), raw_features, log_target=True),
            raw_features,
        ),
        "ridge_with_features": (
            build_pipeline(Ridge(alpha=10.0, random_state=RANDOM_STATE), full_features, log_target=True),
            full_features,
        ),
        "random_forest_with_features": (
            build_pipeline(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
        ),
        "extra_trees_with_features": (
            build_pipeline(
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
        ),
        "hist_gradient_boosting_with_features": (
            build_pipeline(
                HistGradientBoostingRegressor(
                    learning_rate=0.07,
                    max_iter=300,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
        ),
    }
