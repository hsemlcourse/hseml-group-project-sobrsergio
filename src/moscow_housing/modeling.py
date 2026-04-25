from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
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


def build_pca_pipeline(
    estimator,
    feature_columns: Iterable[str],
    n_components: int | float = 0.95,
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
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
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


def get_cp2_experiments(df: pd.DataFrame) -> dict[str, tuple[Pipeline, list[str], str]]:
    raw_features = get_feature_columns(df, use_feature_engineering=False)
    full_features = get_feature_columns(df, use_feature_engineering=True)

    return {
        "dummy_median_raw": (
            build_pipeline(DummyRegressor(strategy="median"), raw_features, log_target=False),
            raw_features,
            "Baseline: constant median prediction on raw features.",
        ),
        "knn_raw_k10": (
            build_pipeline(KNeighborsRegressor(n_neighbors=10), raw_features, log_target=True),
            raw_features,
            "Simple out-of-the-box KNN baseline without feature engineering.",
        ),
        "ridge_alpha_1": (
            build_pipeline(Ridge(alpha=1.0, random_state=RANDOM_STATE), full_features, log_target=True),
            full_features,
            "Regularized linear model with moderate penalty.",
        ),
        "ridge_alpha_30": (
            build_pipeline(Ridge(alpha=30.0, random_state=RANDOM_STATE), full_features, log_target=True),
            full_features,
            "Regularized linear model with stronger penalty.",
        ),
        "ridge_pca_95": (
            build_pca_pipeline(
                Ridge(alpha=10.0, random_state=RANDOM_STATE),
                full_features,
                n_components=0.95,
                log_target=True,
            ),
            full_features,
            "Dimensionality reduction: PCA keeps 95% of encoded-feature variance.",
        ),
        "random_forest_depth_14_leaf_3": (
            build_pipeline(
                RandomForestRegressor(
                    n_estimators=220,
                    max_depth=14,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
            "RandomForest tuning: shallower trees and stronger leaf regularization.",
        ),
        "random_forest_depth_20_leaf_2": (
            build_pipeline(
                RandomForestRegressor(
                    n_estimators=260,
                    max_depth=20,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
            "RandomForest tuning: deeper trees with moderate regularization.",
        ),
        "extra_trees_depth_18_leaf_2": (
            build_pipeline(
                ExtraTreesRegressor(
                    n_estimators=260,
                    max_depth=18,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
            "ExtraTrees tuning: randomized tree ensemble.",
        ),
        "extra_trees_depth_none_leaf_3": (
            build_pipeline(
                ExtraTreesRegressor(
                    n_estimators=260,
                    max_depth=None,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
            "ExtraTrees tuning: unrestricted depth with stronger leaf regularization.",
        ),
        "hist_gradient_lr_005_leaf_31": (
            build_pipeline(
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=420,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
            "Gradient boosting tuning: lower learning rate and more iterations.",
        ),
        "hist_gradient_lr_007_leaf_31": (
            build_pipeline(
                HistGradientBoostingRegressor(
                    learning_rate=0.07,
                    max_iter=350,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
            "Gradient boosting tuning: CP1-like learning rate with more iterations.",
        ),
        "hist_gradient_lr_004_leaf_63": (
            build_pipeline(
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=460,
                    max_leaf_nodes=63,
                    l2_regularization=0.10,
                    random_state=RANDOM_STATE,
                ),
                full_features,
                log_target=True,
            ),
            full_features,
            "Gradient boosting tuning: larger trees with stronger L2 regularization.",
        ),
    }
