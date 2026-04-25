from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features by train quantiles inside sklearn Pipeline.

    Это важно для воспроизводимости и защиты от data leakage:
    пороги выбросов считаются только на обучающей выборке.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> QuantileClipper:
        self.lower_bounds_ = np.nanquantile(x, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(x, self.upper_quantile, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower_bounds_, self.upper_bounds_)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features without using target column."""

    result = df.copy()

    rooms = result["number_of_rooms"].replace(0, np.nan)
    area = result["area"].replace(0, np.nan)
    floors_total = result["number_of_floors"].replace(0, np.nan)

    result["area_per_room"] = result["area"] / rooms
    result["living_area_share"] = result["living_area"] / area
    result["kitchen_area_share"] = result["kitchen_area"] / area
    result["floor_ratio"] = result["floor"] / floors_total

    result["is_first_floor"] = (result["floor"] == 1).astype(int)
    result["is_last_floor"] = (result["floor"] == result["number_of_floors"]).astype(int)
    result["is_studio"] = (result["number_of_rooms"] == 1).astype(int)

    result["area_log"] = np.log1p(result["area"])
    result["minutes_to_metro_log"] = np.log1p(result["minutes_to_metro"])
    result["room_area_interaction"] = result["number_of_rooms"] * result["area"]
    result["kitchen_to_living_ratio"] = result["kitchen_area"] / result["living_area"].replace(0, np.nan)

    result["metro_distance_bucket"] = pd.cut(
        result["minutes_to_metro"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["0_5_min", "5_10_min", "10_20_min", "20_plus_min"],
    ).astype("object")

    result["floor_category"] = pd.cut(
        result["floor_ratio"],
        bins=[-np.inf, 0.10, 0.35, 0.75, np.inf],
        labels=["very_low", "low_middle", "middle_high", "top"],
    ).astype("object")

    result["is_moscow"] = result["region"].astype(str).str.lower().eq("moscow").astype(int)

    return result
