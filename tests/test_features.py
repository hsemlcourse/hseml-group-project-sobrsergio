from __future__ import annotations

import numpy as np
import pandas as pd

from moscow_housing.data import split_data
from moscow_housing.features import add_features
from moscow_housing.metrics import regression_metrics


def test_add_features_creates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "price": [10_000_000],
            "apartment_type": ["Secondary"],
            "metro_station": ["Каширская"],
            "minutes_to_metro": [7],
            "region": ["Moscow"],
            "number_of_rooms": [2],
            "area": [50],
            "living_area": [30],
            "kitchen_area": [10],
            "floor": [5],
            "number_of_floors": [10],
            "renovation": ["cosmetic"],
        }
    )

    result = add_features(df)

    assert result.loc[0, "area_per_room"] == 25
    assert result.loc[0, "living_area_share"] == 0.6
    assert result.loc[0, "kitchen_area_share"] == 0.2
    assert result.loc[0, "floor_ratio"] == 0.5
    assert result.loc[0, "is_first_floor"] == 0
    assert result.loc[0, "is_last_floor"] == 0
    assert result.loc[0, "metro_distance_bucket"] == "5_10_min"
    assert result.loc[0, "is_moscow"] == 1
    assert result.loc[0, "area_log"] == np.log1p(50)
    assert result.loc[0, "minutes_to_metro_log"] == np.log1p(7)
    assert result.loc[0, "room_area_interaction"] == 100
    assert np.isclose(result.loc[0, "kitchen_to_living_ratio"], 1 / 3)
    assert result.loc[0, "is_studio"] == 0
    assert result.loc[0, "floor_category"] == "middle_high"


def test_split_data_is_reproducible_and_complete() -> None:
    df = pd.DataFrame(
        {
            "price": np.linspace(1_000_000, 20_000_000, 100),
            "area": np.linspace(20, 120, 100),
        }
    )

    train_1, val_1, test_1 = split_data(df)
    train_2, val_2, test_2 = split_data(df)

    assert len(train_1) == 70
    assert len(val_1) == 15
    assert len(test_1) == 15
    assert len(train_1) + len(val_1) + len(test_1) == len(df)
    assert train_1.equals(train_2)
    assert val_1.equals(val_2)
    assert test_1.equals(test_2)


def test_regression_metrics_handles_negative_predictions() -> None:
    metrics = regression_metrics(
        y_true=np.array([10.0, 20.0, 30.0]),
        y_pred=np.array([10.0, -5.0, 35.0]),
    )

    assert set(metrics) == {"mae", "rmse", "rmsle", "r2"}
    assert all(np.isfinite(value) for value in metrics.values())
