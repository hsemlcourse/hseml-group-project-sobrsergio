from __future__ import annotations

import pandas as pd

from moscow_housing.features import add_features


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
