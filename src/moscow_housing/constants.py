from __future__ import annotations

TARGET_COLUMN = "price"

RAW_TO_CANONICAL_COLUMNS = {
    "Price": "price",
    "Apartment type": "apartment_type",
    "Metro station": "metro_station",
    "Minutes to metro": "minutes_to_metro",
    "Region": "region",
    "Number of rooms": "number_of_rooms",
    "Area": "area",
    "Living area": "living_area",
    "Kitchen area": "kitchen_area",
    "Floor": "floor",
    "Number of floors": "number_of_floors",
    "Renovation": "renovation",
}

REQUIRED_COLUMNS = list(RAW_TO_CANONICAL_COLUMNS.values())

RAW_FEATURE_COLUMNS = [
    "apartment_type",
    "metro_station",
    "minutes_to_metro",
    "region",
    "number_of_rooms",
    "area",
    "living_area",
    "kitchen_area",
    "floor",
    "number_of_floors",
    "renovation",
]

ENGINEERED_FEATURE_COLUMNS = [
    "area_per_room",
    "living_area_share",
    "kitchen_area_share",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "metro_distance_bucket",
    "is_moscow",
    "area_log",
    "minutes_to_metro_log",
    "room_area_interaction",
    "kitchen_to_living_ratio",
    "is_studio",
    "floor_category",
]

CATEGORICAL_FEATURES = [
    "apartment_type",
    "metro_station",
    "region",
    "renovation",
    "metro_distance_bucket",
    "floor_category",
]

NUMERIC_FEATURES = [
    "minutes_to_metro",
    "number_of_rooms",
    "area",
    "living_area",
    "kitchen_area",
    "floor",
    "number_of_floors",
    "area_per_room",
    "living_area_share",
    "kitchen_area_share",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "is_moscow",
    "area_log",
    "minutes_to_metro_log",
    "room_area_interaction",
    "kitchen_to_living_ratio",
    "is_studio",
]

RANDOM_STATE = 42
