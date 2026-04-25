from __future__ import annotations

import json

from fastapi.testclient import TestClient

from moscow_housing.api import app


def main() -> None:
    client = TestClient(app)
    payload = {
        "apartment_type": "Secondary",
        "metro_station": "Каширская",
        "minutes_to_metro": 7,
        "region": "Moscow",
        "number_of_rooms": 2,
        "area": 50,
        "living_area": 30,
        "kitchen_area": 10,
        "floor": 5,
        "number_of_floors": 10,
        "renovation": "cosmetic",
    }

    health = client.get("/health")
    health.raise_for_status()

    prediction = client.post("/predict", json=payload)
    prediction.raise_for_status()

    print(json.dumps({"health": health.json(), "prediction": prediction.json()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
