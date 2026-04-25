from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from moscow_housing import api


class DummyPipeline:
    def predict(self, x):
        assert list(x.columns)
        return np.array([12_345_678.9])


def _sample_payload() -> dict:
    return {
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


def test_health_endpoint() -> None:
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "get_model_payload",
        lambda: {
            "model_name": "dummy_model",
            "feature_columns": ["area", "number_of_rooms", "area_per_room"],
            "pipeline": DummyPipeline(),
            "test_metrics": {"rmsle": 0.1},
        },
    )
    client = TestClient(api.app)

    response = client.post("/predict", json=_sample_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price"] == 12_345_678.9
    assert data["model_name"] == "dummy_model"
    assert data["currency"] == "RUB"


def test_predict_rejects_invalid_geometry() -> None:
    client = TestClient(api.app)
    payload = _sample_payload()
    payload["floor"] = 20
    payload["number_of_floors"] = 10

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
