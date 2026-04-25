from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from moscow_housing.config import PATHS
from moscow_housing.constants import RAW_FEATURE_COLUMNS
from moscow_housing.features import add_features


class ApartmentFeatures(BaseModel):
    apartment_type: str = Field(..., examples=["Secondary"])
    metro_station: str = Field(..., examples=["Каширская"])
    minutes_to_metro: float = Field(..., ge=0, examples=[7])
    region: str = Field(..., examples=["Moscow"])
    number_of_rooms: int = Field(..., gt=0, examples=[2])
    area: float = Field(..., gt=0, examples=[50])
    living_area: float = Field(..., gt=0, examples=[30])
    kitchen_area: float = Field(..., gt=0, examples=[10])
    floor: int = Field(..., gt=0, examples=[5])
    number_of_floors: int = Field(..., gt=0, examples=[10])
    renovation: str = Field(..., examples=["cosmetic"])

    @model_validator(mode="after")
    def validate_apartment_geometry(self) -> ApartmentFeatures:
        if self.living_area > self.area:
            raise ValueError("living_area must be less than or equal to area")
        if self.kitchen_area > self.area:
            raise ValueError("kitchen_area must be less than or equal to area")
        if self.floor > self.number_of_floors:
            raise ValueError("floor must be less than or equal to number_of_floors")
        return self


class PredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    currency: str = "RUB"


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_columns: list[str]
    test_metrics: dict[str, float]


app = FastAPI(
    title="Moscow Housing Price Prediction API",
    description="FastAPI service for predicting Moscow apartment prices from listing features.",
    version="0.3.0",
)


@lru_cache(maxsize=1)
def get_model_payload() -> dict[str, Any]:
    if not PATHS.final_model_path.exists():
        raise FileNotFoundError(
            f"{PATHS.final_model_path} not found. Run `make train-cp2` before starting the API."
        )
    return joblib.load(PATHS.final_model_path)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        payload = get_model_payload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelInfoResponse(
        model_name=str(payload["model_name"]),
        feature_columns=list(payload["feature_columns"]),
        test_metrics=dict(payload.get("test_metrics", {})),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ApartmentFeatures) -> PredictionResponse:
    try:
        payload = get_model_payload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    raw_df = pd.DataFrame([features.model_dump()])[RAW_FEATURE_COLUMNS]
    model_df = add_features(raw_df)
    feature_columns = list(payload["feature_columns"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*",
            category=UserWarning,
        )
        prediction = float(payload["pipeline"].predict(model_df[feature_columns])[0])

    return PredictionResponse(
        predicted_price=max(prediction, 0.0),
        model_name=str(payload["model_name"]),
    )
