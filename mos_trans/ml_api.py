"""Independent inference service. OpenAPI is available at /docs."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mos_trans.inference import Predictor


class PredictionRequest(BaseModel):
    features: list[dict[str, Any]] = Field(min_length=1, max_length=256)


class Prediction(BaseModel):
    sample_id: str
    predicted_delay_s: float
    probability_delay_over_120s: float


class PredictionResponse(BaseModel):
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(
        os.getenv("REGRESSOR_PATH", "artifacts/route_delay_trend_submission/catboost_residual_final.cbm"),
        os.getenv("PROBABILITY_PATH", "artifacts/calibrated_probability/model.joblib"),
    )
    yield


app = FastAPI(title="Mos-TRANS ML", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = app.state.predictor.predict(pd.DataFrame(request.features))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(predictions=[Prediction(**row) for row in result.to_dict("records")])
