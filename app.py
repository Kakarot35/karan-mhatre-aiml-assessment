"""
FastAPI service for lead conversion prediction and explanation.

Run training first with ``python train.py`` to create model.pkl, then start:
    uvicorn app:app --reload
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from utils import get_confidence_level, get_risk_level, setup_logging

logger = setup_logging()

app = FastAPI(title=config.API_TITLE, version=config.API_VERSION)


class PredictionInput(BaseModel):
    """Request body for lead conversion inference."""

    lead_id: str | None = Field(default=None, examples=["L00123"])
    pages_visited: int = Field(ge=0, examples=[14])
    time_spent_minutes: float = Field(ge=0, examples=[45])
    pricing_views: int = Field(ge=0, examples=[3])
    session_count: int = Field(ge=0, examples=[4])
    days_since_first_visit: int = Field(ge=0, examples=[12])
    first_touch_channel: str = Field(min_length=1, examples=["Google"])
    company_size: str = Field(min_length=1, examples=["Medium"])


class PredictionOutput(BaseModel):
    """Response body returned by the prediction endpoint."""

    lead_id: str
    conversion_probability: float
    confidence: str
    risk_level: str


class ExplainInput(BaseModel):
    """Request body for rule-based conversion explanation."""

    conversion_probability: float = Field(ge=0, le=1, examples=[0.84])
    pages_visited: int = Field(ge=0, examples=[14])
    demo_requests: int = Field(ge=0, examples=[1])
    pricing_views: int = Field(ge=0, examples=[3])


class ExplainOutput(BaseModel):
    """Response body for explanation requests."""

    summary: str


def _load_artifact() -> dict[str, Any] | None:
    """Load model artifact if available; defer failure to predict requests."""
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        logger.warning("Model artifact not found at %s", model_path)
        return None
    try:
        with model_path.open("rb") as model_file:
            artifact = pickle.load(model_file)
        logger.info("Loaded model artifact from %s", model_path)
        return artifact
    except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
        logger.exception("Could not load model artifact: %s", exc)
        return None


MODEL_ARTIFACT = _load_artifact()


def _as_model_frame(payload: PredictionInput, artifact: dict[str, Any]) -> pd.DataFrame:
    """Convert API fields into the feature frame expected by the pipeline."""
    row: dict[str, Any] = {
        "pages_visited": payload.pages_visited,
        "time_spent_minutes": payload.time_spent_minutes,
        "pricing_views": payload.pricing_views,
        "session_count": payload.session_count,
        "days_since_first_visit": payload.days_since_first_visit,
        "days_since_last_visit": payload.days_since_first_visit,
        "avg_scroll_depth": 0,
        "unique_pages": payload.pages_visited,
        "active_days": payload.days_since_first_visit,
        "first_touch_channel": payload.first_touch_channel,
        "company_size": payload.company_size,
        "lead_segment": "unknown",
        "device_type": "unknown",
    }

    columns = artifact.get("numeric_features", []) + artifact.get("categorical_features", [])
    return pd.DataFrame([{column: row.get(column, 0) for column in columns}])


@app.get("/health")
def health() -> dict[str, str]:
    """Expose a lightweight readiness check."""
    return {
        "status": "ok",
        "model_loaded": "true" if MODEL_ARTIFACT is not None else "false",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput) -> PredictionOutput:
    """Predict conversion probability for one lead."""
    if MODEL_ARTIFACT is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run `python train.py` after adding data files.",
        )

    pipeline = MODEL_ARTIFACT["pipeline"]
    features = _as_model_frame(input_data, MODEL_ARTIFACT)
    try:
        probability = float(pipeline.predict_proba(features)[0][1])
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    probability = round(probability, 4)
    return PredictionOutput(
        lead_id=input_data.lead_id or "ad-hoc-lead",
        conversion_probability=probability,
        confidence=get_confidence_level(probability),
        risk_level=get_risk_level(probability),
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(input_data: ExplainInput) -> ExplainOutput:
    """Generate a concise business explanation for a prediction."""
    summary = generate_explanation(input_data)
    return ExplainOutput(summary=summary)


def generate_explanation(data: ExplainInput) -> str:
    """Create a deterministic, business-readable lead intent explanation."""
    signals: list[str] = []
    if data.demo_requests > 0:
        signals.append("a demo request, which is a strong buying-intent signal")
    if data.pricing_views >= 3:
        signals.append("multiple pricing page visits")
    elif data.pricing_views > 0:
        signals.append("pricing page interest")
    if data.pages_visited >= 12:
        signals.append("broad product exploration across many pages")

    if data.conversion_probability >= config.CONFIDENCE_HIGH:
        intent = "high"
    elif data.conversion_probability >= config.CONFIDENCE_LOW:
        intent = "moderate"
    else:
        intent = "low"

    if signals:
        signal_text = ", ".join(signals)
        return (
            f"This lead shows {intent} conversion intent because the behavior includes "
            f"{signal_text}. Sales should prioritize timely follow-up with messaging "
            "focused on product fit and next-step clarity."
        )

    return (
        f"This lead currently shows {intent} conversion intent. Engagement is limited, "
        "so the recommended next action is nurturing with educational content before "
        "a direct sales push."
    )
