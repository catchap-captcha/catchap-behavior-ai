"""Outbound response schemas (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CollectResponse(BaseModel):
    attempt_id: str
    stored: bool
    idempotent: bool               # True when attempt_id already existed
    quality_status: str            # valid | pending | rejected
    rejection_reason: str | None = None
    feature_schema_version: str


class PredictResponse(BaseModel):
    attempt_id: str
    prediction: Literal["human", "bot"]
    human_score: float
    bot_risk_score: float
    bot_decision: str              # low_risk | high_risk
    threshold: float
    model_name: str
    model_version: str
    feature_schema_version: str


class ModelNotReadyResponse(BaseModel):
    """Body returned with HTTP 503 when no production model is loaded.

    A fake score is NEVER returned in this situation.
    """

    reason: Literal["model_not_ready"] = "model_not_ready"
    detail: str = "No production model is loaded. Train and promote a model first."


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    mysql_connected: bool
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    feature_schema_version: str


class ReloadResponse(BaseModel):
    reloaded: bool
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
