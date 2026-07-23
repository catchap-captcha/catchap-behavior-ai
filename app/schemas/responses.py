"""Outbound response schemas (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class CollectResponse(BaseModel):
    attempt_id: str
    stored: bool
    idempotent: bool               # True when attempt_id already existed
    quality_status: str            # valid | pending | rejected
    rejection_reason: str | None = None
    feature_schema_version: str
    learning_stored: bool = False  # True when the WHAT block was stored for recommendation


class PredictResponse(BaseModel):
    attempt_id: str
    # Policy score, not a calibrated probability that the caller may use for
    # direct blocking. The CAPTCHA backend owns the final enforcement decision.
    risk_score: float
    risk_level: Literal["low", "medium", "high"]
    recommended_action: Literal["allow", "step_up", "step_up_and_rate_limit"]
    policy_mode: Literal["shadow", "active"]
    reasons: list[str]
    human_score: float
    bot_risk_score: float
    path_similarity_score: float
    exact_replay_detected: bool
    attempts_per_minute: float
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
    policy_mode: Literal["shadow", "active"]


class ReloadResponse(BaseModel):
    reloaded: bool
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ChallengeIssueResponse(BaseModel):
    challenge_id: str
    nonce: str
    expires_at: datetime


class ChallengeConsumeResponse(BaseModel):
    challenge_id: str
    consumed: bool
    verdict: Literal["passed", "failed"] | None = None


class ShadowOutcomeResponse(BaseModel):
    attempt_id: str
    stored: bool
    idempotent: bool
    policy_mode: Literal["shadow"]
    would_have_action: Literal["allow", "step_up", "step_up_and_rate_limit"]
    risk_level: Literal["low", "medium", "high"]
    model_version: str


class ShadowActionSummary(BaseModel):
    would_have_action: Literal["allow", "step_up", "step_up_and_rate_limit"]
    attempts: int
    main_passed: int
    main_failed: int
    main_pass_rate: float


class ShadowSummaryResponse(BaseModel):
    policy_mode: Literal["shadow", "active"]
    total_outcomes: int
    would_step_up_count: int
    would_step_up_rate: float
    actions: list[ShadowActionSummary]
