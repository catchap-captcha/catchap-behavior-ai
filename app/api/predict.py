"""POST /api/v1/behavior/predict — production inference.

Validates the raw events, computes features, and scores with the loaded
production model. If no model is loaded it returns HTTP 503 with
``model_not_ready`` — never a fabricated score. This endpoint does NOT accept a
label. Raw events + features may optionally be persisted for later review.
"""

from __future__ import annotations

from datetime import timezone

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import AttemptRepository, PredictionRepository
from app.schemas.requests import PredictRequest
from app.schemas.responses import ModelNotReadyResponse, PredictResponse
from app.services.feature_extractor import FEATURE_SCHEMA_VERSION, extract_features
from app.services.model_service import model_service
from app.services.quality_validator import validate_attempt

router = APIRouter(tags=["predict"])


def _to_naive_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@router.post("/api/v1/behavior/predict")
def predict(payload: PredictRequest, session: Session = Depends(get_session)):
    # No model -> 503, never a fake score.
    if not model_service.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ModelNotReadyResponse().model_dump(),
        )

    events = [e.model_dump() for e in payload.events]

    # Validate raw events (result is advisory here; we still score, but a broken
    # payload is recorded with its quality status when persisted).
    quality = validate_attempt(
        events,
        captcha_width=payload.captcha.width,
        captcha_height=payload.captcha.height,
        presented_at=_to_naive_utc(payload.timing.presented_at),
        submitted_at=_to_naive_utc(payload.timing.submitted_at),
    )

    features = extract_features(events, payload.interaction.model_dump())
    result = model_service.score(features)

    # best-effort persistence of raw attempt + features + prediction
    _persist(session, payload, events, features, quality, result)

    return PredictResponse(
        attempt_id=payload.attempt_id,
        prediction=result["prediction"],
        human_score=result["human_score"],
        bot_risk_score=result["bot_risk_score"],
        bot_decision=result["bot_decision"],
        threshold=result["threshold"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        feature_schema_version=result["feature_schema_version"],
    )


def _persist(session: Session, payload, events, features, quality, result) -> None:
    """Store the inference attempt. Failure here never breaks the response."""
    try:
        repo = AttemptRepository(session)
        if not repo.exists(payload.attempt_id):
            repo.save_attempt_bundle(
                attempt={
                    "attempt_id": payload.attempt_id,
                    "challenge_id": payload.challenge_id,
                    "session_id": payload.session_id,
                    "anonymous_participant_id": payload.anonymous_participant_id,
                    "schema_version": payload.schema_version,
                    "captcha_width": payload.captcha.width,
                    "captcha_height": payload.captcha.height,
                    "presented_at": _to_naive_utc(payload.timing.presented_at),
                    "submitted_at": _to_naive_utc(payload.timing.submitted_at),
                    # inference traffic is unlabelled
                    "label": "unknown",
                    "label_source": None,
                    "quality_status": quality.status,
                    "rejection_reason": quality.reason,
                },
                events=events,
                interaction=payload.interaction.model_dump(),
            )
            repo.save_features(payload.attempt_id, features, FEATURE_SCHEMA_VERSION)
        PredictionRepository(session).save_prediction(
            attempt_id=payload.attempt_id,
            human_score=result["human_score"],
            bot_risk_score=result["bot_risk_score"],
            bot_decision=result["bot_decision"],
            threshold=result["threshold"],
            model_name=result["model_name"],
            model_version=result["model_version"],
            feature_schema_version=result["feature_schema_version"],
        )
        session.commit()
    except Exception:
        session.rollback()
