"""POST /api/v1/behavior/predict — advisory production risk assessment.

Validates the raw events, computes features, and scores with the loaded
production model. If no model is loaded it returns HTTP 503 with
``model_not_ready`` — never a fabricated score. The trusted CAPTCHA backend
receives an advisory risk level and recommended follow-up action; it remains
the sole owner of allow/block and challenge-token decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.challenge import require_captcha_backend_key
from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import AttemptRepository, PredictionRepository
from app.schemas.requests import PredictRequest
from app.schemas.responses import ModelNotReadyResponse, PredictResponse
from app.services.feature_profiles import get_feature_profile
from app.services.model_service import model_service
from app.services.quality_validator import validate_attempt
from app.services.replay_detector import compute_replay_features
from app.services.risk_fusion import RiskFusionPolicy, fuse_behavior_risk

router = APIRouter(tags=["predict"])


def _to_naive_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _duration_ms(events: list[dict]) -> float:
    timestamps = [event.get("t_ms") for event in events if event.get("t_ms") is not None]
    return float(max(timestamps) - min(timestamps)) if timestamps else 0.0


@router.post(
    "/api/v1/behavior/predict",
    dependencies=[Depends(require_captcha_backend_key)],
)
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

    profile = get_feature_profile(
        model_service.feature_schema_version,
        trajectory_only=model_service.feature_input_scope == "pointer_trajectory_only",
    )
    features = profile.extractor(events, payload.interaction.model_dump())
    result = model_service.score(features)

    # Compare only with recent attempts from this trusted backend session. A
    # browser never calls this route, so session_id is backend-derived rather
    # than an untrusted client claim.
    settings = get_settings()
    now = _utcnow()
    attempts = AttemptRepository(session)
    replay = compute_replay_features(
        events,
        duration_ms=_duration_ms(events),
        now_epoch_s=now.replace(tzinfo=timezone.utc).timestamp(),
        history=attempts.recent_session_history(
            session_id=payload.session_id,
            now=now,
            window_seconds=settings.risk_history_window_seconds,
            limit=settings.risk_history_max_attempts,
        ),
        recent_window_s=float(settings.risk_history_window_seconds),
    )
    assessment = fuse_behavior_risk(
        result["human_score"],
        replay,
        RiskFusionPolicy(
            model_human_threshold=result["threshold"],
            step_up_human_threshold=result.get("step_up_threshold"),
            dtw_similarity_threshold=settings.risk_dtw_similarity_threshold,
            max_attempts_per_minute=settings.risk_max_attempts_per_minute,
        ),
    )

    # best-effort persistence of raw attempt + features + prediction
    _persist(
        session,
        payload,
        events,
        features,
        profile.version,
        quality,
        result,
        replay,
        assessment,
        settings.risk_policy_mode,
    )

    return PredictResponse(
        attempt_id=payload.attempt_id,
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        recommended_action=assessment.recommended_action,
        policy_mode=settings.risk_policy_mode,
        reasons=list(assessment.reasons),
        human_score=result["human_score"],
        bot_risk_score=result["bot_risk_score"],
        path_similarity_score=assessment.path_similarity_score,
        exact_replay_detected=assessment.exact_replay_detected,
        attempts_per_minute=assessment.attempts_per_minute,
        threshold=result["threshold"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        feature_schema_version=result["feature_schema_version"],
    )


def _persist(
    session: Session,
    payload,
    events,
    features,
    feature_schema_version: str,
    quality,
    result,
    replay,
    assessment,
    policy_mode: str,
) -> None:
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
            repo.save_features(payload.attempt_id, features, feature_schema_version)
            repo.save_security_features(
                payload.attempt_id,
                {
                    "path_similarity_score": replay.path_similarity_score,
                    "exact_replay_detected": replay.exact_replay_detected,
                    "repeated_duration_count": replay.repeated_duration_count,
                    "attempts_per_minute": replay.attempts_per_minute,
                    "recent_attempt_count": replay.recent_attempt_count,
                    "repeated_endpoint_count": replay.repeated_endpoint_count,
                },
            )
        PredictionRepository(session).save_prediction(
            attempt_id=payload.attempt_id,
            human_score=result["human_score"],
            bot_risk_score=result["bot_risk_score"],
            bot_decision=f"{assessment.risk_level}_risk",
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            recommended_action=assessment.recommended_action,
            policy_mode=policy_mode,
            risk_reasons=list(assessment.reasons),
            threshold=result["threshold"],
            model_name=result["model_name"],
            model_version=result["model_version"],
            feature_schema_version=result["feature_schema_version"],
        )
        session.commit()
    except Exception:
        session.rollback()
