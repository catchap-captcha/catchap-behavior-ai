"""POST /api/v1/behavior/collect — controlled data collection.

Only the CAPTCHA backend may call this (X-API-Key). It stores the raw attempt +
pointer events + interaction summary in one transaction, runs quality checks,
computes and stores the 29 features, and NEVER triggers retraining. Duplicate
``attempt_id`` requests are idempotent.

The label is taken from the trusted ``collection`` context, not from anything the
end-user frontend could control (the frontend cannot reach this key-gated route).
"""

from __future__ import annotations

from datetime import timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import AttemptRepository
from app.schemas.requests import CollectRequest
from app.schemas.responses import CollectResponse
from app.services.feature_extractor import (
    FEATURE_SCHEMA_VERSION,
    extract_features,
)
from app.services.quality_validator import validate_attempt

router = APIRouter(tags=["collect"])


def require_collect_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject the request unless a valid collection API key is presented."""
    configured = get_settings().collect_api_key
    if not configured or x_api_key != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )


def _to_naive_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@router.post(
    "/api/v1/behavior/collect",
    response_model=CollectResponse,
    dependencies=[Depends(require_collect_key)],
)
def collect(payload: CollectRequest, session: Session = Depends(get_session)) -> CollectResponse:
    repo = AttemptRepository(session)

    # idempotency: identical attempt_id is accepted without re-writing
    if repo.exists(payload.attempt_id):
        existing = repo.get_attempt(payload.attempt_id)
        return CollectResponse(
            attempt_id=payload.attempt_id,
            stored=False,
            idempotent=True,
            quality_status=existing.quality_status,
            rejection_reason=existing.rejection_reason,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )

    events = [e.model_dump() for e in payload.events]
    presented = _to_naive_utc(payload.timing.presented_at)
    submitted = _to_naive_utc(payload.timing.submitted_at)

    quality = validate_attempt(
        events,
        captcha_width=payload.captcha.width,
        captcha_height=payload.captcha.height,
        presented_at=presented,
        submitted_at=submitted,
    )

    attempt = {
        "attempt_id": payload.attempt_id,
        "challenge_id": payload.challenge_id,
        "session_id": payload.session_id,
        "anonymous_participant_id": payload.anonymous_participant_id,
        "schema_version": payload.schema_version,
        "captcha_width": payload.captcha.width,
        "captcha_height": payload.captcha.height,
        "presented_at": presented,
        "submitted_at": submitted,
        "position_correct": payload.position_correct,
        "interaction_success": payload.interaction_success,
        "final_drop_error": payload.final_drop_error,
        # trusted labelling context (never from the end-user frontend)
        "label": payload.collection.label,
        "label_source": payload.collection.label_source,
        "bot_family": payload.collection.bot_family,
        "generator_version": payload.collection.generator_version,
        "age_group": payload.collection.age_group,
        "consent_version": payload.collection.consent_version,
        "quality_status": quality.status,
        "rejection_reason": quality.reason,
    }

    try:
        repo.save_attempt_bundle(
            attempt=attempt, events=events, interaction=payload.interaction.model_dump()
        )
        # features are always computable (extractor never raises) and are stored
        # even for pending/rejected rows so they can be reviewed; training filters
        # on quality_status via the DB view.
        features = extract_features(events, payload.interaction.model_dump())
        repo.save_features(payload.attempt_id, features, FEATURE_SCHEMA_VERSION)
        session.commit()
    except Exception:
        session.rollback()
        raise

    return CollectResponse(
        attempt_id=payload.attempt_id,
        stored=True,
        idempotent=False,
        quality_status=quality.status,
        rejection_reason=quality.reason,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
