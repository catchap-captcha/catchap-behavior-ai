"""Shadow-mode outcome recording for the trusted CAPTCHA backend.

This router never changes a CAPTCHA verdict. It records the final observed
result so the team can compare it against the action the AI would have taken.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.challenge import require_captcha_backend_key
from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import ShadowOutcomeRepository
from app.schemas.requests import ShadowOutcomeRequest
from app.schemas.responses import ShadowOutcomeResponse

router = APIRouter(tags=["shadow-mode"])


@router.post(
    "/api/v1/behavior/shadow/outcomes",
    response_model=ShadowOutcomeResponse,
    dependencies=[Depends(require_captcha_backend_key)],
)
def record_shadow_outcome(
    payload: ShadowOutcomeRequest,
    session: Session = Depends(get_session),
) -> ShadowOutcomeResponse:
    """Record a trusted final verdict without enforcing the AI recommendation."""
    if get_settings().risk_policy_mode != "shadow":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"reason": "shadow_mode_disabled"},
        )

    try:
        outcome, stored = ShadowOutcomeRepository(session).record(
            attempt_id=payload.attempt_id,
            main_captcha_verdict=payload.main_captcha_verdict,
            final_verdict=payload.final_verdict,
        )
        if stored:
            session.commit()
    except LookupError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"reason": "prediction_not_found"},
        ) from None
    except ValueError as error:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"reason": str(error)},
        ) from None

    return ShadowOutcomeResponse(
        attempt_id=outcome.attempt_id,
        stored=stored,
        idempotent=not stored,
        policy_mode="shadow",
        would_have_action=outcome.would_have_action,
        risk_level=outcome.risk_level,
        model_version=outcome.model_version,
    )
