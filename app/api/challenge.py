"""CAPTCHA-backend-only issue and one-time consume endpoints.

The browser must never receive the backend API key.  The CAPTCHA backend reads
the authenticated pre-auth session itself, verifies the puzzle, then consumes
the challenge with a passed or failed verdict.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import ChallengeRepository
from app.schemas.requests import ChallengeConsumeRequest, ChallengeIssueRequest
from app.schemas.responses import ChallengeConsumeResponse, ChallengeIssueResponse

router = APIRouter(tags=["captcha-protocol"])


def require_captcha_backend_key(
    x_captcha_backend_key: str | None = Header(default=None),
) -> None:
    """Permit only the trusted CAPTCHA backend, never a browser client."""
    configured = get_settings().captcha_backend_api_key
    if not configured or x_captcha_backend_key is None or not hmac.compare_digest(
        x_captcha_backend_key, configured
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing CAPTCHA backend key",
        )


@router.post(
    "/api/v1/captcha/challenges",
    response_model=ChallengeIssueResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_captcha_backend_key)],
)
def issue_challenge(
    payload: ChallengeIssueRequest,
    session: Session = Depends(get_session),
) -> ChallengeIssueResponse:
    settings = get_settings()
    requested_ttl = payload.ttl_seconds or settings.captcha_challenge_ttl_seconds
    ttl_seconds = min(requested_ttl, settings.captcha_challenge_max_ttl_seconds)
    issued = ChallengeRepository(session).issue(
        session_id=payload.session_id,
        site_key=payload.site_key,
        purpose=payload.purpose,
        problem_binding=payload.problem_binding,
        ttl_seconds=ttl_seconds,
    )
    session.commit()
    return ChallengeIssueResponse(
        challenge_id=issued.challenge_id,
        nonce=issued.nonce,
        expires_at=issued.expires_at,
    )


@router.post(
    "/api/v1/captcha/challenges/consume",
    response_model=ChallengeConsumeResponse,
    dependencies=[Depends(require_captcha_backend_key)],
)
def consume_challenge(
    payload: ChallengeConsumeRequest,
    session: Session = Depends(get_session),
) -> ChallengeConsumeResponse:
    result = ChallengeRepository(session).consume(
        challenge_id=payload.challenge_id,
        nonce=payload.nonce,
        session_id=payload.session_id,
        site_key=payload.site_key,
        purpose=payload.purpose,
        problem_binding=payload.problem_binding,
        verdict=payload.verdict,
    )
    if result.accepted:
        session.commit()
        return ChallengeConsumeResponse(
            challenge_id=payload.challenge_id,
            consumed=True,
            verdict=result.verdict,
        )

    session.rollback()
    status_by_reason = {
        "challenge_not_found": status.HTTP_404_NOT_FOUND,
        "challenge_already_consumed": status.HTTP_409_CONFLICT,
        "challenge_expired": status.HTTP_410_GONE,
        "challenge_binding_invalid": status.HTTP_403_FORBIDDEN,
    }
    raise HTTPException(
        status_code=status_by_reason[result.reason],
        detail={"reason": result.reason},
    )
