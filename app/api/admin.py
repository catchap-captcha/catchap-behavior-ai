"""POST /api/v1/admin/model/reload — explicit model reload (admin only).

Used to hot-swap the production model bundle after a promotion, without
restarting the service. Requires X-Admin-Key.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.config import get_settings
from app.schemas.responses import ReloadResponse
from app.services.model_service import model_service

router = APIRouter(tags=["admin"])


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    configured = get_settings().admin_api_key
    if not configured or x_admin_key != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing admin key"
        )


@router.post(
    "/api/v1/admin/model/reload",
    response_model=ReloadResponse,
    dependencies=[Depends(require_admin_key)],
)
def reload_model() -> ReloadResponse:
    loaded = model_service.load()
    return ReloadResponse(
        reloaded=True,
        model_loaded=loaded,
        model_name=model_service.model_name,
        model_version=model_service.model_version,
    )
