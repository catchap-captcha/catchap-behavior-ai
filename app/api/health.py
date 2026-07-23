"""GET /health — liveness + dependency status."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.database.connection import check_connection
from app.schemas.responses import HealthResponse
from app.services.model_service import feature_schema_version, model_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mysql_ok = check_connection()
    model_ok = model_service.is_ready()
    return HealthResponse(
        status="ok" if mysql_ok else "degraded",
        mysql_connected=mysql_ok,
        model_loaded=model_ok,
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        feature_schema_version=feature_schema_version(),
        policy_mode=get_settings().risk_policy_mode,
    )
