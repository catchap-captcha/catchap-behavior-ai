"""FastAPI application entrypoint.

Wires the collect / predict / health / admin routers and attempts to load the
production model at startup. A missing model is fine — /predict will return 503
until one is promoted; the service still starts.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import admin, challenge, collect, health, predict, shadow
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Best-effort load; absence of a model must not stop startup.
    try:
        model_service.load()
    except Exception:
        pass
    yield


app = FastAPI(
    title="catchap ai-service",
    description="Web drag CAPTCHA Human/Bot behavioral analysis service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(collect.router)
app.include_router(predict.router)
app.include_router(shadow.router)
app.include_router(challenge.router)
app.include_router(admin.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "catchap-ai-service", "status": "running"}
