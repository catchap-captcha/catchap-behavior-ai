"""FastAPI application entrypoint.

Wires the collect / predict / health / admin routers and attempts to load the
production model at startup. A missing model is fine — /predict will return 503
until one is promoted; the service still starts.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import secrets_loader
from app.api import admin, challenge, collect, health, predict, shadow
from app.config import get_settings
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    # ★비밀값을 어디서 받았는지 기동 로그에 남긴다. ★값은 안 찍는다(이름과 개수만).
    #   ⚠️제일 위험한 경우는 SECRETS_BACKEND 가 없어서 로더가 ★조용히 아무것도
    #   안 한 것이다 — 그때도 이 줄이 「미사용」이라고 말해 준다.
    #   ★get_settings() 를 먼저 부르는 이유 = 로더는 그 안에서 돈다. 안 부르면
    #   last_result() 가 None 이라 「실행되지 않았습니다」만 찍힌다.
    #   ★logging 대신 print 인 이유 = 이 앱은 로깅 설정을 하지 않아서
    #   uvicorn 기본 설정으로는 INFO 가 어디에도 안 나온다.
    import sys
    get_settings()
    _secrets = secrets_loader.last_result()
    print("[SECRETS] " + (_secrets.summary() if _secrets
                          else "Secrets Manager 로더가 실행되지 않았습니다"), file=sys.stderr)
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
