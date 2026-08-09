# -*- coding: utf-8 -*-
"""금고 로더가 ★Settings 를 만들기 전에 도는지 확인한다.

로더 자체는 `catchap-backend` 에서 이미 시험한 코드를 그대로 옮긴 것이라
여기서 다시 시험하지 않는다. ★이 저장소에서 새로 생긴 것은 「연결」뿐이다.

⚠️잡으려는 사고 — `get_settings()` 안에서 `load_secrets_into_env()` 를
`Settings()` ★뒤에 두면 아무 오류 없이 조용히 옛 값을 쓴다.
"""
import os

from app import config as config_module
from app import secrets_loader
from app.config import get_settings


def test_설정을_읽으면_로더가_돈다():
    get_settings.cache_clear()
    get_settings()
    결과 = secrets_loader.last_result()
    assert 결과 is not None, "get_settings() 를 불렀는데 로더가 안 돌았다"
    # 기본값(SECRETS_BACKEND 없음)에서는 「미사용」으로 끝나야 한다.
    assert 결과.backend == "none"


def test_주입한_값이_Settings_에_닿는다(monkeypatch):
    호출: list[str] = []

    def 가짜로더(environ=None):
        env = environ if environ is not None else os.environ
        env["MYSQL_PASSWORD"] = "주입된-값"
        호출.append("불림")
        return secrets_loader.LoadResult(backend="가짜", loaded=["MYSQL_PASSWORD"])

    monkeypatch.setattr(config_module, "load_secrets_into_env", 가짜로더)
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert 호출, "get_settings() 를 불렀는데 로더가 안 불렸다"
        # ★핵심 — 로더가 Settings() ★앞에서 돌아야 이 값이 보인다.
        assert s.mysql_password == "주입된-값"
    finally:
        os.environ.pop("MYSQL_PASSWORD", None)
        get_settings.cache_clear()
