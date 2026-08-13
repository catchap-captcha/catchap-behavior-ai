"""SQLAlchemy 2.x engine and session management for MySQL 8.0.

The engine is built from environment variables only (see :mod:`app.config`).
This module intentionally provides no ``create_all`` / DDL helper: the DB team
owns table creation. The application connects to tables that already exist.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process-wide engine, creating it lazily.

    Uses a pooled PyMySQL connection with ``pool_pre_ping`` so stale
    connections are recycled transparently.

    ★``connect_timeout`` 을 반드시 짧게 준다 (2026-08-13 실장애로 추가)
    ------------------------------------------------------------------
    보안그룹에서 3306 응답 경로가 사라지면 패킷이 **거부가 아니라 버려진다**.
    그러면 연결이 끊기지 않고 **매단다**. PyMySQL 기본 연결 제한시간은 10초인데
    ``/health`` 를 부르는 liveness 검사의 제한시간은 **5초**다.

    그래서 ``check_connection()`` 이 ``degraded`` 를 돌려주도록 잘 만들어져 있어도
    **그 답이 제시간에 못 온다** → 검사 3회 연속 실패 → 쿠버네티스가 컨테이너를 죽인다.
    0813 에 실제로 **두 파드가 같은 초에 6번** 재시작했다.

    2초로 두면 DB 가 안 되더라도 ``/health`` 가 **200 + degraded 로 제때 응답**하므로
    파드는 살아 있고, DB 가 돌아오면 재시작 없이 스스로 회복한다.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.sqlalchemy_url,
            pool_size=settings.mysql_pool_size,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 2},
            future=True,
        )
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _SessionLocal


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session that is always closed."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


def check_connection() -> bool:
    """Return True if a trivial ``SELECT 1`` succeeds, False otherwise."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
