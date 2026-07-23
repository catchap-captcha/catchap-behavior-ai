"""Tests for server-side one-time challenge consumption."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.mysql_models import Base, CaptchaChallenge
from app.database.repositories import ChallengeRepository, _utcnow


def _issue(session):
    challenge = ChallengeRepository(session).issue(
        session_id="s_1",
        site_key="site_1",
        purpose="login_guard",
        problem_binding="question-batch:hash",
        ttl_seconds=120,
    )
    session.commit()
    return challenge


def _consume(repository, challenge):
    return repository.consume(
        challenge_id=challenge.challenge_id,
        nonce=challenge.nonce,
        session_id="s_1",
        site_key="site_1",
        purpose="login_guard",
        problem_binding="question-batch:hash",
        verdict="passed",
    )


def test_nonce_is_hashed_and_expired_challenge_is_rejected(session):
    issued = _issue(session)
    stored = session.get(CaptchaChallenge, issued.challenge_id)
    assert stored.nonce_hash != issued.nonce
    assert len(stored.nonce_hash) == 64

    stored.expires_at = _utcnow() - timedelta(seconds=1)
    session.commit()
    result = _consume(ChallengeRepository(session), issued)

    assert result.accepted is False
    assert result.reason == "challenge_expired"


def test_concurrent_consumes_allow_only_one_success(tmp_path):
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'challenge.sqlite'}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    first_session = factory()
    try:
        issued = _issue(first_session)
    finally:
        first_session.close()

    def consume_once():
        session = factory()
        try:
            barrier.wait()
            result = _consume(ChallengeRepository(session), issued)
            session.commit()
            return result
        finally:
            session.close()

    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: consume_once(), range(2)))

    assert sum(result.accepted for result in results) == 1
    assert sorted(result.reason for result in results) == ["challenge_already_consumed", "consumed"]
