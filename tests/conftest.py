"""Shared pytest fixtures.

DB tests run against an in-memory SQLite database (schema created here for the
tests only — the application itself never issues DDL). ML/GAN tests that need
xgboost / lightgbm / torch skip automatically if those libraries are absent.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.mysql_models import Base
from app.services.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_features


@pytest.fixture(autouse=True, scope="session")
def _isolate_from_local_env():
    """Keep a developer's local `.env` out of the suite.

    `Settings` reads `.env` from the working directory, so a local
    PRODUCTION_MODEL_DIR pointing at a real candidate bundle silently loads a
    model and breaks every test that asserts the no-model behaviour. Tests must
    describe the code, not the machine they run on.
    """
    import os

    from app.config import get_settings

    previous = os.environ.get("PRODUCTION_MODEL_DIR")
    os.environ["PRODUCTION_MODEL_DIR"] = "models/__tests_no_model__"
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PRODUCTION_MODEL_DIR", None)
        else:
            os.environ["PRODUCTION_MODEL_DIR"] = previous
        get_settings.cache_clear()


@pytest.fixture()
def sqlite_sessionmaker():
    # StaticPool + check_same_thread=False share ONE in-memory DB across threads,
    # so the FastAPI TestClient's worker thread sees the same tables.
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    # SQLite ignores foreign keys unless asked. Without this the tests cannot
    # see wrong insert ordering, which is exactly how a bug that made /predict
    # store nothing on MySQL (errno 1452) passed the whole suite.
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _record):  # noqa: ANN001
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@pytest.fixture()
def session(sqlite_sessionmaker):
    s = sqlite_sessionmaker()
    try:
        yield s
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# synthetic drags (fixtures — NOT real human data, never labelled as such
# outside these tests)
# --------------------------------------------------------------------------- #
def human_like_events(n: int = 40) -> list[dict[str, Any]]:
    """A curved, variable-speed drag with a small pause and a tiny correction."""
    events = []
    for i in range(n):
        frac = i / (n - 1)
        x = 300 * frac
        y = 30 + 8 * math.sin(frac * math.pi)  # gentle arc
        # ease-in/ease-out timing + jitter-free but non-uniform intervals
        t = int(800 * (frac ** 1.5))
        etype = "pointerdown" if i == 0 else "pointerup" if i == n - 1 else "pointermove"
        events.append({
            "seq": i, "event_type": etype, "t_ms": t,
            "x": round(x, 3), "y": round(y, 3),
            "x_normalized": round(x / 420, 6), "y_normalized": round(y / 220, 6),
            "target_role": "slider_handle",
        })
    return events


def bot_like_events(n: int = 20) -> list[dict[str, Any]]:
    """A perfectly straight, constant-speed drag with fixed 16ms intervals."""
    events = []
    for i in range(n):
        frac = i / (n - 1)
        x = 300 * frac
        etype = "pointerdown" if i == 0 else "pointerup" if i == n - 1 else "pointermove"
        events.append({
            "seq": i, "event_type": etype, "t_ms": i * 16,
            "x": round(x, 3), "y": 30.0,
            "x_normalized": round(x / 420, 6), "y_normalized": round(30 / 220, 6),
            "target_role": "slider_handle",
        })
    return events


def make_row(
    label: str,
    *,
    participant: str | None = None,
    bot_family: str | None = None,
    generator_version: str | None = None,
    attempt_id: str = "att",
) -> dict[str, Any]:
    """Build one training-view-shaped row with computed features + metadata."""
    events = human_like_events() if label == "human" else bot_like_events()
    feats = extract_features(events, {})
    row: dict[str, Any] = {
        "attempt_id": attempt_id,
        "challenge_id": "c1",
        "session_id": "s1",
        "anonymous_participant_id": participant,
        "label": label,
        "label_source": "controlled_collection" if label == "human" else "rule_bot",
        "bot_family": bot_family,
        "generator_version": generator_version,
        "age_group": "adult",
        "schema_version": "1.0",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "position_correct": True,
        "interaction_success": True,
        "final_drop_error": 1.0,
    }
    row.update({name: feats[name] for name in FEATURE_NAMES})
    return row


@pytest.fixture()
def training_rows():
    """A small balanced, group-diverse dataset for split/train/evaluate tests."""
    rows: list[dict[str, Any]] = []
    for p in range(8):  # 8 human participants, 3 attempts each
        for k in range(3):
            rows.append(make_row(
                "human", participant=f"adult_{p:03d}",
                attempt_id=f"h_{p}_{k}",
            ))
    for fam_i, fam in enumerate(["straight", "accel", "jitter", "linear"]):
        for k in range(6):
            rows.append(make_row(
                "bot", bot_family=fam, generator_version=f"gen_{fam}_v1",
                attempt_id=f"b_{fam}_{k}",
            ))
    return rows
