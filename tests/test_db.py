"""Repository / persistence tests against in-memory SQLite."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.database.mysql_models import BehaviorAttempt, PointerEvent
from app.database.repositories import AttemptRepository
from app.services.feature_extractor import FEATURE_SCHEMA_VERSION, extract_features
from tests.conftest import human_like_events


def _attempt_dict(attempt_id="att_1", label="human"):
    return {
        "attempt_id": attempt_id,
        "challenge_id": "c1",
        "session_id": "s1",
        "anonymous_participant_id": "adult_001",
        "schema_version": "1.0",
        "captcha_width": 420,
        "captcha_height": 220,
        "label": label,
        "label_source": "controlled_collection",
        "quality_status": "valid",
    }


def test_save_bundle_persists_attempt_and_events(session):
    repo = AttemptRepository(session)
    events = human_like_events()
    repo.save_attempt_bundle(attempt=_attempt_dict(), events=events, interaction={})
    session.commit()

    stored = session.get(BehaviorAttempt, "att_1")
    assert stored is not None
    stored_events = session.execute(
        select(PointerEvent).where(PointerEvent.attempt_id == "att_1")
    ).scalars().all()
    assert len(stored_events) == len(events)
    # events preserved in seq order
    seqs = sorted(e.seq for e in stored_events)
    assert seqs == list(range(len(events)))


def test_duplicate_attempt_id_detected(session):
    repo = AttemptRepository(session)
    repo.save_attempt_bundle(attempt=_attempt_dict(), events=human_like_events(), interaction={})
    session.commit()
    assert repo.exists("att_1") is True
    assert repo.exists("att_missing") is False


def test_save_features_upsert(session):
    repo = AttemptRepository(session)
    events = human_like_events()
    repo.save_attempt_bundle(attempt=_attempt_dict(), events=events, interaction={})
    feats = extract_features(events, {})
    repo.save_features("att_1", feats, FEATURE_SCHEMA_VERSION)
    session.commit()

    from app.database.mysql_models import AttemptFeatures
    row = session.get(AttemptFeatures, "att_1")
    assert row is not None
    assert row.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert abs(row.event_count - feats["event_count"]) < 1e-9

    # upsert: recompute and overwrite
    feats2 = dict(feats)
    feats2["event_count"] = 999.0
    repo.save_features("att_1", feats2, FEATURE_SCHEMA_VERSION)
    session.commit()
    row2 = session.get(AttemptFeatures, "att_1")
    assert row2.event_count == 999.0
