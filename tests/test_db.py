"""Repository / persistence tests against in-memory SQLite."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.database.mysql_models import BehaviorAttempt, PointerEvent
from app.database.repositories import AttemptRepository, attempt_uuid
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

    # The deployed PK is a CHAR(36) surrogate; attempt_uuid() is the mapping.
    stored = session.get(BehaviorAttempt, attempt_uuid("att_1"))
    assert stored is not None
    assert stored.extra_metadata["source_attempt_id"] == "att_1"
    stored_events = session.execute(
        select(PointerEvent).where(PointerEvent.attempt_id == attempt_uuid("att_1"))
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

    def _features_row():
        return session.execute(
            select(AttemptFeatures).where(
                AttemptFeatures.attempt_id == attempt_uuid("att_1")
            )
        ).scalar_one_or_none()

    row = _features_row()
    assert row is not None
    assert row.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert abs(row.event_count - feats["event_count"]) < 1e-9
    # extra_features is the authoritative copy — every extracted feature is there.
    assert row.extra_features.keys() == feats.keys()

    # upsert: recompute and overwrite
    feats2 = dict(feats)
    feats2["event_count"] = 999.0
    repo.save_features("att_1", feats2, FEATURE_SCHEMA_VERSION)
    session.commit()
    row2 = _features_row()
    assert row2.event_count == 999.0
    assert row2.extra_features["event_count"] == 999.0
