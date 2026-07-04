"""API tests: collect idempotency, auth, and predict-without-model 503."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.database.connection import get_session
from app.services.model_service import model_service
from tests.conftest import human_like_events


@pytest.fixture()
def client(sqlite_sessionmaker, monkeypatch):
    monkeypatch.setenv("COLLECT_API_KEY", "test-collect-key")
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    get_settings.cache_clear()

    from app.main import app

    def _override_session():
        s = sqlite_sessionmaker()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_session] = _override_session
    # ensure no model is loaded for these tests
    model_service._bundle = None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def _collect_payload(attempt_id="att_api_1"):
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "challenge_id": "captcha_1",
        "session_id": "session_1",
        "anonymous_participant_id": "adult_001",
        "captcha": {"width": 420, "height": 220},
        "timing": {"presented_at": "2026-07-01T10:00:00Z", "submitted_at": "2026-07-01T10:00:02Z"},
        "events": human_like_events(),
        "interaction": {"regrab_count": 0, "retry_count": 0, "pointercancel_count": 0,
                        "empty_click_count": 0, "failed_drop_count": 0},
        "collection": {"label": "human", "label_source": "controlled_collection", "age_group": "adult"},
    }


def test_collect_requires_api_key(client):
    resp = client.post("/api/v1/behavior/collect", json=_collect_payload())
    assert resp.status_code == 401


def test_collect_and_idempotency(client):
    headers = {"X-API-Key": "test-collect-key"}
    first = client.post("/api/v1/behavior/collect", json=_collect_payload(), headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert body["stored"] is True
    assert body["idempotent"] is False
    assert body["quality_status"] == "valid"

    second = client.post("/api/v1/behavior/collect", json=_collect_payload(), headers=headers)
    assert second.status_code == 200
    body2 = second.json()
    assert body2["stored"] is False
    assert body2["idempotent"] is True


def test_predict_without_model_returns_503(client):
    payload = _collect_payload("att_pred_1")
    payload.pop("collection", None)  # predict takes no label
    resp = client.post("/api/v1/behavior/predict", json=payload)
    assert resp.status_code == 503
    assert resp.json()["reason"] == "model_not_ready"


def test_collect_stores_learning_when_what_present(client):
    headers = {"X-API-Key": "test-collect-key"}
    payload = _collect_payload("att_learn_1")
    # add the WHAT block: correct tile grabbed, dropped in slot -> correct
    payload["final_drop_error"] = 1.0
    payload["learning"] = {
        "question_id": "q1",
        "concept_id": "ADD_WITHIN_5",
        "difficulty": 0.3,
        "answer_options_count": 3,
        "correct_answer_id": "5",
        "grabbed_answer_id": "5",
        "released_target_id": "slot",
    }
    resp = client.post("/api/v1/behavior/collect", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["learning_stored"] is True

    # the learning_attempts row was written with a computed judgment
    from app.database.mysql_models import LearningAttempt
    from app.database.connection import get_session  # overridden dependency

    session_gen = client.app.dependency_overrides[get_session]()
    session = next(session_gen)
    try:
        row = session.get(LearningAttempt, "att_learn_1")
        assert row is not None
        assert row.concept_id == "ADD_WITHIN_5"
        assert row.outcome == "correct" and row.is_correct is True
        assert row.valid_for_learning is True
        assert row.captcha_attempt_id == "att_learn_1"  # linked to the bot record
    finally:
        session.close()


def test_collect_without_learning_block_still_works(client):
    headers = {"X-API-Key": "test-collect-key"}
    payload = _collect_payload("att_botonly_1")  # no learning block
    resp = client.post("/api/v1/behavior/collect", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["learning_stored"] is False


def test_health_reports_no_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False
    assert body["feature_schema_version"] == "1.0"
