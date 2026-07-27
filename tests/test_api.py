"""API tests: collect idempotency, auth, and predict-without-model 503."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.database.connection import get_session
from app.services.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.services.model_service import model_service
from tests.conftest import human_like_events


class FixedHumanProbabilityModel:
    """Tiny sklearn-style model used only to exercise the risk API contract."""

    classes_ = [0, 1]

    def __init__(self, human_probability: float):
        self.human_probability = human_probability
        self.last_vector_width: int | None = None

    def predict_proba(self, vector):
        self.last_vector_width = len(vector[0])
        return [
            [1.0 - self.human_probability, self.human_probability]
            for _ in range(len(vector))
        ]


@pytest.fixture()
def client(sqlite_sessionmaker, monkeypatch):
    monkeypatch.setenv("COLLECT_API_KEY", "test-collect-key")
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setenv("CAPTCHA_BACKEND_API_KEY", "test-captcha-backend-key")
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
    events = human_like_events()
    presented_at_ms = 1_782_900_000_000
    for event in events:
        event["t_ms"] += presented_at_ms + 250
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "challenge_id": "captcha_1",
        "session_id": "session_1",
        "anonymous_participant_id": "adult_001",
        "captcha": {"width": 420, "height": 220},
        "timing": {"presented_at": "2026-07-01T10:00:00Z", "submitted_at": "2026-07-01T10:00:02Z"},
        "events": events,
        "interaction": {"regrab_count": 0, "retry_count": 0, "pointercancel_count": 0,
                        "empty_click_count": 0, "failed_drop_count": 0},
        "collection": {"label": "human", "label_source": "controlled_collection", "age_group": "adult"},
    }


def _backend_headers():
    return {"X-Captcha-Backend-Key": "test-captcha-backend-key"}


def _load_risk_model(
    human_probability: float, *, step_up_threshold: float | None = None
) -> None:
    bundle = {
        "model": FixedHumanProbabilityModel(human_probability),
        "model_name": "fixed-test-model",
        "model_version": "risk-test-v1",
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "threshold": 0.55,
    }
    if step_up_threshold is not None:
        bundle["step_up_threshold"] = step_up_threshold
    model_service._bundle = bundle


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
    resp = client.post("/api/v1/behavior/predict", json=payload, headers=_backend_headers())
    assert resp.status_code == 503
    assert resp.json()["reason"] == "model_not_ready"


def test_predict_requires_captcha_backend_key(client):
    payload = _collect_payload("att_pred_auth")
    payload.pop("collection", None)
    response = client.post("/api/v1/behavior/predict", json=payload)
    assert response.status_code == 401


def test_predict_returns_advisory_risk_and_detects_same_session_replay(client):
    _load_risk_model(0.9)
    first_payload = _collect_payload("att_risk_1")
    first_payload.pop("collection", None)
    first = client.post(
        "/api/v1/behavior/predict", json=first_payload, headers=_backend_headers()
    )
    assert first.status_code == 200, first.text
    assert first.json()["risk_level"] == "low"
    assert first.json()["recommended_action"] == "allow"
    assert first.json()["risk_score"] == 5.0

    replay_payload = _collect_payload("att_risk_2")
    replay_payload.pop("collection", None)
    replay_payload["challenge_id"] = "captcha_2"
    replay = client.post(
        "/api/v1/behavior/predict", json=replay_payload, headers=_backend_headers()
    )
    assert replay.status_code == 200, replay.text
    body = replay.json()
    assert body["risk_level"] == "high"
    assert body["recommended_action"] == "step_up_and_rate_limit"
    assert body["exact_replay_detected"] is True
    assert "exact_trace_fingerprint" in body["reasons"]


def test_predict_routes_candidate_step_up_band_to_second_verification(client):
    _load_risk_model(0.8, step_up_threshold=0.85)
    payload = _collect_payload("att_step_up_band")
    payload.pop("collection", None)

    response = client.post(
        "/api/v1/behavior/predict", json=payload, headers=_backend_headers()
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk_level"] == "medium"
    assert body["recommended_action"] == "step_up"
    assert body["policy_mode"] == "shadow"
    assert body["reasons"] == ["ml_step_up_band"]


def test_predict_never_allows_missing_trusted_server_timing(client):
    _load_risk_model(0.999999)
    payload = _collect_payload("att_missing_server_timing")
    payload.pop("collection", None)
    payload["timing"] = {}

    response = client.post(
        "/api/v1/behavior/predict", json=payload, headers=_backend_headers()
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk_level"] == "medium"
    assert body["recommended_action"] == "step_up"
    assert body["reasons"] == ["invalid_event_telemetry"]


def test_shadow_mode_records_would_have_action_without_changing_captcha_verdict(client):
    _load_risk_model(0.8, step_up_threshold=0.85)
    payload = _collect_payload("att_shadow_step_up")
    payload.pop("collection", None)

    prediction = client.post(
        "/api/v1/behavior/predict", json=payload, headers=_backend_headers()
    )
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()["recommended_action"] == "step_up"
    assert prediction.json()["policy_mode"] == "shadow"

    outcome_payload = {
        "attempt_id": "att_shadow_step_up",
        "main_captcha_verdict": "passed",
        "final_verdict": "passed",
    }
    recorded = client.post(
        "/api/v1/behavior/shadow/outcomes",
        json=outcome_payload,
        headers=_backend_headers(),
    )
    assert recorded.status_code == 200, recorded.text
    assert recorded.json() == {
        "attempt_id": "att_shadow_step_up",
        "stored": True,
        "idempotent": False,
        "policy_mode": "shadow",
        "would_have_action": "step_up",
        "risk_level": "medium",
        "model_version": "risk-test-v1",
    }

    duplicate = client.post(
        "/api/v1/behavior/shadow/outcomes",
        json=outcome_payload,
        headers=_backend_headers(),
    )
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["stored"] is False
    assert duplicate.json()["idempotent"] is True

    summary = client.get(
        "/api/v1/admin/shadow/summary",
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert summary.status_code == 200, summary.text
    assert summary.json() == {
        "policy_mode": "shadow",
        "total_outcomes": 1,
        "would_step_up_count": 1,
        "would_step_up_rate": 1.0,
        "actions": [
            {
                "would_have_action": "step_up",
                "attempts": 1,
                "main_passed": 1,
                "main_failed": 0,
                "main_pass_rate": 1.0,
            }
        ],
    }


def test_shadow_outcome_requires_captcha_backend_key(client):
    response = client.post(
        "/api/v1/behavior/shadow/outcomes",
        json={
            "attempt_id": "att_missing_auth",
            "main_captcha_verdict": "passed",
            "final_verdict": "passed",
        },
    )
    assert response.status_code == 401


def test_shadow_outcome_refuses_a_changed_final_captcha_verdict(client):
    response = client.post(
        "/api/v1/behavior/shadow/outcomes",
        json={
            "attempt_id": "att_changed_shadow_verdict",
            "main_captcha_verdict": "passed",
            "final_verdict": "failed",
        },
        headers=_backend_headers(),
    )

    assert response.status_code == 422
    assert "must preserve the main CAPTCHA verdict" in response.text


def test_shadow_outcome_is_disabled_when_policy_mode_is_active(client, monkeypatch):
    monkeypatch.setenv("RISK_POLICY_MODE", "active")
    get_settings.cache_clear()

    response = client.post(
        "/api/v1/behavior/shadow/outcomes",
        json={
            "attempt_id": "att_active_mode",
            "main_captcha_verdict": "passed",
            "final_verdict": "passed",
        },
        headers=_backend_headers(),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "shadow_mode_disabled"
    get_settings.cache_clear()


def test_predict_uses_only_trajectory_features_for_trajectory_model(client):
    from app.services.feature_extractor_v2 import TRAJECTORY_ONLY_FEATURE_NAMES

    model = FixedHumanProbabilityModel(0.9)
    model_service._bundle = {
        "model": model,
        "model_name": "trajectory-test-model",
        "model_version": "trajectory-test-v1",
        "feature_names": TRAJECTORY_ONLY_FEATURE_NAMES,
        "feature_schema_version": "2.0",
        "feature_input_scope": "pointer_trajectory_only",
        "threshold": 0.55,
    }
    payload = _collect_payload("att_trajectory_only")
    payload.pop("collection", None)
    payload["interaction"]["regrab_count"] = 4

    response = client.post(
        "/api/v1/behavior/predict", json=payload, headers=_backend_headers()
    )

    assert response.status_code == 200, response.text
    assert response.json()["feature_schema_version"] == "2.0"
    assert model.last_vector_width == len(TRAJECTORY_ONLY_FEATURE_NAMES) == 39


def test_two_view_model_service_uses_the_lower_human_score(client):
    from app.services.trajectory_feature_views import FEATURE_VIEWS

    general = FixedHumanProbabilityModel(0.90)
    dynamics = FixedHumanProbabilityModel(0.40)
    model_service._bundle = {
        "models": {
            "general_without_physics": general,
            "dynamics_physics": dynamics,
        },
        "model_name": "two-view-test-model",
        "model_version": "two-view-test-v1",
        "feature_schema_version": "2.3",
        "feature_input_scope": "pointer_trajectory_only",
        "feature_views": FEATURE_VIEWS,
        "score_fusion": "min(P_human_general_without_physics, P_human_dynamics_physics)",
        "threshold": 0.55,
    }

    result = model_service.score({name: 1.0 for names in FEATURE_VIEWS.values() for name in names})

    assert result["human_score"] == 0.4
    assert result["prediction"] == "bot"
    assert general.last_vector_width == len(FEATURE_VIEWS["general_without_physics"])
    assert dynamics.last_vector_width == len(FEATURE_VIEWS["dynamics_physics"])


def test_v21_predict_then_single_use_challenge_consumption(client):
    """Exercise the CAPTCHA backend flow with the v2.1 trajectory profile."""
    from app.services.feature_extractor_v21 import TRAJECTORY_ONLY_FEATURE_NAMES

    model = FixedHumanProbabilityModel(0.9)
    model_service._bundle = {
        "model": model,
        "model_name": "trajectory-v21-test-model",
        "model_version": "trajectory-v21-test-v1",
        "feature_names": TRAJECTORY_ONLY_FEATURE_NAMES,
        "feature_schema_version": "2.1",
        "feature_input_scope": "pointer_trajectory_only",
        "threshold": 0.55,
    }
    issued = _issue_challenge(client)
    payload = _collect_payload("att_v21_challenge_flow")
    payload.pop("collection", None)
    payload["challenge_id"] = issued["challenge_id"]
    payload["session_id"] = "preauth_session_1"

    prediction = client.post(
        "/api/v1/behavior/predict", json=payload, headers=_backend_headers()
    )
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()["feature_schema_version"] == "2.1"
    assert model.last_vector_width == len(TRAJECTORY_ONLY_FEATURE_NAMES) == 44

    consumed = client.post(
        "/api/v1/captcha/challenges/consume",
        headers=_backend_headers(),
        json=_consume_payload(issued),
    )
    assert consumed.status_code == 200, consumed.text

    replay = client.post(
        "/api/v1/captcha/challenges/consume",
        headers=_backend_headers(),
        json=_consume_payload(issued),
    )
    assert replay.status_code == 409
    assert replay.json()["detail"]["reason"] == "challenge_already_consumed"


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
    assert body["policy_mode"] == "shadow"


def _issue_challenge(client):
    response = client.post(
        "/api/v1/captcha/challenges",
        headers={"X-Captcha-Backend-Key": "test-captcha-backend-key"},
        json={
            "session_id": "preauth_session_1",
            "site_key": "catchap-web",
            "purpose": "login_guard",
            "problem_binding": "question-batch-v1:sha256:abc123",
        },
    )
    assert response.status_code == 201
    return response.json()


def _consume_payload(issued, **overrides):
    payload = {
        "challenge_id": issued["challenge_id"],
        "nonce": issued["nonce"],
        "session_id": "preauth_session_1",
        "site_key": "catchap-web",
        "purpose": "login_guard",
        "problem_binding": "question-batch-v1:sha256:abc123",
        "verdict": "passed",
    }
    payload.update(overrides)
    return payload


def test_challenge_requires_backend_key(client):
    response = client.post(
        "/api/v1/captcha/challenges",
        json={
            "session_id": "preauth_session_1",
            "site_key": "catchap-web",
            "purpose": "login_guard",
            "problem_binding": "question-batch-v1:sha256:abc123",
        },
    )
    assert response.status_code == 401


def test_failed_verdict_consumes_challenge_and_replay_is_rejected(client):
    issued = _issue_challenge(client)
    headers = {"X-Captcha-Backend-Key": "test-captcha-backend-key"}

    first = client.post(
        "/api/v1/captcha/challenges/consume",
        headers=headers,
        json=_consume_payload(issued, verdict="failed"),
    )
    assert first.status_code == 200
    assert first.json() == {
        "challenge_id": issued["challenge_id"],
        "consumed": True,
        "verdict": "failed",
    }

    replay = client.post(
        "/api/v1/captcha/challenges/consume",
        headers=headers,
        json=_consume_payload(issued, verdict="passed"),
    )
    assert replay.status_code == 409
    assert replay.json()["detail"]["reason"] == "challenge_already_consumed"


def test_binding_mismatch_is_rejected_without_consuming_valid_challenge(client):
    issued = _issue_challenge(client)
    headers = {"X-Captcha-Backend-Key": "test-captcha-backend-key"}

    wrong_session = client.post(
        "/api/v1/captcha/challenges/consume",
        headers=headers,
        json=_consume_payload(issued, session_id="other_session"),
    )
    assert wrong_session.status_code == 403
    assert wrong_session.json()["detail"]["reason"] == "challenge_binding_invalid"

    valid = client.post(
        "/api/v1/captcha/challenges/consume",
        headers=headers,
        json=_consume_payload(issued),
    )
    assert valid.status_code == 200
