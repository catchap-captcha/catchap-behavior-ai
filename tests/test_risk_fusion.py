"""Tests for advisory model/replay/rate risk fusion."""

from __future__ import annotations

from app.services.replay_detector import ReplayFeatures
from app.services.risk_fusion import RiskFusionPolicy, fuse_behavior_risk


def _replay(**overrides) -> ReplayFeatures:
    values = {
        "path_similarity_score": 0.2,
        "exact_replay_detected": False,
        "repeated_duration_count": 0,
        "attempts_per_minute": 1.0,
        "recent_attempt_count": 1,
        "repeated_endpoint_count": 0,
    }
    values.update(overrides)
    return ReplayFeatures(**values)


def _policy() -> RiskFusionPolicy:
    return RiskFusionPolicy(
        model_human_threshold=0.6,
        dtw_similarity_threshold=0.97,
        max_attempts_per_minute=20.0,
    )


def test_low_risk_attempt_is_allowed():
    decision = fuse_behavior_risk(0.9, _replay(), _policy())

    assert decision.risk_score == 5.0
    assert decision.risk_level == "low"
    assert decision.recommended_action == "allow"
    assert decision.reasons == ()


def test_model_only_signal_recommends_step_up_not_hard_enforcement():
    decision = fuse_behavior_risk(0.1, _replay(), _policy())

    assert decision.risk_level == "medium"
    assert decision.recommended_action == "step_up"
    assert decision.reasons == ("ml_bot_score",)


def test_step_up_band_requests_second_verification_without_bot_verdict():
    policy = RiskFusionPolicy(
        model_human_threshold=0.6,
        step_up_human_threshold=0.85,
        dtw_similarity_threshold=0.97,
        max_attempts_per_minute=20.0,
    )

    decision = fuse_behavior_risk(0.8, _replay(), policy)

    assert decision.risk_score == 40.0
    assert decision.risk_level == "medium"
    assert decision.recommended_action == "step_up"
    assert decision.reasons == ("ml_step_up_band",)


def test_step_up_band_keeps_high_confidence_human_attempt_allowed():
    policy = RiskFusionPolicy(
        model_human_threshold=0.6,
        step_up_human_threshold=0.85,
        dtw_similarity_threshold=0.97,
        max_attempts_per_minute=20.0,
    )

    decision = fuse_behavior_risk(0.9, _replay(), policy)

    assert decision.risk_level == "low"
    assert decision.recommended_action == "allow"
    assert decision.reasons == ()


def test_replay_or_multiple_security_signals_raise_high_risk():
    exact = fuse_behavior_risk(0.9, _replay(exact_replay_detected=True), _policy())
    combined = fuse_behavior_risk(
        0.9,
        _replay(path_similarity_score=0.98, attempts_per_minute=20.0),
        _policy(),
    )

    assert exact.risk_level == "high"
    assert exact.recommended_action == "step_up_and_rate_limit"
    assert "exact_trace_fingerprint" in exact.reasons
    assert combined.risk_level == "high"
    assert combined.reasons == ("dtw_similar_trace", "session_rate_exceeded")


def test_multiple_reasons_are_preserved_for_audit():
    decision = fuse_behavior_risk(
        0.1,
        _replay(
            exact_replay_detected=True,
            path_similarity_score=1.0,
            attempts_per_minute=50.0,
        ),
        _policy(),
    )

    assert decision.risk_score == 100.0
    assert decision.risk_level == "high"
    assert decision.reasons == (
        "ml_bot_score",
        "exact_trace_fingerprint",
        "dtw_similar_trace",
        "session_rate_exceeded",
    )


def test_invalid_event_telemetry_never_receives_allow():
    decision = fuse_behavior_risk(
        0.999999,
        _replay(),
        _policy(),
        quality_rejected=True,
    )

    assert decision.risk_level == "medium"
    assert decision.recommended_action == "step_up"
    assert decision.reasons == ("invalid_event_telemetry",)


def test_inhuman_session_rate_alone_forces_step_up():
    # A human-scored trajectory fired at the rate limit must not stay allow:
    # session_rate_weight (30) sits below medium (40), so before the floor a
    # 20+/min replay session still returned allow (red-team R8). No real user
    # solves 20 CAPTCHAs a minute, so flooring at medium is FRR-safe.
    decision = fuse_behavior_risk(
        0.999999,
        _replay(attempts_per_minute=25.0, recent_attempt_count=25),
        _policy(),
    )

    assert decision.recommended_action == "step_up"
    assert "session_rate_exceeded" in decision.reasons


def test_replay_similarity_alone_reaches_step_up():
    """★A repeated path must cost something on its own.

    Before the floor the additive weight (35) sat below medium (40), so a bot
    the model reads as human replayed the same shape and got `allow` every
    time. Measured 2026-08-13 in production: six consecutive wrong answers,
    `dtw_similar_trace` on every one, risk 35.0x, all allowed.
    """
    decision = fuse_behavior_risk(
        0.999, _replay(path_similarity_score=0.98), _policy(),
    )

    assert decision.reasons == ("dtw_similar_trace",)
    assert decision.risk_level == "medium"
    assert decision.recommended_action == "step_up"


def test_replay_similarity_stays_below_hard_enforcement():
    """A repeat is worth a second check, not a block.

    `step_up` already rejects the attempt even when the answer was right, so
    the person solves one more CAPTCHA. At the measured ~0.26% of attempts
    (same-challenge retries, 9.0% of which cross the threshold) that is
    affordable; rate limiting on top of it would not be.
    """
    decision = fuse_behavior_risk(
        0.999, _replay(path_similarity_score=0.98), _policy(),
    )

    assert decision.recommended_action != "step_up_and_rate_limit"


def test_path_below_threshold_is_untouched():
    """Ordinary drags must not move. Across different challenges the signal
    never fired on real people (0 of 19,131 pairs)."""
    decision = fuse_behavior_risk(
        0.999, _replay(path_similarity_score=0.5), _policy(),
    )

    assert decision.risk_level == "low"
    assert decision.recommended_action == "allow"
    assert decision.reasons == ()
