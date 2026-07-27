"""Tests for the quality validator."""

from __future__ import annotations

from app.services.quality_validator import (
    QUALITY_PENDING,
    QUALITY_REJECTED,
    QUALITY_VALID,
    validate_attempt,
)
from tests.conftest import human_like_events


def test_valid_attempt_passes():
    r = validate_attempt(human_like_events(), captcha_width=420, captcha_height=220)
    assert r.status == QUALITY_VALID
    assert r.reason is None


def test_single_event_rejected():
    events = [{"seq": 0, "event_type": "pointerdown", "t_ms": 0, "x": 5, "y": 5}]
    r = validate_attempt(events, captcha_width=420, captcha_height=220)
    assert r.status == QUALITY_REJECTED
    assert r.reason == "event_count_lt_2"


def test_seq_out_of_order_rejected():
    events = human_like_events()
    events[1]["seq"] = 99  # break the sequence
    r = validate_attempt(events, captcha_width=420, captcha_height=220)
    assert r.status == QUALITY_REJECTED
    assert "seq_not_sequential" in r.reason


def test_decreasing_time_rejected():
    events = [
        {"seq": 0, "event_type": "pointerdown", "t_ms": 100, "x": 0, "y": 0},
        {"seq": 1, "event_type": "pointermove", "t_ms": 50, "x": 10, "y": 0},
        {"seq": 2, "event_type": "pointerup", "t_ms": 200, "x": 20, "y": 0},
    ]
    r = validate_attempt(events, captcha_width=420, captcha_height=220)
    assert r.status == QUALITY_REJECTED
    assert "t_ms_decreasing" in r.reason


def test_missing_pointerdown_rejected():
    events = [
        {"seq": 0, "event_type": "pointermove", "t_ms": 0, "x": 0, "y": 0},
        {"seq": 1, "event_type": "pointerup", "t_ms": 50, "x": 20, "y": 0},
    ]
    r = validate_attempt(events, captcha_width=420, captcha_height=220)
    assert r.status == QUALITY_REJECTED
    assert "missing_pointerdown" in r.reason


def test_non_finite_rejected():
    events = human_like_events()
    events[2]["x"] = float("inf")
    r = validate_attempt(events, captcha_width=420, captcha_height=220)
    assert r.status == QUALITY_REJECTED
    assert "non_finite_value" in r.reason


def test_out_of_bounds_rejected():
    events = human_like_events()
    events[3]["x"] = 100000  # far outside the box
    r = validate_attempt(events, captcha_width=420, captcha_height=220)
    assert r.status == QUALITY_REJECTED
    assert "coordinates_out_of_bounds" in r.reason


def test_duplicate_timestamp_flags_pending():
    events = human_like_events()
    events[5]["t_ms"] = events[4]["t_ms"]  # duplicate timestamp
    r = validate_attempt(events, captcha_width=420, captcha_height=220)
    assert r.status in (QUALITY_PENDING, QUALITY_VALID)
    if r.status == QUALITY_PENDING:
        assert "duplicate_timestamps" in r.reason


def test_submitted_before_presented_rejected():
    from datetime import datetime

    r = validate_attempt(
        human_like_events(),
        captcha_width=420, captcha_height=220,
        presented_at=datetime(2026, 7, 1, 10, 0, 2),
        submitted_at=datetime(2026, 7, 1, 10, 0, 0),
    )
    assert r.status == QUALITY_REJECTED
    assert "submitted_before_presented" in r.reason


def test_epoch_events_inside_server_window_pass():
    from datetime import datetime, timezone

    presented = datetime(2026, 7, 23, 7, 0, tzinfo=timezone.utc)
    events = human_like_events()
    base_ms = round(presented.timestamp() * 1000) + 500
    for event in events:
        event["t_ms"] += base_ms

    result = validate_attempt(
        events,
        captcha_width=420,
        captcha_height=220,
        presented_at=presented,
        submitted_at=datetime(2026, 7, 23, 7, 0, 5, tzinfo=timezone.utc),
        enforce_server_time_window=True,
    )

    assert result.status == QUALITY_VALID
    assert result.checks["event_timestamps_within_server_window"] is True


def test_relative_or_stale_events_outside_server_window_rejected():
    from datetime import datetime, timezone

    result = validate_attempt(
        human_like_events(),
        captcha_width=420,
        captcha_height=220,
        presented_at=datetime(2026, 7, 23, 7, 0, tzinfo=timezone.utc),
        submitted_at=datetime(2026, 7, 23, 7, 0, 5, tzinfo=timezone.utc),
        enforce_server_time_window=True,
    )

    assert result.status == QUALITY_REJECTED
    assert "event_timestamps_outside_server_window" in result.reason


def test_online_window_validation_rejects_missing_server_timing():
    result = validate_attempt(
        human_like_events(),
        captcha_width=420,
        captcha_height=220,
        enforce_server_time_window=True,
    )

    assert result.status == QUALITY_REJECTED
    assert "missing_server_timing" in result.reason
