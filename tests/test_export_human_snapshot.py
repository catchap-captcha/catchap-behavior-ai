"""Tests for the read-only Human snapshot exporter."""

from __future__ import annotations

from tools.export_human_snapshot import (
    _max_normalized_speed,
    _parse_points,
    _pseudonym,
    _quality_decision,
)


def test_pseudonym_is_stable_and_scoped():
    key = b"test-key"
    assert _pseudonym("participant", "abc", key) == _pseudonym("participant", "abc", key)
    assert _pseudonym("participant", "abc", key) != _pseudonym("organization", "abc", key)
    assert "abc" not in _pseudonym("participant", "abc", key)


def test_parse_points_builds_pointer_events():
    events, reasons = _parse_points(
        [[0, 0.1, 0.2], [16, 0.2, 0.3], [32, 0.3, 0.4], [48, 0.4, 0.5]],
        400,
        200,
    )

    assert reasons == []
    assert events[0]["event_type"] == "pointerdown"
    assert events[-1]["event_type"] == "pointerup"
    assert events[1]["x"] == 80.0
    assert events[1]["y"] == 60.0


def test_parse_points_rejects_bad_trace():
    assert _parse_points(None, 400, 200)[1] == ["missing_or_invalid_trace"]
    assert _parse_points([[0, 0, 0]], 400, 200)[1] == ["too_few_points"]
    points = [[0, 0, 0], [16, 0.1, 0], [8, 0.2, 0], [32, 0.3, 0]]
    assert _parse_points(points, 400, 200)[1] == ["non_monotonic_time"]


def test_quality_flags_anonymous_and_extreme_speed():
    events, _ = _parse_points(
        [[0, 0.0, 0.0], [1, 0.5, 0.0], [2, 0.8, 0.0], [3, 1.0, 0.0]],
        400,
        200,
    )
    status, reasons = _quality_decision(events=events, parse_reasons=[], participant=None)

    assert status == "pending"
    assert reasons == ["participant_group_unknown", "extreme_normalized_speed_review"]
    assert _max_normalized_speed(events) > 0.02
