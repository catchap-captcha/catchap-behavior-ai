"""Unit tests for the 29-feature behavioral extractor."""

from __future__ import annotations

import math

from app.services.feature_extractor import FEATURE_NAMES, extract_features
from tests.conftest import bot_like_events, human_like_events


def _finite(feats):
    return all(math.isfinite(v) for v in feats.values())


def test_exactly_29_features_and_all_finite():
    feats = extract_features(human_like_events(), {})
    assert list(feats.keys()) == FEATURE_NAMES
    assert len(feats) == 29
    assert _finite(feats)


def test_single_coordinate_is_safe():
    events = [{"seq": 0, "event_type": "pointerdown", "t_ms": 0, "x": 5, "y": 5}]
    feats = extract_features(events, {})
    assert feats["event_count"] == 1.0
    assert feats["duration_ms"] == 0.0
    assert _finite(feats)


def test_empty_events_is_zero_vector():
    feats = extract_features([], {})
    assert feats["event_count"] == 0.0
    assert _finite(feats)


def test_straight_path_has_high_linearity():
    feats = extract_features(bot_like_events(), {})
    # perfectly straight -> displacement ~= total_distance -> linearity ~ 1
    assert feats["linearity"] > 0.99
    assert feats["y_deviation"] < 1e-6


def test_curved_path_has_lower_linearity_than_straight():
    human = extract_features(human_like_events(), {})
    bot = extract_features(bot_like_events(), {})
    assert human["linearity"] < bot["linearity"]
    assert human["y_deviation"] > 0.0


def test_same_timestamps_do_not_produce_infinity():
    events = [
        {"seq": 0, "event_type": "pointerdown", "t_ms": 0, "x": 0, "y": 0},
        {"seq": 1, "event_type": "pointermove", "t_ms": 0, "x": 10, "y": 0},
        {"seq": 2, "event_type": "pointerup", "t_ms": 0, "x": 20, "y": 0},
    ]
    feats = extract_features(events, {})
    assert _finite(feats)
    assert feats["max_speed"] == 0.0  # zero duration segments -> speed 0, not inf


def test_pause_detection():
    # move, then hold still for many samples, then move again
    events = [{"seq": 0, "event_type": "pointerdown", "t_ms": 0, "x": 0, "y": 0}]
    for i in range(1, 10):
        events.append({"seq": i, "event_type": "pointermove", "t_ms": 100 + i * 50, "x": 50, "y": 0})
    events.append({"seq": 10, "event_type": "pointerup", "t_ms": 800, "x": 100, "y": 0})
    feats = extract_features(events, {})
    assert feats["pause_count"] >= 1
    assert 0.0 <= feats["pause_ratio"] <= 1.0


def test_direction_change_counted():
    # zig-zag on x axis -> several >90 degree turns
    xs = [0, 30, 5, 35, 8, 40]
    events = []
    for i, x in enumerate(xs):
        etype = "pointerdown" if i == 0 else "pointerup" if i == len(xs) - 1 else "pointermove"
        events.append({"seq": i, "event_type": etype, "t_ms": i * 50, "x": x, "y": 0})
    feats = extract_features(events, {})
    assert feats["direction_changes"] >= 2


def test_interval_features_for_constant_cadence():
    feats = extract_features(bot_like_events(), {})
    # fixed 16ms cadence -> near-zero interval std and high duplicate ratio
    assert feats["interval_std_ms"] < 1e-6
    assert feats["duplicate_interval_ratio"] > 0.9


def test_interaction_features_passthrough():
    interaction = {
        "regrab_count": 2, "retry_count": 1, "pointercancel_count": 3,
        "empty_click_count": 4, "failed_drop_count": 5,
    }
    feats = extract_features(human_like_events(), interaction)
    assert feats["regrab_count"] == 2.0
    assert feats["failed_drop_count"] == 5.0


def test_correction_features_present_and_finite():
    # overshoot: go past x=100 then come back to 100
    xs = [0, 40, 80, 120, 110, 100]
    events = []
    for i, x in enumerate(xs):
        etype = "pointerdown" if i == 0 else "pointerup" if i == len(xs) - 1 else "pointermove"
        events.append({"seq": i, "event_type": etype, "t_ms": i * 60, "x": x, "y": 0})
    feats = extract_features(events, {})
    assert feats["overshoot_count"] >= 1
    assert feats["overshoot_distance"] > 0.0
    assert math.isfinite(feats["final_segment_speed"])
