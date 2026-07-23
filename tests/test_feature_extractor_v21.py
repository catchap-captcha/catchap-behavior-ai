"""Tests for the speed-burst feature schema 2.1."""

from __future__ import annotations

import math

from app.services.feature_extractor_v21 import (
    FEATURE_NAMES,
    TRAJECTORY_ONLY_FEATURE_NAMES,
    V21_ADDITIONAL_FEATURES,
    extract_features,
)


def _ease_burst_events() -> list[dict]:
    x_values = [0.0, 0.04, 0.09, 0.15, 0.62, 0.9, 0.96, 1.0]
    return [
        {
            "seq": index,
            "event_type": "pointerdown" if index == 0 else "pointerup" if index == 7 else "pointermove",
            "t_ms": index * 20,
            "x_normalized": x,
            "y_normalized": 0.5,
            "x": x * 420,
            "y": 110,
        }
        for index, x in enumerate(x_values)
    ]


def test_v21_adds_finite_speed_burst_features():
    features = extract_features(_ease_burst_events(), {})

    assert len(FEATURE_NAMES) == len(features) == 49
    assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 44
    assert all(math.isfinite(features[name]) for name in V21_ADDITIONAL_FEATURES)
    assert features["mid_to_edge_speed_ratio"] > 1.0
    assert features["speed_burst_concentration"] > 1.0
    assert features["straight_burst_score"] > 1.0
