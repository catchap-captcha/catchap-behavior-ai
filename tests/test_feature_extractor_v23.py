"""Tests for trajectory physics-consistency features in schema 2.3."""

from __future__ import annotations

import math

from app.services.feature_extractor_v23 import (
    FEATURE_NAMES,
    TRAJECTORY_ONLY_FEATURE_NAMES,
    V23_ADDITIONAL_FEATURES,
    extract_features,
)


def _events() -> list[dict]:
    points = [(0.0, 0.0), (0.15, 0.0), (0.32, 0.10), (0.52, 0.30), (0.70, 0.65), (1.0, 1.0)]
    timestamps = [0, 15, 45, 150, 170, 310]
    return [
        {
            "seq": index,
            "event_type": "pointerdown" if index == 0 else "pointerup" if index == 5 else "pointermove",
            "t_ms": timestamps[index],
            "x_normalized": x,
            "y_normalized": y,
            "x": x * 400,
            "y": y * 300,
        }
        for index, (x, y) in enumerate(points)
    ]


def test_v23_adds_finite_trajectory_physics_features():
    features = extract_features(_events(), {})

    assert len(FEATURE_NAMES) == len(features) == 55
    assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 50
    assert all(math.isfinite(features[name]) for name in V23_ADDITIONAL_FEATURES)
    assert abs(features["speed_turn_abs_correlation"]) > 0.01
    assert features["turn_change_smoothness"] >= 0.0
