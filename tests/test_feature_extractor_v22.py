"""Tests for temporal interpolation features in schema 2.2."""

from __future__ import annotations

import math

from app.services.feature_extractor_v22 import (
    FEATURE_NAMES,
    TRAJECTORY_ONLY_FEATURE_NAMES,
    V22_ADDITIONAL_FEATURES,
    extract_features,
)


def _smooth_timing_events() -> list[dict]:
    timestamps = [0, 5, 12, 21, 32, 45, 60]
    return [
        {
            "seq": index,
            "event_type": "pointerdown" if index == 0 else "pointerup" if index == 6 else "pointermove",
            "t_ms": timestamp,
            "x_normalized": index / 6,
            "y_normalized": 0.5,
            "x": 420 * index / 6,
            "y": 110,
        }
        for index, timestamp in enumerate(timestamps)
    ]


def test_v22_adds_finite_temporal_interpolation_features():
    features = extract_features(_smooth_timing_events(), {})

    assert len(FEATURE_NAMES) == len(features) == 52
    assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 47
    assert all(math.isfinite(features[name]) for name in V22_ADDITIONAL_FEATURES)
    assert features["interval_lag1_autocorrelation"] > 0.9
    assert features["interval_second_difference_relative"] == 0.0
