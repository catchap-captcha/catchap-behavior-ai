"""Tests for the candidate Feature v2 profile."""

from __future__ import annotations

import math

from app.services.feature_extractor_v2 import (
    FEATURE_NAMES,
    TRAJECTORY_ONLY_FEATURE_NAMES,
    V2_ADDITIONAL_FEATURES,
    extract_features,
)
from tests.conftest import human_like_events


def test_v2_has_29_plus_15_finite_features():
    features = extract_features(human_like_events(), {})

    assert len(FEATURE_NAMES) == len(features) == 44
    assert len(V2_ADDITIONAL_FEATURES) == 15
    assert set(features) == set(FEATURE_NAMES)
    assert all(math.isfinite(value) for value in features.values())


def test_v2_trajectory_only_profile_excludes_ui_interaction_counters():
    assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 39
    assert not {
        "regrab_count",
        "retry_count",
        "pointercancel_count",
        "empty_click_count",
        "failed_drop_count",
    }.intersection(TRAJECTORY_ONLY_FEATURE_NAMES)


def test_v2_degenerate_trace_is_finite():
    features = extract_features([], {})

    assert all(math.isfinite(value) for value in features.values())
    assert all(features[name] == 0.0 for name in V2_ADDITIONAL_FEATURES)
