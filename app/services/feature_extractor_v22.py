"""Feature schema 2.2 with temporal interpolation signals for trajectory bots."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from app.services.feature_extractor_v2 import _arrays
from app.services.feature_extractor_v21 import (
    FEATURE_NAMES as V21_FEATURE_NAMES,
    TRAJECTORY_ONLY_FEATURE_NAMES as V21_TRAJECTORY_ONLY_FEATURE_NAMES,
    extract_features as extract_v21_features,
)


FEATURE_SCHEMA_VERSION = "2.2"
V22_ADDITIONAL_FEATURES = [
    "interval_lag1_autocorrelation",
    "interval_delta_lag1_autocorrelation",
    "interval_second_difference_relative",
]
FEATURE_NAMES = [*V21_FEATURE_NAMES, *V22_ADDITIONAL_FEATURES]
TRAJECTORY_ONLY_FEATURE_NAMES = [
    *V21_TRAJECTORY_ONLY_FEATURE_NAMES,
    *V22_ADDITIONAL_FEATURES,
]
assert len(FEATURE_NAMES) == 52
assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 47


def _lag1_autocorrelation(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    left = values[:-1] - values[:-1].mean()
    right = values[1:] - values[1:].mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _temporal_interpolation_features(events: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(events)
    output = {name: 0.0 for name in V22_ADDITIONAL_FEATURES}
    t_ms, _ = _arrays(rows)
    if t_ms.size < 4:
        return output

    intervals = np.clip(np.diff(t_ms), 0.0, None)
    positive_intervals = intervals[intervals > 0.0]
    if positive_intervals.size < 3:
        return output

    output["interval_lag1_autocorrelation"] = _lag1_autocorrelation(positive_intervals)
    interval_delta = np.diff(positive_intervals)
    output["interval_delta_lag1_autocorrelation"] = _lag1_autocorrelation(interval_delta)
    if positive_intervals.size >= 3:
        second_difference = np.diff(positive_intervals, n=2)
        scale = max(float(positive_intervals.mean()), 1.0)
        output["interval_second_difference_relative"] = float(
            np.mean(np.abs(second_difference)) / scale
        )

    return {
        name: float(value) if math.isfinite(float(value)) else 0.0
        for name, value in output.items()
    }


def extract_features(
    events: Iterable[dict[str, Any]], interaction: dict[str, Any] | None = None
) -> dict[str, float]:
    rows = list(events)
    return {
        **extract_v21_features(rows, interaction),
        **_temporal_interpolation_features(rows),
    }


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "TRAJECTORY_ONLY_FEATURE_NAMES",
    "V22_ADDITIONAL_FEATURES",
    "extract_features",
]
