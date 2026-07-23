"""Feature schema 2.3 with trajectory physics-consistency signals.

The new signals describe relationships inside one trajectory rather than a
specific red-team generator. They are intended to generalize to bot families
that were not part of the detector's fitted bot labels.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from app.services.feature_extractor_v2 import _arrays
from app.services.feature_extractor_v22 import (
    FEATURE_NAMES as V22_FEATURE_NAMES,
    TRAJECTORY_ONLY_FEATURE_NAMES as V22_TRAJECTORY_ONLY_FEATURE_NAMES,
    extract_features as extract_v22_features,
)


FEATURE_SCHEMA_VERSION = "2.3"
V23_ADDITIONAL_FEATURES = [
    "speed_turn_abs_correlation",
    "turn_change_smoothness",
    "pause_position_entropy",
]
FEATURE_NAMES = [*V22_FEATURE_NAMES, *V23_ADDITIONAL_FEATURES]
TRAJECTORY_ONLY_FEATURE_NAMES = [
    *V22_TRAJECTORY_ONLY_FEATURE_NAMES,
    *V23_ADDITIONAL_FEATURES,
]
assert len(FEATURE_NAMES) == 55
assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 50


def _normalized_entropy(values: np.ndarray, bins: int) -> float:
    if values.size < 2 or np.allclose(values, values[0]):
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(bins) if bins > 1 else 0.0


def _trajectory_physics_features(events: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(events)
    output = {name: 0.0 for name in V23_ADDITIONAL_FEATURES}
    t_ms, points = _arrays(rows)
    if points.shape[0] < 4:
        return output

    vectors = np.diff(points, axis=0)
    distances = np.linalg.norm(vectors, axis=1)
    intervals = np.clip(np.diff(t_ms), 0.0, None)
    speed = np.divide(distances, intervals, out=np.zeros_like(distances), where=intervals > 0)

    moving_vectors = vectors[distances > 1e-9]
    moving_indices = np.flatnonzero(distances > 1e-9)
    if moving_vectors.shape[0] >= 3:
        angles = np.arctan2(moving_vectors[:, 1], moving_vectors[:, 0])
        absolute_turns = np.abs(np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles))))
        if absolute_turns.size >= 2:
            output["turn_change_smoothness"] = float(np.mean(np.abs(np.diff(absolute_turns))))

            # A person commonly slows while changing direction. This pairs each
            # turn with the segment that follows it, even when zero-length events
            # were present in the raw trace.
            following_speed = speed[moving_indices[1:]]
            if following_speed.size == absolute_turns.size:
                centered_speed = following_speed - following_speed.mean()
                centered_turns = absolute_turns - absolute_turns.mean()
                denominator = float(np.linalg.norm(centered_speed) * np.linalg.norm(centered_turns))
                if denominator > 1e-12:
                    output["speed_turn_abs_correlation"] = float(
                        np.clip(np.dot(centered_speed, centered_turns) / denominator, -1.0, 1.0)
                    )

    if intervals.size:
        pause_threshold = max(100.0, float(np.percentile(intervals, 90)))
        pause_indices = np.flatnonzero(intervals >= pause_threshold)
        if pause_indices.size >= 2:
            cumulative_distance = np.concatenate(([0.0], np.cumsum(distances)))
            total_distance = float(cumulative_distance[-1])
            if total_distance > 1e-12:
                pause_progress = cumulative_distance[pause_indices + 1] / total_distance
                output["pause_position_entropy"] = _normalized_entropy(pause_progress, bins=4)

    return {
        name: float(value) if math.isfinite(float(value)) else 0.0
        for name, value in output.items()
    }


def extract_features(
    events: Iterable[dict[str, Any]], interaction: dict[str, Any] | None = None
) -> dict[str, float]:
    rows = list(events)
    return {
        **extract_v22_features(rows, interaction),
        **_trajectory_physics_features(rows),
    }


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "TRAJECTORY_ONLY_FEATURE_NAMES",
    "V23_ADDITIONAL_FEATURES",
    "extract_features",
]
