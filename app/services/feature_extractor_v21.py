"""Feature schema 2.1 with local speed-burst trajectory signals."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from app.services.feature_extractor_v2 import (
    FEATURE_NAMES as V20_FEATURE_NAMES,
    TRAJECTORY_ONLY_FEATURE_NAMES as V20_TRAJECTORY_ONLY_FEATURE_NAMES,
    _arrays,
    extract_features as extract_v20_features,
)


FEATURE_SCHEMA_VERSION = "2.1"
V21_ADDITIONAL_FEATURES = [
    "speed_peak_position_ratio",
    "mid_to_edge_speed_ratio",
    "speed_burst_concentration",
    "peak_accel_decel_symmetry",
    "straight_burst_score",
]
FEATURE_NAMES = [*V20_FEATURE_NAMES, *V21_ADDITIONAL_FEATURES]
TRAJECTORY_ONLY_FEATURE_NAMES = [
    *V20_TRAJECTORY_ONLY_FEATURE_NAMES,
    *V21_ADDITIONAL_FEATURES,
]
assert len(FEATURE_NAMES) == 49
assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 44


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _speed_burst_features(events: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(events)
    output = {name: 0.0 for name in V21_ADDITIONAL_FEATURES}
    t_ms, points = _arrays(rows)
    if points.shape[0] < 4:
        return output

    vectors = np.diff(points, axis=0)
    distances = np.linalg.norm(vectors, axis=1)
    dt = np.clip(np.diff(t_ms), 0.0, None)
    speed = np.divide(distances, dt, out=np.zeros_like(distances), where=dt > 0)
    if speed.size < 3 or np.all(speed <= 1e-12):
        return output

    peak_index = int(np.argmax(speed))
    output["speed_peak_position_ratio"] = peak_index / max(speed.size - 1, 1)

    count = speed.size
    early = speed[: max(1, int(math.ceil(count * 0.25)))]
    middle = speed[int(math.floor(count * 0.35)) : max(1, int(math.ceil(count * 0.65)))]
    late = speed[max(0, int(math.floor(count * 0.75))) :]
    edge_speed = (_mean(early) + _mean(late)) / 2.0
    middle_speed = _mean(middle)
    output["mid_to_edge_speed_ratio"] = middle_speed / max(edge_speed, 1e-9)

    window = max(2, int(math.ceil(count * 0.20)))
    rolling = np.convolve(speed, np.ones(window) / window, mode="valid")
    output["speed_burst_concentration"] = float(rolling.max() / max(_mean(speed), 1e-9))

    rising = np.diff(speed[: peak_index + 1])
    falling = -np.diff(speed[peak_index:])
    rise_strength = _mean(np.maximum(rising, 0.0))
    fall_strength = _mean(np.maximum(falling, 0.0))
    if rise_strength > 1e-12 or fall_strength > 1e-12:
        output["peak_accel_decel_symmetry"] = min(rise_strength, fall_strength) / max(
            rise_strength, fall_strength, 1e-12
        )

    displacement = float(np.linalg.norm(points[-1] - points[0]))
    path_length = float(distances.sum())
    linearity = displacement / path_length if path_length > 1e-12 else 0.0
    output["straight_burst_score"] = linearity * output["speed_burst_concentration"]
    return {
        name: float(value) if math.isfinite(float(value)) else 0.0
        for name, value in output.items()
    }


def extract_features(
    events: Iterable[dict[str, Any]], interaction: dict[str, Any] | None = None
) -> dict[str, float]:
    rows = list(events)
    return {
        **extract_v20_features(rows, interaction),
        **_speed_burst_features(rows),
    }


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "TRAJECTORY_ONLY_FEATURE_NAMES",
    "V21_ADDITIONAL_FEATURES",
    "extract_features",
]
