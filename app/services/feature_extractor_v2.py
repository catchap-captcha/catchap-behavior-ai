"""Experimental 44-feature trajectory profile.

Version 2 keeps the production-compatible 29 features and adds 15 normalized
shape, timing, and submovement signals. It remains candidate-only until the
same held-out security gates show a material improvement over version 1.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from app.services.feature_extractor import FEATURE_NAMES as V1_FEATURE_NAMES
from app.services.feature_extractor import (
    TRAJECTORY_ONLY_FEATURE_NAMES as V1_TRAJECTORY_ONLY_FEATURE_NAMES,
)
from app.services.feature_extractor import extract_features as extract_v1_features


FEATURE_SCHEMA_VERSION = "2.0"
V2_ADDITIONAL_FEATURES = [
    "normalized_path_length",
    "normalized_speed_p10",
    "normalized_speed_p50",
    "normalized_speed_p90",
    "normalized_acceleration_std",
    "normalized_jerk_std",
    "turn_angle_mean",
    "turn_angle_std",
    "turn_angle_p90",
    "turn_direction_change_ratio",
    "micro_move_ratio",
    "dwell_burst_count",
    "timing_entropy",
    "spatial_entropy",
    "speed_peak_count",
]
FEATURE_NAMES = [*V1_FEATURE_NAMES, *V2_ADDITIONAL_FEATURES]
assert len(FEATURE_NAMES) == 44

# 24 base trajectory signals + 15 v2 shape/timing signals.  The five UI
# interaction counters in the v1 schema are intentionally excluded.
TRAJECTORY_ONLY_FEATURE_NAMES = [
    *V1_TRAJECTORY_ONLY_FEATURE_NAMES,
    *V2_ADDITIONAL_FEATURES,
]
assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 39


def _arrays(events: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    for event in sorted(events, key=lambda item: item.get("seq", 0)):
        try:
            t_ms = float(event["t_ms"])
            x = float(event.get("x_normalized", event["x"]))
            y = float(event.get("y_normalized", event["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(t_ms) and math.isfinite(x) and math.isfinite(y):
            rows.append((t_ms, x, y))
    if not rows:
        return np.zeros(0, dtype=float), np.zeros((0, 2), dtype=float)

    array = np.asarray(rows, dtype=float)
    points = array[:, 1:]
    if not all("x_normalized" in event and "y_normalized" in event for event in events):
        for dimension in range(2):
            low = float(points[:, dimension].min())
            span = float(points[:, dimension].max() - low)
            points[:, dimension] = (points[:, dimension] - low) / span if span > 1e-12 else 0.0
    return array[:, 0], points


def _normalized_entropy(values: np.ndarray, bins: int) -> float:
    if values.size < 2 or np.allclose(values, values[0]):
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(bins) if bins > 1 else 0.0


def _spatial_entropy(points: np.ndarray, bins: int = 4) -> float:
    if points.shape[0] < 2:
        return 0.0
    normalized = points.copy()
    for dimension in range(2):
        low = float(normalized[:, dimension].min())
        span = float(normalized[:, dimension].max() - low)
        normalized[:, dimension] = (
            (normalized[:, dimension] - low) / span if span > 1e-12 else 0.0
        )
    x_bin = np.clip((normalized[:, 0] * bins).astype(int), 0, bins - 1)
    y_bin = np.clip((normalized[:, 1] * bins).astype(int), 0, bins - 1)
    cells = x_bin * bins + y_bin
    counts = np.bincount(cells, minlength=bins * bins)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(bins * bins)


def _count_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))


def _additional_features(events: Iterable[dict[str, Any]]) -> dict[str, float]:
    t_ms, points = _arrays(list(events))
    output = {name: 0.0 for name in V2_ADDITIONAL_FEATURES}
    if points.shape[0] < 2:
        return output

    vectors = np.diff(points, axis=0)
    distances = np.linalg.norm(vectors, axis=1)
    dt = np.clip(np.diff(t_ms), 0.0, None)
    moving_time = np.where(dt > 0, dt, 1.0)
    speed = np.where(dt > 0, distances / moving_time, 0.0)

    output["normalized_path_length"] = float(distances.sum())
    if speed.size:
        p10, p50, p90 = np.percentile(speed, [10, 50, 90])
        output["normalized_speed_p10"] = float(p10)
        output["normalized_speed_p50"] = float(p50)
        output["normalized_speed_p90"] = float(p90)

    acceleration = np.diff(speed) / np.clip(dt[1:], 1e-9, None) if speed.size >= 2 else np.zeros(0)
    jerk = (
        np.diff(acceleration) / np.clip(dt[2:], 1e-9, None)
        if acceleration.size >= 2
        else np.zeros(0)
    )
    output["normalized_acceleration_std"] = float(acceleration.std()) if acceleration.size else 0.0
    output["normalized_jerk_std"] = float(jerk.std()) if jerk.size else 0.0

    moving = distances > 1e-9
    angles = np.arctan2(vectors[moving, 1], vectors[moving, 0])
    turns = np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles)))
    absolute_turns = np.abs(turns)
    if turns.size:
        output["turn_angle_mean"] = float(absolute_turns.mean())
        output["turn_angle_std"] = float(absolute_turns.std())
        output["turn_angle_p90"] = float(np.percentile(absolute_turns, 90))
        signs = np.sign(turns[np.abs(turns) > 1e-6])
        if signs.size >= 2:
            output["turn_direction_change_ratio"] = float(np.mean(signs[1:] != signs[:-1]))

    positive_distances = distances[distances > 0]
    micro_threshold = (
        max(0.002, float(np.percentile(positive_distances, 25)) * 0.5)
        if positive_distances.size
        else 0.002
    )
    output["micro_move_ratio"] = float(np.mean(distances <= micro_threshold))

    dwell_threshold = max(100.0, float(np.percentile(dt, 90))) if dt.size else 100.0
    output["dwell_burst_count"] = float(_count_runs(dt >= dwell_threshold))
    output["timing_entropy"] = _normalized_entropy(dt, bins=8)
    output["spatial_entropy"] = _spatial_entropy(points)

    if speed.size >= 3:
        peak_floor = float(np.median(speed) + speed.std())
        peaks = (speed[1:-1] > speed[:-2]) & (speed[1:-1] >= speed[2:])
        output["speed_peak_count"] = float(np.sum(peaks & (speed[1:-1] > peak_floor)))

    return {
        name: float(value) if math.isfinite(float(value)) else 0.0
        for name, value in output.items()
    }


def extract_features(
    events: Iterable[dict[str, Any]], interaction: dict[str, Any] | None = None
) -> dict[str, float]:
    event_rows = list(events)
    return {
        **extract_v1_features(event_rows, interaction),
        **_additional_features(event_rows),
    }


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "TRAJECTORY_ONLY_FEATURE_NAMES",
    "V2_ADDITIONAL_FEATURES",
    "extract_features",
]
