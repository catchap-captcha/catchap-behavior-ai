"""Behavioral feature extraction for web drag CAPTCHA attempts.

This module is the single source of truth for the 29-feature behavioral vector.
Every consumer (collect API, predict API, training pipeline, tests) imports the
feature list and the extraction function from here so the contract can never
drift.

Design rules honoured throughout:
  * No division by zero — every ratio guards its denominator.
  * Duplicate / equal timestamps are tolerated (dt clamped to >= 0).
  * Missing or degenerate input yields deterministic, finite numbers (0.0),
    never NaN/Infinity.
  * Features are computed purely from the raw pointer events + interaction
    summary, so they can be re-derived at any time from stored raw data.

Units:
  * Distances are in CAPTCHA-area pixels (the `x`, `y` fields).
  * Time is in milliseconds (`t_ms`).
  * Speed is pixels per millisecond (px/ms).
  * Acceleration is px/ms^2; jerk is px/ms^3.

Note: `position_correct`, `interaction_success` and `final_drop_error` are
CAPTCHA pass/fail signals, NOT behavioral features, and are deliberately absent.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

# Canonical feature-schema version. `config.feature_schema_version` mirrors this;
# training refuses to mix rows produced under a different version.
FEATURE_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Feature name groups (order is part of the contract)
# ---------------------------------------------------------------------------
BASIC_FEATURES = [
    "event_count",
    "duration_ms",
    "total_distance",
    "displacement",
    "avg_speed",
    "max_speed",
    "speed_std",
    "avg_acceleration",
    "max_acceleration",
    "jerk_mean",
    "direction_changes",
    "pause_count",
    "pause_ratio",
    "linearity",
    "y_deviation",
]
INTERVAL_FEATURES = [
    "interval_mean_ms",
    "interval_std_ms",
    "interval_cv",
    "duplicate_interval_ratio",
]
CORRECTION_FEATURES = [
    "overshoot_count",
    "overshoot_distance",
    "correction_count",
    "endpoint_adjustment_time",
    "final_segment_speed",
]
INTERACTION_FEATURES = [
    "regrab_count",
    "retry_count",
    "pointercancel_count",
    "empty_click_count",
    "failed_drop_count",
]

FEATURE_NAMES: list[str] = (
    BASIC_FEATURES + INTERVAL_FEATURES + CORRECTION_FEATURES + INTERACTION_FEATURES
)
assert len(FEATURE_NAMES) == 29, "the behavioral vector must contain exactly 29 features"

# These features are derivable from a single ordered pointer trace ``(x, y, t)``.
# Interaction summary counters deliberately stay out of the trajectory-only model.
TRAJECTORY_ONLY_FEATURE_NAMES: list[str] = (
    BASIC_FEATURES + INTERVAL_FEATURES + CORRECTION_FEATURES
)
assert len(TRAJECTORY_ONLY_FEATURE_NAMES) == 24

# Columns that must never enter a model as an input feature. Used by the
# training pipeline to strip identifiers, provenance, labels and pass/fail
# signals from the dataset.
MODEL_INPUT_EXCLUDE_COLUMNS = [
    "attempt_id",
    "challenge_id",
    "session_id",
    "anonymous_participant_id",
    "label",
    "label_source",
    "bot_family",
    "generator_version",
    "schema_version",
    "feature_schema_version",
    "position_correct",
    "interaction_success",
    "final_drop_error",
    "human_score",
    "bot_risk_score",
    "bot_decision",
    "risk_score",
    "risk_level",
    "recommended_action",
    "risk_reasons",
    "model_version",
]

# --- tuning constants (documented, kept explicit for reproducibility) ---
_PAUSE_DIST_PX = 1.0          # a segment moving less than this counts as a pause
_DIR_CHANGE_MIN_DIST_PX = 1.0  # segments shorter than this are ignored for turns
_FINAL_SEGMENT_RATIO = 0.25    # last 25% of the path == "near target" region
_EPS = 1e-9                    # generic denominator guard


def _zero_vector() -> dict[str, float]:
    """A fully-zero, finite feature vector (used for degenerate input)."""
    return {name: 0.0 for name in FEATURE_NAMES}


def _to_arrays(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_ms, x, y) float arrays for the ordered trajectory points.

    Only events carrying coordinates are used, sorted by `seq`. Values are
    coerced to float; rows with non-finite coordinates are dropped so downstream
    maths never sees NaN/Infinity.
    """
    pts = sorted(events, key=lambda e: e.get("seq", 0))
    t, x, y = [], [], []
    for e in pts:
        try:
            ti = float(e["t_ms"])
            xi = float(e["x"])
            yi = float(e["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(ti) and math.isfinite(xi) and math.isfinite(yi)):
            continue
        t.append(ti)
        x.append(xi)
        y.append(yi)
    return np.asarray(t, float), np.asarray(x, float), np.asarray(y, float)


def _segment_speeds(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dist, dt, speed) per consecutive segment.

    `dt` is clamped to >= 0 (duplicate/equal timestamps -> 0). Segments with
    dt == 0 get speed 0 rather than infinity, so equal timestamps are safe.
    """
    dist = np.hypot(np.diff(x), np.diff(y))
    dt = np.clip(np.diff(t), 0.0, None)
    speed = np.where(dt > 0, dist / np.where(dt > 0, dt, 1.0), 0.0)
    return dist, dt, speed


def _basic_features(t, x, y) -> dict[str, float]:
    """Compute the 15 basic behavioral features. See module docstring for units."""
    n = int(len(x))
    out = {name: 0.0 for name in BASIC_FEATURES}
    out["event_count"] = float(n)
    if n < 2:
        return out

    dist, dt, speed = _segment_speeds(t, x, y)
    duration = float(t[-1] - t[0])
    total_distance = float(dist.sum())
    displacement = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))

    out["duration_ms"] = max(duration, 0.0)
    out["total_distance"] = total_distance
    out["displacement"] = displacement
    out["avg_speed"] = total_distance / (duration + _EPS) if duration > 0 else 0.0
    out["max_speed"] = float(speed.max()) if speed.size else 0.0
    out["speed_std"] = float(speed.std()) if speed.size else 0.0

    # acceleration = change in speed over time; jerk = change in acceleration
    if speed.size >= 2:
        dt_mid = np.clip(dt[1:], _EPS, None)
        accel = np.diff(speed) / dt_mid
        out["avg_acceleration"] = float(np.abs(accel).mean())
        out["max_acceleration"] = float(np.abs(accel).max())
        if accel.size >= 2:
            jerk = np.diff(accel) / np.clip(dt[2:], _EPS, None)
            out["jerk_mean"] = float(np.abs(jerk).mean())

    # direction changes: count turns > 90 degrees between consecutive movement
    # vectors, ignoring near-stationary micro-segments.
    out["direction_changes"] = float(_count_direction_changes(x, y))

    # pauses: segments that barely move
    is_pause = dist < _PAUSE_DIST_PX
    out["pause_count"] = float(int(is_pause.sum()))
    paused_time = float(dt[is_pause].sum())
    out["pause_ratio"] = paused_time / (duration + _EPS) if duration > 0 else 0.0

    # linearity: how straight the path is (1.0 == perfectly straight)
    out["linearity"] = displacement / (total_distance + _EPS) if total_distance > 0 else 0.0

    # y_deviation: max perpendicular distance from the start->end straight line
    out["y_deviation"] = float(_max_perpendicular_deviation(x, y))
    return out


def _count_direction_changes(x: np.ndarray, y: np.ndarray) -> int:
    """Number of >90 degree turns between successive movement vectors.

    Consecutive movement vectors whose dot product is negative represent a turn
    sharper than 90 degrees. Micro-segments below `_DIR_CHANGE_MIN_DIST_PX` are
    skipped so sensor jitter is not counted as a real direction change.
    """
    vecs = []
    for i in range(1, len(x)):
        dx, dy = x[i] - x[i - 1], y[i] - y[i - 1]
        if math.hypot(dx, dy) >= _DIR_CHANGE_MIN_DIST_PX:
            vecs.append((dx, dy))
    changes = 0
    for i in range(1, len(vecs)):
        ax, ay = vecs[i - 1]
        bx, by = vecs[i]
        if (ax * bx + ay * by) < 0:  # angle > 90 degrees
            changes += 1
    return changes


def _max_perpendicular_deviation(x: np.ndarray, y: np.ndarray) -> float:
    """Max perpendicular distance of any point from the start->end line.

    If start and end coincide, falls back to the max distance from the start
    point. Always finite and >= 0.
    """
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    line_len = math.hypot(x1 - x0, y1 - y0)
    if line_len < _EPS:
        return float(np.hypot(x - x0, y - y0).max())
    # perpendicular distance via 2D cross product magnitude / line length
    cross = np.abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0))
    return float((cross / line_len).max())


def _interval_features(t: np.ndarray) -> dict[str, float]:
    """Compute the 4 inter-event-interval features.

    duplicate_interval_ratio = fraction of intervals whose exact value repeats.
    A scripted bot with a fixed 16ms cadence pushes this toward 1.0.
    """
    out = {name: 0.0 for name in INTERVAL_FEATURES}
    if len(t) < 2:
        return out
    intervals = np.clip(np.diff(t), 0.0, None)
    if intervals.size == 0:
        return out
    mean = float(intervals.mean())
    std = float(intervals.std())
    out["interval_mean_ms"] = mean
    out["interval_std_ms"] = std
    out["interval_cv"] = std / (mean + _EPS) if mean > 0 else 0.0

    values, counts = np.unique(intervals, return_counts=True)
    duplicated = int(counts[counts > 1].sum())
    out["duplicate_interval_ratio"] = duplicated / (intervals.size + _EPS)
    return out


def _correction_features(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute the 5 target-approach / correction features.

    Movement is projected onto the dominant start->end axis; "overshoot" means
    travelling past the endpoint's projection and coming back. Corrections are
    small direction reversals within the final `_FINAL_SEGMENT_RATIO` of the
    path, characteristic of a human fine-tuning the drop position.
    """
    out = {name: 0.0 for name in CORRECTION_FEATURES}
    n = len(x)
    if n < 2:
        return out

    # unit axis from start to end
    ax, ay = x[-1] - x[0], y[-1] - y[0]
    axis_len = math.hypot(ax, ay)
    if axis_len < _EPS:
        proj = np.hypot(x - x[0], y - y[0])
        target_proj = 0.0
    else:
        ux, uy = ax / axis_len, ay / axis_len
        proj = (x - x[0]) * ux + (y - y[0]) * uy
        target_proj = float(axis_len)

    beyond = proj - target_proj
    over_mask = beyond > _PAUSE_DIST_PX
    # count overshoot episodes (contiguous runs of "beyond target")
    episodes, prev = 0, False
    for flag in over_mask:
        if flag and not prev:
            episodes += 1
        prev = bool(flag)
    out["overshoot_count"] = float(episodes)
    out["overshoot_distance"] = float(beyond[over_mask].max()) if over_mask.any() else 0.0

    # final segment: points within the last quarter of the projected axis
    if axis_len >= _EPS:
        threshold = target_proj * (1.0 - _FINAL_SEGMENT_RATIO)
        final_idx = np.where(proj >= threshold)[0]
    else:
        final_idx = np.arange(max(0, n - max(2, n // 4)), n)
    if final_idx.size >= 2:
        fx, fy = x[final_idx], y[final_idx]
        ft = t[final_idx]
        out["correction_count"] = float(_count_direction_changes(fx, fy))
        out["endpoint_adjustment_time"] = max(float(ft[-1] - ft[0]), 0.0)
        dist, dt, speed = _segment_speeds(ft, fx, fy)
        out["final_segment_speed"] = float(speed.mean()) if speed.size else 0.0
    return out


def _interaction_features(interaction: dict[str, Any] | None) -> dict[str, float]:
    """The 5 interaction-summary counters (passed through, coerced to float)."""
    src = interaction or {}
    out = {}
    for name in INTERACTION_FEATURES:
        try:
            val = float(src.get(name, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        out[name] = val if math.isfinite(val) else 0.0
    return out


def extract_features(
    events: Iterable[dict[str, Any]],
    interaction: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute the full 29-feature vector for one attempt.

    Args:
        events: raw pointer events (each a dict with seq, t_ms, x, y, ...).
        interaction: the interaction-summary counters for the attempt.

    Returns:
        A dict of exactly 29 finite float features keyed by `FEATURE_NAMES`.
        Degenerate input (0 or 1 usable points) yields a zero vector plus the
        interaction counters, never NaN/Infinity.
    """
    ev = list(events)
    t, x, y = _to_arrays(ev)

    feats = _zero_vector()
    feats.update(_interaction_features(interaction))

    if len(x) == 0:
        return _sanitize(feats)
    if len(x) == 1:
        feats["event_count"] = 1.0
        return _sanitize(feats)

    feats.update(_basic_features(t, x, y))
    feats.update(_interval_features(t))
    feats.update(_correction_features(t, x, y))
    return _sanitize(feats)


def _sanitize(feats: dict[str, float]) -> dict[str, float]:
    """Force every feature to a finite float and enforce the 29-key contract."""
    clean = {}
    for name in FEATURE_NAMES:
        val = feats.get(name, 0.0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        clean[name] = val if math.isfinite(val) else 0.0
    return clean
