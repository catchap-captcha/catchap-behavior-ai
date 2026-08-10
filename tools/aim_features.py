"""Features for one aim segment.

Chosen against what the human data actually showed, not against what seemed
plausible. Measured over 185 human aim segments:

    peak_at         0.000   the fastest step is the first one
    slow_fraction   0.625   most of the trip is spent below a fifth of peak speed
    straightness    0.862
    speed_cv        1.190
    submovements    1

So a human aim arrives already moving and then crawls. That asymmetry is the
core of the schema; several features exist only to describe its shape from
different directions, because a bot can fake any single summary statistic and
matching all of them at once is the actual cost.

Scale handling: everything is computed on stage-normalized coordinates, and the
absolute magnitudes (duration, path length) are kept separate from the shape
features so a scale-sensitive experiment can drop them. Resizing the window is
free for an attacker; a defence resting on window size is not a defence.
"""

from __future__ import annotations

import numpy as np

from tools.aim_segments import speed_profile, submovements, to_arrays

SCHEMA_VERSION = "aim-burst v1"

# Magnitudes an attacker changes for free, or that depend on the display rather
# than the hand. Kept nameable so experiments can exclude them as a group.
MAGNITUDE_BOUND = ("duration_ms", "path_length", "straight_distance",
                   "peak_speed", "mean_speed", "point_count")


def extract(burst: list[dict]) -> dict[str, float]:
    xy, t = to_arrays(burst)
    speed = speed_profile(xy, t)
    n = speed.size
    if n < 2:
        return {}

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_len = float(steps.sum())
    straight = float(np.linalg.norm(xy[-1] - xy[0]))
    peak = float(speed.max()) or 1e-9
    intervals = np.diff(t)

    # Distance still to go, as a fraction, at every recorded moment. A human
    # closes most of the gap early; the tail of this curve is the endgame.
    remaining = np.linalg.norm(xy - xy[-1], axis=1)
    remaining = remaining / (remaining[0] + 1e-9)

    # Signed turning between consecutive steps. Humans drift and correct in one
    # direction more than the other; a symmetric curve does not.
    direction = np.diff(xy, axis=0)
    angles = np.arctan2(direction[:, 1], direction[:, 0])
    turn = np.diff(np.unwrap(angles)) if angles.size > 1 else np.zeros(1)

    accel = np.diff(speed) / np.maximum(intervals[1:] / 1000.0, 1e-6) if n > 1 else np.zeros(1)

    feats = {
        # --- magnitudes -------------------------------------------------
        "duration_ms": float(t[-1]),
        "point_count": float(len(burst)),
        "path_length": path_len,
        "straight_distance": straight,
        "peak_speed": peak,
        "mean_speed": float(speed.mean()),

        # --- the arrival/crawl asymmetry --------------------------------
        "peak_at": float(np.argmax(speed)) / max(n - 1, 1),
        "slow_fraction": float((speed < 0.2 * peak).mean()),
        "fast_fraction": float((speed > 0.6 * peak).mean()),
        # Time to cover the first half of the distance, as a fraction of total.
        # Ballistic arrival puts this well below 0.5.
        "half_distance_at": float(np.argmax(remaining <= 0.5) / max(n, 1)),
        "final_speed_ratio": float(speed[-1] / peak),
        "first_speed_ratio": float(speed[0] / peak),
        # Mean speed of the last quarter against the first quarter.
        "decel_ratio": float(speed[3 * n // 4:].mean() / (speed[:max(n // 4, 1)].mean() + 1e-9)),

        # --- shape ------------------------------------------------------
        "straightness": straight / path_len if path_len > 1e-9 else 0.0,
        "speed_cv": float(speed.std() / (speed.mean() + 1e-9)),
        "step_len_cv": float(steps.std() / (steps.mean() + 1e-9)),
        "submovement_count": float(len(submovements(speed))),
        "turn_abs_mean": float(np.abs(turn).mean()),
        "turn_signed_mean": float(turn.mean()),
        "turn_cv": float(np.abs(turn).std() / (np.abs(turn).mean() + 1e-9)),
        "accel_std": float(accel.std()),
        # A jerky hand reverses acceleration often; a spline almost never does.
        "accel_sign_flips": float((np.diff(np.sign(accel)) != 0).mean()) if accel.size > 1 else 0.0,

        # --- overshoot --------------------------------------------------
        # Did the pointer pass the target and come back? Humans do; a curve
        # fitted to end at the target does not.
        "overshoot": float(max(0.0, np.max(np.linalg.norm(xy - xy[-1], axis=1)[np.argmin(remaining > 0.15):]) / (straight + 1e-9))),

        # --- timing -----------------------------------------------------
        "interval_median_ms": float(np.median(intervals)),
        "interval_cv": float(intervals.std() / (intervals.mean() + 1e-9)),
        # The throttle floor is 40ms. What fraction of steps sit exactly on it
        # tells you whether the pointer was outrunning the sampler.
        "interval_at_floor": float((intervals <= 45.0).mean()),
    }
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in feats.items()}


FEATURE_NAMES: tuple[str, ...] = tuple(sorted({
    "duration_ms", "point_count", "path_length", "straight_distance", "peak_speed",
    "mean_speed", "peak_at", "slow_fraction", "fast_fraction", "half_distance_at",
    "final_speed_ratio", "first_speed_ratio", "decel_ratio", "straightness",
    "speed_cv", "step_len_cv", "submovement_count", "turn_abs_mean",
    "turn_signed_mean", "turn_cv", "accel_std", "accel_sign_flips", "overshoot",
    "interval_median_ms", "interval_cv", "interval_at_floor",
}))


def matrix(bursts: list[list[dict]], names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    rows = []
    for burst in bursts:
        feats = extract(burst)
        if feats:
            rows.append([feats.get(n, 0.0) for n in names])
    return np.nan_to_num(np.asarray(rows, dtype=float))
