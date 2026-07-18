"""Replay / session-abuse feature computation.

These signals are deliberately kept OUT of the first-stage behavioral model.
They describe *relationships between attempts* (is this a replay of an earlier
drag? how fast is this session firing attempts?) rather than the shape of a
single drag, and are combined only at the final risk-fusion step.

The similarity backend is pluggable behind :class:`PathComparator` so the
DTW implementation used today can be swapped for an
approximate-nearest-neighbour index later without touching callers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


class PathComparator(Protocol):
    """Strategy interface for path similarity (0.0 = unrelated, 1.0 = identical)."""

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...


class NormalizedPathComparator:
    """Default comparator: resample both paths to a fixed length, compare.

    Paths are resampled to `n_points`, min-max normalized per axis, and scored
    with ``1 / (1 + mean_euclidean_distance)``. Cheap, order-preserving, and
    good enough to catch literal replays. Swap for DTW/ANN later.
    """

    def __init__(self, n_points: int = 32) -> None:
        self.n_points = n_points

    def _prepare(self, path: np.ndarray) -> np.ndarray:
        if path.shape[0] < 2:
            return np.zeros((self.n_points, 2), float)
        idx = np.linspace(0, path.shape[0] - 1, self.n_points)
        resampled = np.stack(
            [np.interp(idx, np.arange(path.shape[0]), path[:, d]) for d in range(2)],
            axis=1,
        )
        for d in range(2):
            col = resampled[:, d]
            span = col.max() - col.min()
            resampled[:, d] = (col - col.min()) / span if span > 1e-9 else 0.0
        return resampled

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if not _is_valid_path(a) or not _is_valid_path(b):
            return 0.0
        pa, pb = self._prepare(a), self._prepare(b)
        dist = float(np.linalg.norm(pa - pb, axis=1).mean())
        return 1.0 / (1.0 + dist)


class DynamicTimeWarpingComparator:
    """Compare path geometry while tolerating resampling and simple warps.

    Each path is translated to its own origin and divided by total path length,
    making the score invariant to translation and uniform spatial scaling.
    Dynamic time warping then aligns paths with different event counts. This is
    useful for replay variants that alter timing or drop/interpolate events.
    """

    def __init__(self, max_points: int = 96) -> None:
        if max_points < 2:
            raise ValueError("max_points must be at least 2")
        self.max_points = max_points

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        pa = self._prepare(a)
        pb = self._prepare(b)
        if pa.shape[0] < 2 or pb.shape[0] < 2:
            return 0.0

        n, m = len(pa), len(pb)
        costs = np.full((n + 1, m + 1), np.inf, dtype=float)
        steps = np.zeros((n + 1, m + 1), dtype=np.int32)
        costs[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                predecessors = (
                    (costs[i - 1, j], steps[i - 1, j]),
                    (costs[i, j - 1], steps[i, j - 1]),
                    (costs[i - 1, j - 1], steps[i - 1, j - 1]),
                )
                previous_cost, previous_steps = min(predecessors, key=lambda item: item[0])
                local_cost = float(np.linalg.norm(pa[i - 1] - pb[j - 1]))
                costs[i, j] = previous_cost + local_cost
                steps[i, j] = previous_steps + 1

        mean_cost = costs[n, m] / max(int(steps[n, m]), 1)
        return float(np.clip(1.0 / (1.0 + mean_cost), 0.0, 1.0))

    def _prepare(self, path: np.ndarray) -> np.ndarray:
        clean = _clean_path(path)
        if clean.shape[0] < 2:
            return clean
        if clean.shape[0] > self.max_points:
            clean = _arc_length_resample(clean, self.max_points)

        relative = clean - clean[0]
        total_length = float(np.linalg.norm(np.diff(relative, axis=0), axis=1).sum())
        if total_length <= 1e-12:
            return np.zeros((0, 2), dtype=float)
        return relative / total_length


def trace_fingerprint(path: np.ndarray, decimals: int = 6) -> str | None:
    """Hash the ordered relative coordinates of a valid path.

    Translation is ignored, while changed scale, geometry, event count, or
    order produces another fingerprint. Timing is intentionally excluded and
    is handled by separate replay features.
    """
    clean = _clean_path(path, remove_consecutive_duplicates=False)
    if clean.shape[0] < 2:
        return None
    relative = np.round(clean - clean[0], decimals=decimals)
    relative[np.abs(relative) < 10 ** (-decimals)] = 0.0
    digest = hashlib.sha256()
    digest.update(np.asarray(relative.shape, dtype=np.int64).tobytes())
    digest.update(relative.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _is_valid_path(path: np.ndarray) -> bool:
    arr = np.asarray(path)
    return bool(
        arr.ndim == 2
        and arr.shape[0] >= 2
        and arr.shape[1] == 2
        and np.isfinite(arr).all()
    )


def _clean_path(
    path: np.ndarray,
    *,
    remove_consecutive_duplicates: bool = True,
) -> np.ndarray:
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if remove_consecutive_duplicates and arr.shape[0] > 1:
        keep = np.concatenate(([True], np.any(np.diff(arr, axis=0) != 0.0, axis=1)))
        arr = arr[keep]
    return arr


def _arc_length_resample(path: np.ndarray, n_points: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 1e-12:
        return np.zeros((0, 2), dtype=float)
    targets = np.linspace(0.0, cumulative[-1], n_points)
    return np.column_stack(
        [np.interp(targets, cumulative, path[:, dimension]) for dimension in range(2)]
    )


@dataclass
class ReplayFeatures:
    path_similarity_score: float
    exact_replay_detected: bool
    repeated_duration_count: int
    attempts_per_minute: float
    recent_attempt_count: int
    repeated_endpoint_count: int


@dataclass
class HistoricalAttempt:
    """Minimal projection of a prior attempt used for replay comparison."""

    path: np.ndarray                # shape (N, 2), normalized coords preferred
    duration_ms: float
    endpoint: tuple[float, float]   # final (x, y)
    created_at_epoch_s: float       # submission time, seconds
    path_fingerprint: str | None = None


def _to_path(events: Sequence[dict[str, Any]]) -> np.ndarray:
    pts = sorted(events, key=lambda e: e.get("seq", 0))
    rows = []
    for e in pts:
        x = e.get("x_normalized", e.get("x"))
        y = e.get("y_normalized", e.get("y"))
        if x is None or y is None:
            continue
        try:
            rows.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    return np.asarray(rows, float) if rows else np.zeros((0, 2), float)


def compute_replay_features(
    events: Sequence[dict[str, Any]],
    *,
    duration_ms: float,
    now_epoch_s: float,
    history: Sequence[HistoricalAttempt] = (),
    comparator: PathComparator | None = None,
    duration_tolerance_ms: float = 2.0,
    endpoint_tolerance: float = 0.01,
    recent_window_s: float = 60.0,
) -> ReplayFeatures:
    """Compute the six replay/session signals for the current attempt.

    Args:
        events: current attempt's raw pointer events.
        duration_ms: current attempt drag duration.
        now_epoch_s: current attempt submission time (epoch seconds).
        history: prior attempts in the same session/scope to compare against.
        comparator: path-similarity strategy (defaults to DTW).
        recent_window_s: window for ``recent_attempt_count`` / rate.

    Returns:
        :class:`ReplayFeatures` — all values finite, counts >= 0.
    """
    comparator = comparator or DynamicTimeWarpingComparator()
    cur_path = _to_path(events)
    cur_fingerprint = trace_fingerprint(cur_path)
    cur_end = (float(cur_path[-1][0]), float(cur_path[-1][1])) if cur_path.shape[0] else (0.0, 0.0)

    best_sim = 0.0
    repeated_duration = 0
    repeated_endpoint = 0
    recent = 0
    exact_replay = False

    for h in history:
        sim = comparator.similarity(cur_path, h.path)
        best_sim = max(best_sim, sim)
        historical_fingerprint = h.path_fingerprint or trace_fingerprint(h.path)
        if cur_fingerprint is not None and cur_fingerprint == historical_fingerprint:
            exact_replay = True
        if abs(h.duration_ms - duration_ms) <= duration_tolerance_ms:
            repeated_duration += 1
        if (
            abs(h.endpoint[0] - cur_end[0]) <= endpoint_tolerance
            and abs(h.endpoint[1] - cur_end[1]) <= endpoint_tolerance
        ):
            repeated_endpoint += 1
        if 0.0 <= (now_epoch_s - h.created_at_epoch_s) <= recent_window_s:
            recent += 1

    apm = (recent / recent_window_s) * 60.0 if recent_window_s > 0 else 0.0

    return ReplayFeatures(
        path_similarity_score=round(float(best_sim), 6),
        exact_replay_detected=exact_replay,
        repeated_duration_count=int(repeated_duration),
        attempts_per_minute=round(float(apm), 6),
        recent_attempt_count=int(recent),
        repeated_endpoint_count=int(repeated_endpoint),
    )


__all__ = [
    "DynamicTimeWarpingComparator",
    "HistoricalAttempt",
    "NormalizedPathComparator",
    "PathComparator",
    "ReplayFeatures",
    "compute_replay_features",
    "trace_fingerprint",
]
