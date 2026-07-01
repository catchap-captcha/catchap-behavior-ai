"""Replay / session-abuse feature computation.

These signals are deliberately kept OUT of the first-stage behavioral model.
They describe *relationships between attempts* (is this a replay of an earlier
drag? how fast is this session firing attempts?) rather than the shape of a
single drag, and are combined only at the final risk-fusion step.

The similarity backend is pluggable behind :class:`PathComparator` so the
normalized-path / DTW implementation used today can be swapped for an
approximate-nearest-neighbour index later without touching callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

# Two attempts whose normalized paths are closer than this are treated as an
# exact replay. Tuned conservatively; revisit once real data exists.
EXACT_REPLAY_SIMILARITY = 0.98


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
        pa, pb = self._prepare(a), self._prepare(b)
        dist = float(np.linalg.norm(pa - pb, axis=1).mean())
        return 1.0 / (1.0 + dist)


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
        comparator: path-similarity strategy (defaults to normalized path).
        recent_window_s: window for ``recent_attempt_count`` / rate.

    Returns:
        :class:`ReplayFeatures` — all values finite, counts >= 0.
    """
    comparator = comparator or NormalizedPathComparator()
    cur_path = _to_path(events)
    cur_end = (float(cur_path[-1][0]), float(cur_path[-1][1])) if cur_path.shape[0] else (0.0, 0.0)

    best_sim = 0.0
    repeated_duration = 0
    repeated_endpoint = 0
    recent = 0

    for h in history:
        sim = comparator.similarity(cur_path, h.path)
        best_sim = max(best_sim, sim)
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
        exact_replay_detected=bool(best_sim >= EXACT_REPLAY_SIMILARITY),
        repeated_duration_count=int(repeated_duration),
        attempts_per_minute=round(float(apm), 6),
        recent_attempt_count=int(recent),
        repeated_endpoint_count=int(repeated_endpoint),
    )
