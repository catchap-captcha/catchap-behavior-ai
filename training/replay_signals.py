"""Pairwise replay signals used only by the offline security evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np

from app.services.replay_detector import DynamicTimeWarpingComparator, path_from_events


SIGNAL_NAMES = (
    "dtw_similarity",
    "affine_median_similarity",
    "affine_p90_similarity",
    "direction_similarity",
    "curvature_similarity",
    "timing_similarity",
    "speed_profile_similarity",
    "multi_scale_shape_similarity",
    "event_count_ratio",
    "procrustes_shape_similarity",
    "arc_curvature_profile_similarity",
    "distance_shape_similarity",
    "trimmed_procrustes_shape_similarity",
    "aligned_chamfer_shape_similarity",
    "boundary_inlier_procrustes_similarity",
)


@dataclass(frozen=True)
class ReplayPairSignals:
    dtw_similarity: float
    affine_median_similarity: float
    affine_p90_similarity: float
    direction_similarity: float
    curvature_similarity: float
    timing_similarity: float
    speed_profile_similarity: float
    multi_scale_shape_similarity: float
    event_count_ratio: float
    procrustes_shape_similarity: float
    arc_curvature_profile_similarity: float
    distance_shape_similarity: float
    trimmed_procrustes_shape_similarity: float
    aligned_chamfer_shape_similarity: float
    boundary_inlier_procrustes_similarity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_replay_pair_signals(
    current_events: Sequence[dict[str, Any]],
    source_events: Sequence[dict[str, Any]],
    *,
    dtw_comparator: DynamicTimeWarpingComparator | None = None,
) -> ReplayPairSignals:
    """Compare a current trajectory with one retained historical source."""
    current_path, current_time = _event_arrays(current_events)
    source_path, source_time = _event_arrays(source_events)
    if len(current_path) < 2 or len(source_path) < 2:
        return ReplayPairSignals(*(0.0 for _ in SIGNAL_NAMES))

    comparator = dtw_comparator or DynamicTimeWarpingComparator(max_points=48)
    dtw = comparator.similarity(current_path, source_path)

    source_index = _index_resample(source_path, 48)
    current_index = _index_resample(current_path, 48)
    median_residual, p90_residual = _affine_residuals(source_index, current_index)

    source_arc = _arc_resample(source_path, 48)
    current_arc = _arc_resample(current_path, 48)
    direction = _direction_similarity(source_arc, current_arc)
    curvature = _curvature_similarity(source_arc, current_arc)
    timing = _timing_similarity(source_time, current_time)
    speed = _speed_profile_similarity(source_path, source_time, current_path, current_time)
    multi_scale = float(
        np.mean(
            [
                _shape_similarity(_arc_resample(source_path, count), _arc_resample(current_path, count))
                for count in (16, 32, 64)
            ]
        )
    )
    count_ratio = min(len(current_path), len(source_path)) / max(len(current_path), len(source_path))
    procrustes = _procrustes_similarity(source_arc, current_arc)
    arc_curvature = _arc_curvature_profile_similarity(source_arc, current_arc)
    distance_shape = _distance_shape_similarity(source_arc, current_arc)
    trimmed_procrustes, aligned_source = _trimmed_procrustes_similarity(source_arc, current_arc)
    aligned_chamfer = _aligned_chamfer_similarity(aligned_source, current_arc)
    boundary_inlier_procrustes = _boundary_inlier_procrustes_similarity(source_arc, current_arc)

    return ReplayPairSignals(
        dtw_similarity=_finite_unit(dtw),
        affine_median_similarity=_residual_similarity(median_residual),
        affine_p90_similarity=_residual_similarity(p90_residual),
        direction_similarity=_finite_unit(direction),
        curvature_similarity=_finite_unit(curvature),
        timing_similarity=_finite_unit(timing),
        speed_profile_similarity=_finite_unit(speed),
        multi_scale_shape_similarity=_finite_unit(multi_scale),
        event_count_ratio=_finite_unit(count_ratio),
        procrustes_shape_similarity=_finite_unit(procrustes),
        arc_curvature_profile_similarity=_finite_unit(arc_curvature),
        distance_shape_similarity=_finite_unit(distance_shape),
        trimmed_procrustes_shape_similarity=_finite_unit(trimmed_procrustes),
        aligned_chamfer_shape_similarity=_finite_unit(aligned_chamfer),
        boundary_inlier_procrustes_similarity=_finite_unit(boundary_inlier_procrustes),
    )


def signal_vector(
    signals: ReplayPairSignals,
    signal_names: Sequence[str] = SIGNAL_NAMES,
) -> np.ndarray:
    """Return the requested signal schema, including frozen older bundles."""
    unknown = set(signal_names) - set(SIGNAL_NAMES)
    if unknown:
        raise ValueError(f"unknown replay signal names: {sorted(unknown)}")
    return np.asarray([getattr(signals, name) for name in signal_names], dtype=float)


def _event_arrays(events: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(events, key=lambda item: item.get("seq", 0))
    points: list[tuple[float, float]] = []
    times: list[float] = []
    for event in ordered:
        x = event.get("x_normalized", event.get("x"))
        y = event.get("y_normalized", event.get("y"))
        t = event.get("t_ms")
        try:
            values = float(x), float(y), float(t)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).all():
            continue
        points.append(values[:2])
        times.append(values[2])
    return np.asarray(points, dtype=float), np.asarray(times, dtype=float)


def _index_resample(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((count, values.shape[1] if values.ndim == 2 else 1), dtype=float)
    source = np.arange(len(values), dtype=float)
    target = np.linspace(0.0, len(values) - 1, count)
    if values.ndim == 1:
        return np.interp(target, source, values)
    return np.column_stack(
        [np.interp(target, source, values[:, dimension]) for dimension in range(values.shape[1])]
    )


def _arc_resample(path: np.ndarray, count: int) -> np.ndarray:
    if len(path) < 2:
        return np.zeros((count, 2), dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-12:
        return np.repeat(path[:1], count, axis=0)
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    clean = path[keep]
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, clean[:, dimension]) for dimension in range(2)]
    )


def _affine_residuals(source: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    source_centered = source - source.mean(axis=0)
    current_centered = current - current.mean(axis=0)
    denominator = float(np.sum(source_centered * source_centered))
    scale = (
        float(np.sum(source_centered * current_centered)) / denominator
        if denominator > 1e-12
        else 0.0
    )
    aligned = source_centered * scale + current.mean(axis=0)
    normalizer = max(float(np.linalg.norm(np.diff(current, axis=0), axis=1).sum()), 1e-9)
    residuals = np.linalg.norm(aligned - current, axis=1) / normalizer
    return float(np.median(residuals)), float(np.percentile(residuals, 90))


def _residual_similarity(residual: float) -> float:
    return _finite_unit(1.0 / (1.0 + 100.0 * max(float(residual), 0.0)))


def _direction_similarity(source: np.ndarray, current: np.ndarray) -> float:
    left = np.diff(source, axis=0)
    right = np.diff(current, axis=0)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 1e-12
    if not valid.any():
        return 0.0
    cosine = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return float(np.mean((np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0))


def _curvature_similarity(source: np.ndarray, current: np.ndarray) -> float:
    source_angle = _turning_angles(source)
    current_angle = _turning_angles(current)
    if not len(source_angle) or not len(current_angle):
        return 0.0
    count = min(len(source_angle), len(current_angle))
    difference = source_angle[:count] - current_angle[:count]
    return float(np.mean((np.cos(difference) + 1.0) / 2.0))


def _turning_angles(path: np.ndarray) -> np.ndarray:
    delta = np.diff(path, axis=0)
    if len(delta) < 2:
        return np.zeros(0, dtype=float)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    return np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))


def _timing_similarity(source_time: np.ndarray, current_time: np.ndarray) -> float:
    source = _normalized_time_profile(source_time, 48)
    current = _normalized_time_profile(current_time, 48)
    mae = float(np.mean(np.abs(source - current)))
    return 1.0 / (1.0 + 20.0 * mae)


def _normalized_time_profile(times: np.ndarray, count: int) -> np.ndarray:
    if len(times) < 2:
        return np.zeros(count, dtype=float)
    relative = np.maximum.accumulate(times - times[0])
    duration = float(relative[-1])
    if duration <= 1e-12:
        return np.zeros(count, dtype=float)
    return _index_resample(relative / duration, count)


def _speed_profile_similarity(
    source_path: np.ndarray,
    source_time: np.ndarray,
    current_path: np.ndarray,
    current_time: np.ndarray,
) -> float:
    source = _normalized_speed_profile(source_path, source_time, 48)
    current = _normalized_speed_profile(current_path, current_time, 48)
    if np.std(source) <= 1e-12 or np.std(current) <= 1e-12:
        return float(1.0 / (1.0 + np.mean(np.abs(source - current))))
    correlation = float(np.corrcoef(source, current)[0, 1])
    return (np.clip(correlation, -1.0, 1.0) + 1.0) / 2.0


def _normalized_speed_profile(path: np.ndarray, times: np.ndarray, count: int) -> np.ndarray:
    if len(path) < 2 or len(times) != len(path):
        return np.zeros(count, dtype=float)
    distance = np.linalg.norm(np.diff(path, axis=0), axis=1)
    dt = np.maximum(np.diff(np.maximum.accumulate(times)), 1e-6)
    speed = distance / dt
    positive = speed[speed > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    normalized = np.log1p(speed / max(scale, 1e-12))
    return _index_resample(normalized, count)


def _shape_similarity(source: np.ndarray, current: np.ndarray) -> float:
    median_residual, _ = _affine_residuals(source, current)
    return _residual_similarity(median_residual)


def _procrustes_similarity(source: np.ndarray, current: np.ndarray) -> float:
    """Compare shape after the best translation, rotation, and scale alignment.

    Both paths are arc-resampled before this function.  That removes event
    deletion/interpolation and local timing as sources of mismatch, while the
    Procrustes alignment removes rigid rotation and uniform spatial scale.
    """
    if len(source) < 2 or len(current) < 2:
        return 0.0
    left = source - source.mean(axis=0)
    right = current - current.mean(axis=0)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    left = left / left_norm
    right = right / right_norm
    u, _, vt = np.linalg.svd(left.T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    residual = float(np.mean(np.linalg.norm(left @ rotation - right, axis=1)))
    return 1.0 / (1.0 + 12.0 * residual)


def _fit_similarity_transform(
    source: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    left = source[mask]
    right = current[mask]
    left_center = left.mean(axis=0)
    right_center = right.mean(axis=0)
    left_zero = left - left_center
    right_zero = right - right_center
    denominator = float(np.sum(left_zero * left_zero))
    if denominator <= 1e-12:
        return np.repeat(right_center[None, :], len(source), axis=0)
    u, _, vt = np.linalg.svd(left_zero.T @ right_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum((left_zero @ rotation) * right_zero)) / denominator
    return (source - left_center) @ rotation * scale + right_center


def _trimmed_procrustes_similarity(source: np.ndarray, current: np.ndarray) -> tuple[float, np.ndarray]:
    """Align a shared trajectory core while downweighting clipped end regions."""
    if len(source) < 4 or len(current) < 4:
        return 0.0, np.zeros_like(source)
    mask = np.ones(len(source), dtype=bool)
    retained = max(4, int(math.ceil(len(source) * 0.72)))
    aligned = source
    for _ in range(3):
        aligned = _fit_similarity_transform(source, current, mask)
        residual = np.linalg.norm(aligned - current, axis=1)
        keep = np.argpartition(residual, retained - 1)[:retained]
        mask = np.zeros(len(source), dtype=bool)
        mask[keep] = True
    residual = np.linalg.norm(aligned[mask] - current[mask], axis=1)
    span = max(
        float(np.linalg.norm(current.max(axis=0) - current.min(axis=0))),
        float(np.linalg.norm(source.max(axis=0) - source.min(axis=0))),
        1e-9,
    )
    normalized = float(np.median(residual)) / span
    return 1.0 / (1.0 + 32.0 * normalized), aligned


def _aligned_chamfer_similarity(aligned_source: np.ndarray, current: np.ndarray) -> float:
    """Allow nonuniform resampling after pose alignment via nearest-path distance."""
    if len(aligned_source) < 2 or len(current) < 2:
        return 0.0
    distances = np.linalg.norm(aligned_source[:, None, :] - current[None, :, :], axis=2)
    nearest = np.concatenate((distances.min(axis=0), distances.min(axis=1)))
    retained = max(2, int(math.ceil(len(nearest) * 0.80)))
    core = np.partition(nearest, retained - 1)[:retained]
    span = max(
        float(np.linalg.norm(current.max(axis=0) - current.min(axis=0))),
        float(np.linalg.norm(aligned_source.max(axis=0) - aligned_source.min(axis=0))),
        1e-9,
    )
    return 1.0 / (1.0 + 32.0 * float(np.mean(core)) / span)


def _boundary_inlier_procrustes_similarity(source: np.ndarray, current: np.ndarray) -> float:
    """Fit only points not clamped at a CAPTCHA boundary.

    A rotated replay may retain its original path through most of the gesture
    but flatten a short section at the visible edge.  Boundary points are not
    useful for estimating the rigid transform, so this signal leaves them out
    of the fit and then scores the retained interior path.
    """
    if len(source) < 4 or len(current) < 4:
        return 0.0
    interior = np.all((current > 0.003) & (current < 0.997), axis=1)
    if int(interior.sum()) < 4:
        interior = np.ones(len(current), dtype=bool)
    aligned = _fit_similarity_transform(source, current, interior)
    residual = np.linalg.norm(aligned[interior] - current[interior], axis=1)
    span = max(
        float(np.linalg.norm(current[interior].max(axis=0) - current[interior].min(axis=0))),
        float(np.linalg.norm(source[interior].max(axis=0) - source[interior].min(axis=0))),
        1e-9,
    )
    return 1.0 / (1.0 + 24.0 * float(np.median(residual)) / span)


def _arc_curvature_profile_similarity(source: np.ndarray, current: np.ndarray) -> float:
    """Compare relative turning, which is invariant to global rotation."""
    left = np.abs(_turning_angles(source))
    right = np.abs(_turning_angles(current))
    if not len(left) or not len(right):
        return 0.0
    count = 48
    left = _index_resample(left, count)
    right = _index_resample(right, count)
    normalizer = max(float(np.mean(left) + np.mean(right)), 1e-6)
    mae = float(np.mean(np.abs(left - right))) / normalizer
    return 1.0 / (1.0 + 3.0 * mae)


def _distance_shape_similarity(source: np.ndarray, current: np.ndarray) -> float:
    """Compare normalized chord distances, invariant to pose and time warp."""
    count = 24
    left = _arc_resample(source, count)
    right = _arc_resample(current, count)
    left_distances = np.linalg.norm(left[:, None, :] - left[None, :, :], axis=2)
    right_distances = np.linalg.norm(right[:, None, :] - right[None, :, :], axis=2)
    upper = np.triu_indices(count, k=1)
    left_values = left_distances[upper]
    right_values = right_distances[upper]
    left_scale = max(float(np.max(left_values)), 1e-9)
    right_scale = max(float(np.max(right_values)), 1e-9)
    mae = float(np.mean(np.abs(left_values / left_scale - right_values / right_scale)))
    return 1.0 / (1.0 + 10.0 * mae)


def _finite_unit(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


__all__ = [
    "ReplayPairSignals",
    "SIGNAL_NAMES",
    "compute_replay_pair_signals",
    "signal_vector",
]
