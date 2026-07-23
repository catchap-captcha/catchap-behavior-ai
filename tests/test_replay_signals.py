"""Tests for additional offline replay-pair signals."""

from __future__ import annotations

import numpy as np

from training.replay_signals import SIGNAL_NAMES, compute_replay_pair_signals, signal_vector


def _events(path: np.ndarray, times: np.ndarray) -> list[dict]:
    return [
        {
            "seq": index,
            "x_normalized": float(point[0]),
            "y_normalized": float(point[1]),
            "t_ms": float(times[index]),
        }
        for index, point in enumerate(path)
    ]


def test_uniform_space_and_time_warp_keeps_replay_signals_high():
    progress = np.linspace(0.0, 1.0, 61)
    source_path = np.column_stack((progress, 0.2 * np.sin(progress * np.pi)))
    source_time = (progress**1.7) * 900.0
    warped_path = source_path * 1.08 + np.array([0.03, -0.02])
    warped_time = source_time * 1.31 + 40.0

    signals = compute_replay_pair_signals(
        _events(warped_path, warped_time),
        _events(source_path, source_time),
    )

    assert signals.dtw_similarity > 0.99
    assert signals.affine_median_similarity > 0.99
    assert signals.direction_similarity > 0.99
    assert signals.timing_similarity > 0.99
    assert signals.speed_profile_similarity > 0.99


def test_rotation_resampling_and_local_time_warp_keep_pose_invariant_shape_signals_high():
    progress = np.linspace(0.0, 1.0, 71)
    source_path = np.column_stack((progress, 0.18 * np.sin(progress * np.pi * 1.4)))
    source_time = (progress**1.35) * 980.0
    sample_progress = np.linspace(0.0, 1.0, 49) ** 1.23
    sampled = np.column_stack(
        [np.interp(sample_progress, progress, source_path[:, dimension]) for dimension in range(2)]
    )
    angle = np.deg2rad(12.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = (sampled - sampled.mean(axis=0)) @ rotation.T * 1.04 + np.array([0.52, 0.18])
    local_time = np.cumsum(np.r_[0.0, 8.0 + 30.0 * (1.0 + np.sin(sample_progress[1:] * 5.0))])

    signals = compute_replay_pair_signals(
        _events(transformed, local_time),
        _events(source_path, source_time),
    )

    assert signals.procrustes_shape_similarity > 0.94
    assert signals.arc_curvature_profile_similarity > 0.88
    assert signals.distance_shape_similarity > 0.94
    assert signals.trimmed_procrustes_shape_similarity > 0.96
    assert signals.aligned_chamfer_shape_similarity > 0.96
    assert signals.boundary_inlier_procrustes_similarity > 0.96


def test_different_geometry_and_timing_scores_lower_than_replay():
    progress = np.linspace(0.0, 1.0, 61)
    source_path = np.column_stack((progress, 0.25 * np.sin(progress * np.pi)))
    source_time = (progress**1.5) * 800.0
    replay = compute_replay_pair_signals(
        _events(source_path * 0.9 + 0.04, source_time * 1.2),
        _events(source_path, source_time),
    )
    different_path = np.column_stack((progress, 0.18 * np.sin(progress * 4.0 * np.pi)))
    different_time = np.sqrt(progress) * 800.0
    different = compute_replay_pair_signals(
        _events(different_path, different_time),
        _events(source_path, source_time),
    )

    assert different.multi_scale_shape_similarity < replay.multi_scale_shape_similarity
    assert different.curvature_similarity < replay.curvature_similarity
    assert different.timing_similarity < replay.timing_similarity


def test_signal_vector_is_finite_and_canonical():
    path = np.array([[0.0, 0.0], [1.0, 0.0]])
    signals = compute_replay_pair_signals(_events(path, np.array([0.0, 10.0])), [])
    vector = signal_vector(signals)

    assert vector.shape == (len(SIGNAL_NAMES),)
    assert np.isfinite(vector).all()
    assert (vector == 0.0).all()
