"""Tests for exact and warped pointer-path replay detection."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.replay_detector import (
    DynamicTimeWarpingComparator,
    HistoricalAttempt,
    NormalizedPathComparator,
    compute_replay_features,
    trace_fingerprint,
)


def _curve(n: int = 31) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((t, 0.3 * np.sin(np.pi * t)))


def _events(path: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {"seq": i, "x_normalized": float(x), "y_normalized": float(y)}
        for i, (x, y) in enumerate(path)
    ]


def test_trace_fingerprint_ignores_translation_but_not_scale():
    path = _curve()
    translated = path + np.array([5.0, -2.0])
    scaled = path * 1.1

    assert trace_fingerprint(path) == trace_fingerprint(translated)
    assert trace_fingerprint(path) != trace_fingerprint(scaled)


def test_dtw_handles_resampling_translation_and_uniform_scale():
    comparator = DynamicTimeWarpingComparator()
    path = _curve(31)
    warped = (_curve(67) * 1.2) + np.array([0.3, -0.4])
    different = np.column_stack((np.linspace(0.0, 1.0, 31), np.zeros(31)))

    same_score = comparator.similarity(path, warped)
    different_score = comparator.similarity(path, different)

    assert comparator.similarity(path, path) == pytest.approx(1.0)
    assert same_score > 0.97
    assert same_score > different_score


def test_compute_replay_features_separates_exact_and_warped_replays():
    path = _curve()
    translated = path + np.array([0.2, -0.1])
    scaled = path * 1.15
    history = [
        HistoricalAttempt(
            path=translated,
            duration_ms=900.0,
            endpoint=tuple(translated[-1]),
            created_at_epoch_s=970.0,
        ),
        HistoricalAttempt(
            path=scaled,
            duration_ms=750.0,
            endpoint=tuple(scaled[-1]),
            created_at_epoch_s=980.0,
        ),
    ]

    features = compute_replay_features(
        _events(path),
        duration_ms=900.0,
        now_epoch_s=1000.0,
        history=history,
    )

    assert features.exact_replay_detected is True
    assert features.path_similarity_score > 0.99
    assert features.repeated_duration_count == 1
    assert features.recent_attempt_count == 2
    assert features.attempts_per_minute == pytest.approx(2.0)


def test_short_or_invalid_paths_do_not_look_identical():
    dtw = DynamicTimeWarpingComparator()
    normalized = NormalizedPathComparator()
    point = np.array([[0.0, 0.0]])
    invalid = np.array([[0.0, np.nan], [1.0, 1.0]])

    assert dtw.similarity(point, point) == 0.0
    assert dtw.similarity(invalid, _curve()) == 0.0
    assert normalized.similarity(point, point) == 0.0
    assert trace_fingerprint(point) is None
