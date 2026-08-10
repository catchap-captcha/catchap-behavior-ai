"""Tests for exact and warped pointer-path replay detection."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.replay_detector import (
    DynamicTimeWarpingComparator,
    HistoricalAttempt,
    NormalizedPathComparator,
    ProcrustesPathComparator,
    compute_replay_features,
    trace_fingerprint,
    trace_fingerprint_from_events,
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


def test_event_fingerprint_uses_normalized_coordinates():
    events = [
        {"seq": 0, "x": 10, "y": 20, "x_normalized": 0.1, "y_normalized": 0.2},
        {"seq": 1, "x": 20, "y": 30, "x_normalized": 0.2, "y_normalized": 0.3},
    ]
    expected = trace_fingerprint(np.asarray([[0.1, 0.2], [0.2, 0.3]]))

    assert trace_fingerprint_from_events(events) == expected


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


def _rotate(path: np.ndarray, radians: float, scale: float = 1.0) -> np.ndarray:
    """What a replay attacker does to reuse a captured path on a new target."""
    rotation = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)],
    ])
    return (path - path[0]) @ rotation.T * scale + path[0]


def test_procrustes_sees_through_the_rotation_a_replay_must_apply():
    """The transform that defines the attack must not be the one that hides it.

    An attacker reusing a captured trajectory has to rotate it, because the
    object is somewhere else this time. DTW normalizes away translation and
    scale but not rotation, so it reads a rotated replay as an unrelated path —
    measured at ~0.61 on real data, far below any threshold worth setting.
    """
    procrustes = ProcrustesPathComparator()
    dtw = DynamicTimeWarpingComparator()
    original = _curve()

    for radians in (0.4, 1.2, 2.5, 4.0):
        replayed = _rotate(original, radians, scale=1.3)
        assert procrustes.similarity(original, replayed) > 0.99
        assert dtw.similarity(original, replayed) < 0.9

    # Invariance must not become blindness: a genuinely different shape stays
    # far away no matter how it is turned.
    other = np.column_stack((np.linspace(0.0, 1.0, 31), np.zeros(31)))
    assert procrustes.similarity(original, _rotate(other, 1.0)) < 0.9


def test_procrustes_is_symmetric_and_refuses_degenerate_paths():
    procrustes = ProcrustesPathComparator()
    a, b = _curve(), _rotate(_curve(), 0.8, scale=0.7)
    assert procrustes.similarity(a, b) == pytest.approx(procrustes.similarity(b, a))

    point = np.array([[0.0, 0.0]])
    stationary = np.tile([0.5, 0.5], (20, 1))
    invalid = np.array([[0.0, np.nan], [1.0, 1.0]])
    assert procrustes.similarity(point, point) == 0.0
    assert procrustes.similarity(stationary, _curve()) == 0.0
    assert procrustes.similarity(invalid, _curve()) == 0.0


def test_rotated_replay_scores_above_the_deployed_threshold():
    """The comparator and the threshold have to move together.

    `risk_dtw_similarity_threshold` was recalibrated for this comparator; if
    either is changed alone the signal goes quiet without failing anything.
    """
    from app.config import Settings

    threshold = Settings().risk_dtw_similarity_threshold
    original = _curve()
    history = [HistoricalAttempt(
        path=original, duration_ms=700.0,
        endpoint=(float(original[-1][0]), float(original[-1][1])),
        created_at_epoch_s=1000.0,
    )]
    replayed = _rotate(original, 1.1, scale=1.2)
    events = [{"seq": i, "event_type": "pointer_move", "t_ms": i * 50.0,
               "x": float(x), "y": float(y)} for i, (x, y) in enumerate(replayed)]

    features = compute_replay_features(
        events, duration_ms=700.0, now_epoch_s=1001.0, history=history)
    assert features.path_similarity_score >= threshold
    assert not features.exact_replay_detected  # rotation changes the hash
