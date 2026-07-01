"""GAN readiness gate + trajectory preprocessing tests."""

from __future__ import annotations

import math

import numpy as np

from training.train_gan import (
    FIXED_LEN,
    GanThresholds,
    compute_gan_readiness,
    normalize_trajectory,
)
from tests.conftest import human_like_events

THR = GanThresholds(min_gan_human_samples=100, min_gan_human_participants=10)


def test_gan_blocked_when_insufficient_human_data():
    rows = [{"anonymous_participant_id": f"adult_{i:03d}"} for i in range(5)]
    report = compute_gan_readiness(rows, THR)
    assert report.ready is False
    assert report.reason == "gan_data_not_ready"
    assert any("Human 원본" in m for m in report.missing)
    assert any("참여자" in m for m in report.missing)


def test_gan_ready_when_enough():
    rows = [{"anonymous_participant_id": f"adult_{i:03d}"} for i in range(120)]
    report = compute_gan_readiness(rows, THR)
    assert report.ready is True
    assert report.reason == "ready"


def test_normalize_trajectory_shape_and_anchors():
    traj = normalize_trajectory(human_like_events())
    assert traj.shape == (FIXED_LEN, 3)
    # start anchored to (0,0), end to (1,0) along the drag axis
    assert abs(traj[0, 0]) < 1e-6 and abs(traj[0, 1]) < 1e-6
    assert abs(traj[-1, 0] - 1.0) < 1e-6 and abs(traj[-1, 1]) < 1e-6
    # time normalized to [0,1], non-decreasing
    assert abs(traj[0, 2]) < 1e-6 and abs(traj[-1, 2] - 1.0) < 1e-6
    assert np.all(np.diff(traj[:, 2]) >= -1e-9)
    assert np.all(np.isfinite(traj))


def test_normalize_degenerate_returns_zeros():
    traj = normalize_trajectory([{"seq": 0, "event_type": "pointerdown", "t_ms": 0, "x": 1, "y": 1}])
    assert traj.shape == (FIXED_LEN, 3)
    assert np.all(traj == 0.0)
