"""Tests for replay-detector threshold calibration."""

from __future__ import annotations

from tools.evaluate_replay_detection import select_similarity_threshold


def test_similarity_threshold_respects_human_fpr_and_maximizes_recall():
    threshold, recall, fpr = select_similarity_threshold(
        positive=[0.99, 0.98, 0.97, 0.80],
        negative=[0.95, 0.70, 0.60, 0.50],
        max_human_fpr=0.0,
    )

    assert threshold == 0.97
    assert recall == 0.75
    assert fpr == 0.0
