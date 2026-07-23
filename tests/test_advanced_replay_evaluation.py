"""Tests for grouped advanced-replay threshold selection."""

from __future__ import annotations

import numpy as np

from tools.evaluate_advanced_replay import select_grouped_replay_threshold


def test_grouped_replay_threshold_respects_each_fold_human_fpr():
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.85, 0.95])
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    threshold = select_grouped_replay_threshold(
        scores,
        labels,
        folds,
        max_human_fpr=0.0,
    )

    assert threshold == 0.8
    for fold in (0, 1):
        human = scores[(folds == fold) & (labels == 0)]
        assert float((human >= threshold).mean()) == 0.0


def test_grouped_replay_threshold_rejects_incomplete_fold():
    scores = np.array([0.1, 0.9, 0.2, 0.3])
    labels = np.array([0, 1, 0, 0])
    folds = np.array([0, 0, 1, 1])

    try:
        select_grouped_replay_threshold(
            scores,
            labels,
            folds,
            max_human_fpr=0.1,
        )
    except ValueError as error:
        assert "must contain Human and replay" in str(error)
    else:
        raise AssertionError("expected an incomplete-fold error")
