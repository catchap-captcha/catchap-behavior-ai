"""Tests for participant-group out-of-fold threshold calibration."""

from __future__ import annotations

import numpy as np

from training.build_dataset import build_dataset
from training.evaluate_models import (
    select_threshold_from_scores,
    select_threshold_per_human_group,
)
from training.group_threshold_cv import calibrate_grouped_threshold
from tests.conftest import make_row


def test_shared_threshold_respects_every_fold_frr_budget():
    scores = np.array([0.91, 0.92, 0.99, 0.995, 0.10, 0.20, 0.30, 0.40])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    folds = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    threshold = select_threshold_from_scores(
        scores,
        labels,
        max_frr=0.0,
        fold_ids=folds,
    )

    assert threshold == 0.91
    for fold in (0, 1):
        human_scores = scores[(folds == fold) & (labels == 1)]
        assert float((human_scores < threshold).mean()) == 0.0


def test_participant_threshold_respects_each_human_group_budget():
    scores = np.array([0.70, 0.99, 0.85, 0.90, 0.10, 0.20, 0.80, 0.95])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array(["human::a", "human::a", "human::b", "human::b", "bot", "bot", "bot", "bot"])

    threshold = select_threshold_per_human_group(
        scores,
        labels,
        groups,
        max_frr=0.0,
    )

    assert threshold == 0.70
    for group in ("human::a", "human::b"):
        group_scores = scores[(groups == group) & (labels == 1)]
        assert float((group_scores < threshold).mean()) == 0.0


def test_grouped_calibration_produces_oof_metrics_without_anonymous_humans():
    rows = []
    for participant in range(9):
        for attempt in range(2):
            rows.append(
                make_row(
                    "human",
                    participant=f"p_{participant}",
                    attempt_id=f"h_{participant}_{attempt}",
                )
            )
    rows.append(make_row("human", participant=None, attempt_id="h_anonymous"))

    for family in ("straight", "accel", "jitter"):
        for group in range(3):
            for attempt in range(2):
                row = make_row(
                    "bot",
                    participant=None,
                    attempt_id=f"b_{family}_{group}_{attempt}",
                    bot_family=family,
                )
                row["generator_version"] = f"rule_batch_{group}"
                rows.append(row)

    ds = build_dataset(rows)
    result = calibrate_grouped_threshold(
        ds,
        rows,
        list(range(len(rows))),
        "random_forest",
        seed=7,
        n_splits=3,
        max_human_frr=0.5,
    )

    assert result.actual_splits == 3
    assert result.excluded_anonymous_human_rows == 1
    assert result.linked_human_groups == 9
    assert result.bot_groups == 9
    assert len(result.folds) == 3
    assert result.pooled_oof_metrics.metrics_on == "group_cv_oof"
    assert result.worst_fold_human_frr <= 0.5
