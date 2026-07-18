"""Tests for the confirmed-human Isolation Forest auxiliary score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.anomaly_detector import HumanIsolationForest


def test_far_samples_receive_higher_anomaly_percentiles():
    rng = np.random.default_rng(42)
    human = pd.DataFrame(
        rng.normal(loc=0.0, scale=0.25, size=(240, 3)),
        columns=["speed", "jitter", "pause"],
    )
    detector = HumanIsolationForest(human.columns, seed=42).fit(human)

    held_out_human = rng.normal(loc=0.0, scale=0.25, size=(40, 3))
    obvious_outliers = rng.normal(loc=4.0, scale=0.15, size=(40, 3))

    human_scores = detector.anomaly_percentile(held_out_human)
    outlier_scores = detector.anomaly_percentile(obvious_outliers)

    assert np.all((0.0 <= human_scores) & (human_scores <= 1.0))
    assert np.all((0.0 <= outlier_scores) & (outlier_scores <= 1.0))
    assert outlier_scores.mean() > human_scores.mean()
    assert np.quantile(outlier_scores, 0.25) > np.quantile(human_scores, 0.75)


def test_score_one_uses_canonical_feature_order():
    rng = np.random.default_rng(7)
    human = pd.DataFrame(rng.normal(size=(80, 2)), columns=["x", "y"])
    detector = HumanIsolationForest(["x", "y"], seed=7).fit(human)

    mapping_score = detector.score_one({"y": 0.2, "x": -0.1})
    matrix_score = detector.anomaly_percentile(np.array([[-0.1, 0.2]]))[0]

    assert mapping_score == pytest.approx(matrix_score)


def test_rejects_too_few_or_non_finite_training_rows():
    detector = HumanIsolationForest(["x", "y"])
    with pytest.raises(ValueError, match="not enough confirmed-human"):
        detector.fit(np.zeros((10, 2)))

    invalid = np.zeros((20, 2))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        detector.fit(invalid)
