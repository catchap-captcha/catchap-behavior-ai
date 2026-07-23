"""Tests for descriptive score-guided red-team analysis helpers."""

from __future__ import annotations

import numpy as np

from tools.analyze_redteam_weaknesses import rank_feature_separation


def test_rank_feature_separation_orders_largest_standardized_difference_first():
    ranked = rank_feature_separation(
        np.asarray([[10.0, 2.0], [12.0, 2.0], [14.0, 2.0]]),
        np.asarray([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]]),
        ("large_gap", "same_value"),
        limit=2,
    )

    assert [row["feature"] for row in ranked] == ["large_gap", "same_value"]
    assert ranked[0]["standardized_mean_difference"] > 0
    assert ranked[1]["standardized_mean_difference"] == 0.0
