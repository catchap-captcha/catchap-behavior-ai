"""Tests for descriptive red-team generator comparison."""

from __future__ import annotations

from tools.compare_redteam_generator_summaries import compare_feature_recurrence


def test_compare_feature_recurrence_finds_shared_and_candidate_only_patterns():
    baseline = [
        {"feature": "shared", "recurrent": True, "repeated_direction": "higher", "mean_delta": 0.1, "already_in_v23": True},
        {"feature": "not_repeated", "recurrent": False, "repeated_direction": "higher", "mean_delta": 0.01, "already_in_v23": True},
    ]
    candidate = [
        {"feature": "shared", "recurrent": True, "repeated_direction": "higher", "mean_delta": 0.2, "already_in_v23": True},
        {"feature": "candidate_only", "recurrent": True, "repeated_direction": "lower", "mean_delta": -0.2, "already_in_v23": False},
    ]

    result = {item["feature"]: item for item in compare_feature_recurrence(baseline, candidate)}

    assert result["shared"]["status"] == "shared_recurrent"
    assert result["candidate_only"]["status"] == "candidate_generator_recurrent"
    assert result["not_repeated"]["status"] == "not_recurrent"
