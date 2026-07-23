"""Tests for repeated fixed-detector red-team report aggregation."""

from __future__ import annotations

from tools.summarize_repeated_redteam_scan import summarize_feature_deltas


def _report(weak_pause: float, blocked_pause: float, weak_turn: float, blocked_turn: float) -> dict:
    return {
        "feature_summary": {
            "weak_set": {
                "pause_position_entropy": {"mean": weak_pause},
                "turn_change_smoothness": {"mean": weak_turn},
            },
            "blocked": {
                "pause_position_entropy": {"mean": blocked_pause},
                "turn_change_smoothness": {"mean": blocked_turn},
            },
        }
    }


def test_summarize_feature_deltas_requires_repeated_direction_and_magnitude():
    rows = summarize_feature_deltas(
        [
            _report(0.60, 0.20, 0.30, 0.45),
            _report(0.55, 0.25, 0.25, 0.50),
            _report(0.70, 0.30, 0.40, 0.45),
            _report(0.52, 0.22, 0.45, 0.50),
        ],
        min_runs=4,
        min_absolute_delta=0.02,
    )
    by_feature = {row["feature"]: row for row in rows}

    assert by_feature["pause_position_entropy"]["recurrent"] is True
    assert by_feature["pause_position_entropy"]["repeated_direction"] == "higher_in_weak_set"
    assert by_feature["turn_change_smoothness"]["recurrent"] is True
    assert by_feature["turn_change_smoothness"]["repeated_direction"] == "lower_in_weak_set"
