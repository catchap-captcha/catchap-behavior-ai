"""Tests for the Feature v1/v2 comparison report."""

from __future__ import annotations

from tools.compare_feature_versions import compare


def _summary(version: str, frr: float, bot_recall: float):
    holdout = [
        {"held_out_bot_family": "straight", "bot_recall": 0.95, "human_frr": frr},
        {"held_out_bot_family": "replay_warp", "bot_recall": 0.96, "human_frr": frr},
    ]
    return {
        "feature_schema_version": version,
        "split": {"counts": {"train": 1}},
        "test": {"random_forest": {"human_frr": frr, "bot_recall": bot_recall}},
        "family_holdout_stress_test": {"random_forest": holdout},
        "external_browser_bot_holdout": {
            "random_forest": [{"bot_asr": 0.02, "bot_recall": 0.98}]
        },
        "selection": {
            "robust_candidate": {
                "acceptance_criteria": {
                    "experiment_human_frr_max": 0.03,
                    "known_bot_asr_max": 0.05,
                    "unseen_bot_worst_asr_max": 0.10,
                    "replay_warp_asr_max": 0.05,
                }
            }
        },
    }


def test_compare_requires_every_gate():
    report = compare(_summary("1.0", 0.02, 0.96), _summary("2.0", 0.01, 0.90))

    assert report["same_split_required"] is True
    assert report["rows"][0]["experiment_gate_passed"] is True
    assert report["rows"][1]["experiment_gate_passed"] is False
    assert report["recommendation"] == "promote_best_passing_candidate"
