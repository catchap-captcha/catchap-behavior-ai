"""Tests for score-guided red-team weakness selection."""

from __future__ import annotations

import numpy as np

from tools.mine_hybrid_redteam_weaknesses import (
    WEAKSET_BOT_FAMILY,
    WEAKSET_GENERATOR_VERSION,
    select_weak_payloads,
)


def _payload(index: int) -> dict:
    return {
        "attempt_id": f"redteam_candidate_{index}",
        "challenge_id": "candidate_challenge",
        "session_id": "candidate_session",
        "collection": {
            "label": "bot",
            "label_source": "hybrid_redteam_generated",
            "bot_family": "hybrid_motion_redteam",
            "generator_version": "hybrid_pca_gmm_motion_v1",
            "training_usage": "redteam_only",
            "evaluation_role": "redteam_calibration",
        },
    }


def test_select_weak_payloads_keeps_highest_human_scores_and_preserves_redteam_only():
    selected, summary = select_weak_payloads(
        [_payload(index) for index in range(4)],
        np.asarray([0.13, 0.94, 0.71, 0.97]),
        threshold=0.95,
        top_k=2,
        model_name="frozen_two_view",
    )

    assert summary["candidate_count"] == 4
    assert summary["selected_count"] == 2
    assert summary["detector_pass_count"] == 1
    assert [row["collection"]["redteam_selection"]["human_score"] for row in selected] == [
        0.97,
        0.94,
    ]
    assert selected[0]["collection"]["redteam_selection"]["passed_detector_threshold"] is True
    assert selected[1]["collection"]["redteam_selection"]["passed_detector_threshold"] is False
    assert all(row["collection"]["training_usage"] == "redteam_only" for row in selected)
    assert all(row["collection"]["bot_family"] == WEAKSET_BOT_FAMILY for row in selected)
    assert all(row["collection"]["generator_version"] == WEAKSET_GENERATOR_VERSION for row in selected)
    assert all(row["attempt_id"].startswith("redteam_weak_") for row in selected)
    assert all(
        row["collection"]["redteam_selection"]["source_candidate_attempt_id"].startswith(
            "redteam_candidate_"
        )
        for row in selected
    )
