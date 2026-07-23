"""Tests for fixed-policy Bot-only external holdout summaries."""

from __future__ import annotations

import numpy as np
import pytest

from tools.evaluate_two_view_risk_policy_holdout import (
    evaluate_holdout,
    summarize_policy_scores,
)


def test_policy_summary_separates_allow_step_up_band_and_model_bot_risk():
    summary = summarize_policy_scores(
        np.asarray([0.2, 0.8, 0.9, 0.95]),
        hard_threshold=0.8,
        step_up_threshold=0.9,
    )

    assert summary == {
        "rows": 4,
        "hard_model_asr": 0.75,
        "direct_auto_allow_asr": 0.5,
        "step_up_rate": 0.5,
        "model_bot_risk_count": 1,
        "step_up_band_count": 1,
        "direct_auto_allow_count": 2,
    }


def test_policy_summary_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="hard <= step_up"):
        summarize_policy_scores(
            np.asarray([0.9]),
            hard_threshold=0.91,
            step_up_threshold=0.9,
        )


def test_consumed_external_holdout_is_refused_before_any_model_scoring(tmp_path):
    holdout = tmp_path / "sealed.jsonl"
    holdout.write_text("{}\n", encoding="utf-8")
    manifest = holdout.with_suffix(holdout.suffix + ".manifest.json")
    manifest.write_text(
        '{"training_usage":"external_holdout_only","evaluation_consumed":{"report_path":"old.json"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already been scored"):
        evaluate_holdout(
            model_path=tmp_path / "unused.joblib",
            holdout_path=holdout,
            report_path=tmp_path / "new.json",
        )
