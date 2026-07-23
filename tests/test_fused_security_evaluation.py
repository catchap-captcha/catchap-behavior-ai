"""Tests for combined security-evaluation calibration helpers."""

from __future__ import annotations

from tools.evaluate_fused_security import select_session_rate_limit


def test_rate_limit_uses_smallest_value_inside_human_fpr_budget():
    limit, fpr = select_session_rate_limit([0] * 95 + [3] * 4 + [8], max_human_fpr=0.01)

    assert limit == 4
    assert fpr == 0.01


def test_rate_limit_can_disable_all_observed_human_blocks():
    limit, fpr = select_session_rate_limit([0, 1, 2], max_human_fpr=0.0)

    assert limit == 3
    assert fpr == 0.0
