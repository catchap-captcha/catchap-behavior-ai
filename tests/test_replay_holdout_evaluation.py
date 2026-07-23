"""Tests for holdout confidence interval helpers."""

from __future__ import annotations

import pytest

from tools.evaluate_replay_holdout import wilson_interval


def test_wilson_interval_for_zero_successes_has_nonzero_upper_bound():
    lower, upper = wilson_interval(0, 1000)

    assert lower == 0.0
    assert upper == pytest.approx(0.0038267585)
