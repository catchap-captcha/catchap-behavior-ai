"""Tests for development/external adversarial replay isolation."""

from __future__ import annotations

import random

import numpy as np
import pytest

from tools.generate_adversarial_replay_dataset import generate_dataset
from training.adversarial_replay import (
    DEVELOPMENT_BROAD_PROFILE,
    DEVELOPMENT_COMPOSITE_PROFILE,
    DEVELOPMENT_PROFILE,
    EXTERNAL_HOLDOUT_PROFILE,
    FRESH_EXTERNAL_HOLDOUT_PROFILE,
    adversarial_replay_warp,
)


def _source() -> dict:
    width, height = 400, 240
    progress = np.linspace(0.0, 1.0, 50)
    return {
        "captcha": {"width": width, "height": height},
        "events": [
            {
                "seq": index,
                "t_ms": int(round((progress[index] ** 1.4) * 900)),
                "x": float(35 + progress[index] * 300),
                "y": float(80 + np.sin(progress[index] * np.pi) * 28),
            }
            for index in range(len(progress))
        ],
    }


def test_profiles_have_disjoint_transform_ranges_and_time_curves():
    assert DEVELOPMENT_PROFILE.rotation_abs_degrees[1] < EXTERNAL_HOLDOUT_PROFILE.rotation_abs_degrees[0]
    development_max = max(upper for _, upper in DEVELOPMENT_PROFILE.resample_ratio_ranges)
    external_min = min(lower for lower, _ in EXTERNAL_HOLDOUT_PROFILE.resample_ratio_ranges)
    assert external_min < min(lower for lower, _ in DEVELOPMENT_PROFILE.resample_ratio_ranges)
    assert development_max < min(lower for lower, _ in EXTERNAL_HOLDOUT_PROFILE.resample_ratio_ranges[1:])
    assert max(upper for _, upper in DEVELOPMENT_PROFILE.time_scale_ranges) < min(
        lower for lower, _ in EXTERNAL_HOLDOUT_PROFILE.time_scale_ranges[1:]
    )
    assert DEVELOPMENT_PROFILE.local_curve != EXTERNAL_HOLDOUT_PROFILE.local_curve


def test_broad_development_profile_has_curvature_and_fresh_external_stays_disjoint():
    assert DEVELOPMENT_BROAD_PROFILE.curvature_amplitude_ratio[1] > 0.0
    assert (
        DEVELOPMENT_BROAD_PROFILE.rotation_abs_degrees[1]
        < FRESH_EXTERNAL_HOLDOUT_PROFILE.rotation_abs_degrees[0]
    )
    assert (
        max(upper for _, upper in DEVELOPMENT_BROAD_PROFILE.resample_ratio_ranges)
        < min(lower for lower, _ in FRESH_EXTERNAL_HOLDOUT_PROFILE.resample_ratio_ranges[1:])
    )
    assert DEVELOPMENT_BROAD_PROFILE.local_curve != FRESH_EXTERNAL_HOLDOUT_PROFILE.local_curve


def test_broad_profile_records_nonzero_curvature_drift():
    _, _, _, transform = adversarial_replay_warp(
        _source(), random.Random(31), DEVELOPMENT_BROAD_PROFILE
    )

    assert transform["curvature_amplitude_px"] > 0.0
    assert transform["curvature_cycles"] > 0.0


def test_composite_profile_records_adaptive_resampling_and_micro_jitter():
    _, _, _, transform = adversarial_replay_warp(
        _source(), random.Random(37), DEVELOPMENT_COMPOSITE_PROFILE
    )

    assert transform["adaptive_resample_strength"] > 0.0
    assert transform["micro_jitter_amplitude_px"] > 0.0
    assert transform["curvature_amplitude_px"] > 0.0


@pytest.mark.parametrize("profile", (DEVELOPMENT_PROFILE, EXTERNAL_HOLDOUT_PROFILE))
def test_profiled_warp_has_monotonic_time_and_auditable_profile(profile):
    events, _, _, transform = adversarial_replay_warp(_source(), random.Random(29), profile)

    assert transform["profile"] == profile.name
    assert transform["local_curve"] == profile.local_curve
    assert profile.rotation_abs_degrees[0] <= abs(transform["rotation_degrees"]) <= profile.rotation_abs_degrees[1]
    assert [event["t_ms"] for event in events] == sorted(event["t_ms"] for event in events)


def test_generator_rejects_profile_role_mismatch(tmp_path):
    with pytest.raises(ValueError, match="isolated profile"):
        generate_dataset(
            tmp_path / "human.jsonl",
            tmp_path / "split.json",
            tmp_path / "output.jsonl",
            source_role="development",
            profile_name=EXTERNAL_HOLDOUT_PROFILE.name,
            count=1,
            seed=1,
        )
