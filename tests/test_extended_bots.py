"""Tests for the seven harder defensive bot trajectory families."""

from __future__ import annotations

import random

import pytest

from app.services.feature_extractor import extract_features
from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.generate_extended_bots import GENERATORS, NEW_FAMILIES, _replay_warp


@pytest.mark.parametrize("family", sorted(GENERATORS))
def test_procedural_extended_family_is_usable(family: str):
    events = GENERATORS[family](random.Random(17), 420, 220)
    quality = validate_attempt(events, captcha_width=420, captcha_height=220)
    features = extract_features(events, {})

    assert quality.status != QUALITY_REJECTED
    assert events[0]["event_type"] == "pointerdown"
    assert events[-1]["event_type"] == "pointerup"
    assert features["event_count"] >= 20
    assert features["duration_ms"] > 0


def test_new_family_registry_has_seven_unique_names():
    assert len(NEW_FAMILIES) == len(set(NEW_FAMILIES)) == 7
    assert set(GENERATORS) == set(NEW_FAMILIES) - {"replay_warp"}


def test_replay_warp_does_not_mutate_source():
    source = {
        "captcha": {"width": 420, "height": 220},
        "events": [
            {"seq": 0, "event_type": "pointerdown", "t_ms": 0, "x": 10, "y": 80},
            {"seq": 1, "event_type": "pointermove", "t_ms": 20, "x": 60, "y": 83},
            {"seq": 2, "event_type": "pointermove", "t_ms": 45, "x": 140, "y": 87},
            {"seq": 3, "event_type": "pointerup", "t_ms": 80, "x": 260, "y": 90},
        ],
    }
    before = [dict(event) for event in source["events"]]
    events, width, height = _replay_warp(source, random.Random(23))

    assert source["events"] == before
    assert validate_attempt(events, captcha_width=width, captcha_height=height).status != QUALITY_REJECTED
    assert len(events) == len(source["events"])
