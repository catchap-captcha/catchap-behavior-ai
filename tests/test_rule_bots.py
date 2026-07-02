"""Tests for the rule-based bot generator.

Verifies generated bots (a) pass quality validation, (b) carry bot-like feature
signatures, and (c) are reproducible for a fixed seed.
"""

from __future__ import annotations

import random

from app.services.feature_extractor import extract_features
from app.services.quality_validator import QUALITY_VALID, validate_attempt
from training.generate_rule_bots import (
    FAMILIES,
    build_collect_payload,
    generate_batch,
    generate_events,
)


def test_all_families_pass_quality():
    for family in FAMILIES:
        events = generate_events(family, rng=random.Random(1))
        r = validate_attempt(events, captcha_width=420, captcha_height=220)
        assert r.status == QUALITY_VALID, f"{family} should be valid, got {r.reason}"


def test_straight_bot_has_bot_signature():
    events = generate_events("straight", rng=random.Random(2))
    feats = extract_features(events, {})
    # perfectly straight, constant cadence -> strong bot signals
    assert feats["linearity"] > 0.99
    assert feats["y_deviation"] < 1e-6
    assert feats["direction_changes"] == 0.0
    assert feats["duplicate_interval_ratio"] > 0.9


def test_jitter_bot_less_perfect_than_straight():
    straight = extract_features(generate_events("straight", rng=random.Random(3)), {})
    jitter = extract_features(generate_events("jitter", rng=random.Random(3)), {})
    # jitter introduces noise -> lower linearity and less duplicated cadence
    assert jitter["linearity"] <= straight["linearity"]
    assert jitter["duplicate_interval_ratio"] <= straight["duplicate_interval_ratio"]


def test_payload_is_labelled_bot():
    events = generate_events("accel", rng=random.Random(4))
    payload = build_collect_payload("att_x", events, "accel")
    assert payload["collection"]["label"] == "bot"
    assert payload["collection"]["label_source"] == "rule_bot"
    assert payload["collection"]["bot_family"] == "accel"


def test_batch_reproducible_and_spread_across_families():
    a = generate_batch(9, FAMILIES, seed=7)
    b = generate_batch(9, FAMILIES, seed=7)
    assert [p["attempt_id"] for p in a] == [p["attempt_id"] for p in b]
    fams = {p["collection"]["bot_family"] for p in a}
    assert fams == set(FAMILIES)  # all three families present
