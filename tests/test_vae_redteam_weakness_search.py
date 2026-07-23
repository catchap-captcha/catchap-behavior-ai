"""Tests for VAE red-team provenance guards."""

from __future__ import annotations

from tools.vae_redteam_weakness_search import (
    VAE_REDTEAM_FAMILY,
    VAE_REDTEAM_GENERATOR_VERSION,
    build_redteam_payload,
)


def test_vae_redteam_payload_is_detector_forbidden_and_has_no_source_identifier():
    payload = build_redteam_payload(
        attempt_id="vae_redteam_search_seed_000001",
        events=[
            {"seq": 0, "event_type": "pointerdown", "t_ms": 0, "x": 1, "y": 1, "x_normalized": 0.01, "y_normalized": 0.01, "target_role": "captcha_area"},
            {"seq": 1, "event_type": "pointermove", "t_ms": 10, "x": 50, "y": 20, "x_normalized": 0.5, "y_normalized": 0.2, "target_role": "captcha_area"},
            {"seq": 2, "event_type": "pointermove", "t_ms": 20, "x": 80, "y": 30, "x_normalized": 0.8, "y_normalized": 0.3, "target_role": "captcha_area"},
            {"seq": 3, "event_type": "pointerup", "t_ms": 30, "x": 99, "y": 50, "x_normalized": 0.99, "y_normalized": 0.5, "target_role": "captcha_area"},
        ],
        width=100,
        height=100,
        novelty_distance=0.22,
        mutation={"curvature_amplitude": 0.03},
        vae_novelty_distance=0.18,
    )

    collection = payload["collection"]
    assert collection["training_usage"] == "redteam_only"
    assert collection["bot_family"] == VAE_REDTEAM_FAMILY
    assert collection["generator_version"] == VAE_REDTEAM_GENERATOR_VERSION
    assert collection["base_generator"] == "conditional_vae"
    assert "source_attempt_id" not in collection
    assert "source_attempt_id" not in payload
