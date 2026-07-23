"""Tests for the offline conditional-VAE defensive bot generator."""

from __future__ import annotations

import json
import math

import pytest

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.generate_vae_bots import VaeConfig, generate_dataset, load_jsonl


def _human_attempt(attempt_id: str, offset: float) -> dict:
    width, height = 420, 220
    events = []
    for index in range(16):
        progress = index / 15
        x_normalized = 0.08 + progress * 0.78 + offset * 0.001
        y_normalized = 0.47 + math.sin(progress * math.pi) * (0.09 + offset * 0.001)
        events.append(
            {
                "seq": index,
                "event_type": "pointerdown" if index == 0 else "pointerup" if index == 15 else "pointermove",
                "t_ms": int(round((progress ** (1.05 + offset * 0.002)) * (620 + offset * 4))),
                "x": x_normalized * width,
                "y": y_normalized * height,
                "x_normalized": x_normalized,
                "y_normalized": y_normalized,
                "target_role": "captcha_area",
            }
        )
    return {
        "attempt_id": attempt_id,
        "captcha": {"width": width, "height": height},
        "events": events,
        "collection": {"label": "human"},
    }


def test_conditional_vae_generator_creates_valid_development_bots(tmp_path):
    pytest.importorskip("torch")
    rows = [_human_attempt(f"human_train_{index}", float(index)) for index in range(24)]
    attempts = tmp_path / "human_attempts.jsonl"
    attempts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split = tmp_path / "split_manifest.json"
    split.write_text(
        json.dumps({"attempt_to_split": {row["attempt_id"]: "train" for row in rows}}),
        encoding="utf-8",
    )
    output = tmp_path / "vae_development.jsonl"
    model = tmp_path / "vae.pt"

    manifest = generate_dataset(
        human_attempts_path=attempts,
        split_manifest_path=split,
        output_path=output,
        model_path=model,
        role="development",
        count=4,
        config=VaeConfig(
            point_count=16,
            latent_dim=4,
            hidden_dim=16,
            batch_size=8,
            epochs=1,
            min_novelty_distance=0.0,
            seed=19,
        ),
    )

    generated = load_jsonl(output)
    assert manifest["training_usage"] == "development_only"
    assert model.exists()
    assert len(generated) == 4
    assert all(row["collection"]["training_usage"] == "development_only" for row in generated)
    assert all(
        validate_attempt(
            row["events"],
            captcha_width=row["captcha"]["width"],
            captcha_height=row["captcha"]["height"],
        ).status != QUALITY_REJECTED
        for row in generated
    )
