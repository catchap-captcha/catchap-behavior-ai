"""Tests for detector-forbidden hybrid red-team bot generation."""

from __future__ import annotations

import json
import math

import pytest

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.generate_hybrid_redteam_bots import (
    BOT_FAMILY,
    GENERATOR_VERSION,
    HARD_BOT_FAMILY,
    generate_dataset,
)
from training.generate_ml_bots import GeneratorConfig, load_jsonl
from training.run_local_training import build_bot_feature_rows


def _human_attempt(attempt_id: str, offset: float) -> dict:
    width, height = 420, 220
    events = []
    for index in range(18):
        progress = index / 17
        x_normalized = 0.08 + progress * 0.78 + offset * 0.001
        y_normalized = 0.46 + math.sin(progress * math.pi) * (0.10 + offset * 0.001)
        events.append(
            {
                "seq": index,
                "event_type": "pointerdown" if index == 0 else "pointerup" if index == 17 else "pointermove",
                "t_ms": int(round((progress ** (0.9 + offset * 0.01)) * (620 + offset * 10))),
                "x": x_normalized * width,
                "y": y_normalized * height,
                "x_normalized": x_normalized,
                "y_normalized": y_normalized,
                "target_role": "captcha_area",
            }
        )
    return {"attempt_id": attempt_id, "captcha": {"width": width, "height": height}, "events": events}


def _write_source(tmp_path):
    attempts = tmp_path / "human_attempts.jsonl"
    source_features = tmp_path / "human_development.jsonl"
    rows = [_human_attempt(f"human_dev_{index}", float(index)) for index in range(32)]
    attempts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source_features.write_text(
        "".join(json.dumps({"attempt_id": row["attempt_id"], "label": "human"}) + "\n" for row in rows),
        encoding="utf-8",
    )
    return attempts, source_features


def _config() -> GeneratorConfig:
    return GeneratorConfig(
        point_count=18,
        pca_components=8,
        gmm_components=3,
        min_novelty_distance=0.00001,
        seed=29,
    )


def test_hybrid_redteam_calibration_is_valid_novel_and_detector_forbidden(tmp_path):
    attempts, source_features = _write_source(tmp_path)
    output = tmp_path / "calibration.jsonl"
    manifest = generate_dataset(
        human_attempts_path=attempts,
        source_human_features_path=source_features,
        output_path=output,
        role="calibration",
        count=8,
        config=_config(),
        model_path=tmp_path / "generator.joblib",
    )

    rows = load_jsonl(output)
    assert manifest["training_usage"] == "redteam_only"
    assert manifest["detector_training_forbidden"] is True
    assert len(rows) == 8
    assert all(row["collection"]["bot_family"] == BOT_FAMILY for row in rows)
    assert all(row["collection"]["generator_version"] == GENERATOR_VERSION for row in rows)
    assert all(row["collection"]["novelty_distance"] >= 0.00001 for row in rows)
    assert "human_dev_" not in output.read_text(encoding="utf-8")
    assert all(
        validate_attempt(
            row["events"],
            captcha_width=row["captcha"]["width"],
            captcha_height=row["captcha"]["height"],
        ).status
        != QUALITY_REJECTED
        for row in rows
    )
    with pytest.raises(ValueError, match="detector training"):
        build_bot_feature_rows(rows, groups_per_family=3)


def test_hybrid_redteam_external_holdout_is_sealed(tmp_path):
    attempts, source_features = _write_source(tmp_path)
    output = tmp_path / "external.jsonl"
    manifest = generate_dataset(
        human_attempts_path=attempts,
        source_human_features_path=source_features,
        output_path=output,
        role="external_holdout",
        count=6,
        config=_config(),
    )

    rows = load_jsonl(output)
    assert manifest["training_usage"] == "external_holdout_only"
    assert manifest["threshold_tuning_forbidden"] is True
    assert all(row["collection"]["evaluation_role"] == "external_holdout" for row in rows)
    with pytest.raises(ValueError, match="external-holdout"):
        build_bot_feature_rows(rows, groups_per_family=3)


def test_hybrid_redteam_hard_external_holdout_is_a_separate_sealed_family(tmp_path):
    attempts, source_features = _write_source(tmp_path)
    output = tmp_path / "external_hard.jsonl"
    manifest = generate_dataset(
        human_attempts_path=attempts,
        source_human_features_path=source_features,
        output_path=output,
        role="external_holdout_hard",
        count=6,
        config=_config(),
    )

    rows = load_jsonl(output)
    assert manifest["training_usage"] == "external_holdout_only"
    assert manifest["bot_family"] == HARD_BOT_FAMILY
    assert manifest["policy"]["frame_ms"] == (8, 10, 12, 16)
    assert all(row["collection"]["bot_family"] == HARD_BOT_FAMILY for row in rows)
    with pytest.raises(ValueError, match="external-holdout"):
        build_bot_feature_rows(rows, groups_per_family=3)
