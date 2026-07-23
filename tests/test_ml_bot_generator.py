"""Tests for the offline PCA + GMM defensive surrogate generator."""

from __future__ import annotations

import json
import math

import joblib
import pytest

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.generate_ml_bots import GeneratorConfig, generate_dataset, load_jsonl
from training.run_local_training import build_bot_feature_rows


def _human_attempt(attempt_id: str, offset: float) -> dict:
    width, height = 420, 220
    events = []
    for index in range(16):
        progress = index / 15
        x_normalized = 0.08 + progress * 0.78 + offset * 0.012
        y_normalized = 0.47 + math.sin(progress * math.pi) * (0.09 + offset * 0.003)
        events.append(
            {
                "seq": index,
                "event_type": "pointerdown" if index == 0 else "pointerup" if index == 15 else "pointermove",
                "t_ms": int(round((progress ** (1.1 + offset * 0.01)) * (620 + offset * 9))),
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


def _write_source(tmp_path):
    rows = [
        *[_human_attempt(f"human_train_{index}", float(index)) for index in range(24)],
        *[_human_attempt(f"human_test_{index}", float(index + 30)) for index in range(24)],
    ]
    attempts = tmp_path / "human_attempts.jsonl"
    attempts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "attempt_to_split": {
            **{f"human_train_{index}": "train" for index in range(24)},
            **{f"human_test_{index}": "test" for index in range(24)},
        }
    }
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps(manifest), encoding="utf-8")
    return attempts, split


def test_ml_generator_creates_novel_development_surrogates(tmp_path):
    attempts, split = _write_source(tmp_path)
    output = tmp_path / "development.jsonl"
    model_path = tmp_path / "generator.joblib"
    manifest = generate_dataset(
        human_attempts_path=attempts,
        split_manifest_path=split,
        output_path=output,
        role="development",
        count=12,
        config=GeneratorConfig(
            point_count=16,
            pca_components=6,
            gmm_components=2,
            min_novelty_distance=0.0001,
            seed=17,
        ),
        model_path=model_path,
    )

    rows = load_jsonl(output)
    assert manifest["source_attempt_count"] == 24
    assert len(rows) == 12
    assert all(row["collection"]["training_usage"] == "development_only" for row in rows)
    assert all(row["collection"]["novelty_distance"] >= 0.0001 for row in rows)
    assert all(
        validate_attempt(
            row["events"],
            captcha_width=row["captcha"]["width"],
            captcha_height=row["captcha"]["height"],
        ).status
        != QUALITY_REJECTED
        for row in rows
    )
    assert "human_train_" not in output.read_text(encoding="utf-8")
    bundle = joblib.load(model_path)
    assert bundle["source_role"] == "development"
    assert bundle["source_count"] == 24


def test_ml_generator_keeps_external_holdout_out_of_detector_training(tmp_path):
    attempts, split = _write_source(tmp_path)
    output = tmp_path / "external.jsonl"
    generate_dataset(
        human_attempts_path=attempts,
        split_manifest_path=split,
        output_path=output,
        role="external_holdout",
        count=6,
        config=GeneratorConfig(
            point_count=16,
            pca_components=6,
            gmm_components=2,
            min_novelty_distance=0.0001,
            seed=19,
        ),
    )

    rows = load_jsonl(output)
    assert all(row["collection"]["training_usage"] == "external_holdout_only" for row in rows)
    with pytest.raises(ValueError, match="external-holdout"):
        build_bot_feature_rows(rows, groups_per_family=3)
