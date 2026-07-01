"""Data readiness gate tests."""

from __future__ import annotations

from training.build_dataset import build_dataset
from training.check_data_readiness import Thresholds, compute_readiness
from tests.conftest import make_row

THR = Thresholds(
    min_human_samples=10,
    min_bot_samples=10,
    min_human_participants=3,
    min_bot_families=2,
)


def test_insufficient_data_not_ready():
    rows = [make_row("human", participant="adult_001", attempt_id="h1")]
    report = compute_readiness(rows, THR)
    assert report.ready is False
    assert report.reason == "data_not_ready"
    assert any("Human 데이터" in m for m in report.missing)
    assert any("Bot 데이터" in m for m in report.missing)


def test_only_one_class_is_not_ready():
    rows = [make_row("human", participant=f"adult_{i:03d}", attempt_id=f"h{i}") for i in range(20)]
    report = compute_readiness(rows, THR)
    assert report.ready is False
    assert report.bot_samples == 0
    assert any("Bot 데이터" in m for m in report.missing)


def test_only_one_class_blocks_dataset_split(training_rows):
    # a dataset with a single class still builds, but has no bot groups to split
    humans = [r for r in training_rows if r["label"] == "human"]
    ds = build_dataset(humans)
    assert set(ds.y.unique()) == {1}


def test_ready_when_thresholds_met(training_rows):
    report = compute_readiness(training_rows, THR)
    assert report.ready is True
    assert report.reason == "ready"
    assert report.class_imbalance_ratio is not None


def test_feature_schema_mismatch_flagged():
    rows = [make_row("human", participant="adult_001", attempt_id="h1")]
    rows[0]["feature_schema_version"] = "9.9"
    report = compute_readiness(rows, THR)
    assert report.feature_schema_mismatch == 1
    assert any("스키마 버전 불일치" in m for m in report.missing)
