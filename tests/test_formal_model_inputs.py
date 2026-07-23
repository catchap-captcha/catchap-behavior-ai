"""Tests for selecting precomputed Bot features in the local trainer."""

from __future__ import annotations

from argparse import Namespace

import pytest

from training.run_local_training import run


def test_local_training_rejects_raw_and_precomputed_bot_inputs_together():
    args = Namespace(
        feature_schema_version="1.0",
        trajectory_only=False,
        human_features="unused.jsonl",
        bot_attempts=["raw.jsonl"],
        bot_features=["features.jsonl"],
        external_bot_holdout=[],
        dataset_dir="unused",
        report_dir="unused",
        candidate_dir="unused",
        bot_groups_per_family=3,
        seed=42,
        threshold_cv_folds=5,
        skip_dataset_copy=False,
        skip_family_holdout=False,
        skip_external_holdout=False,
        model=None,
        dataset_version="unused",
    )

    with pytest.raises(ValueError, match="either --bot-attempts or --bot-features"):
        run(args)
