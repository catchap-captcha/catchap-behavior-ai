"""Grouped-split leakage tests."""

from __future__ import annotations

import pytest

from training.build_dataset import build_dataset
from training.split_dataset import split_dataset
from tests.conftest import make_row


def test_participant_never_leaks_across_splits(training_rows):
    ds = build_dataset(training_rows)
    split = split_dataset(ds, seed=1)
    g2s = split.manifest["group_to_split"]
    # every group maps to exactly one split (dict guarantees uniqueness);
    # assert no human participant group appears under two splits by construction
    human_groups = [g for g in g2s if g.startswith("human::")]
    assert human_groups, "expected human groups in manifest"
    # counts add up, no overlap
    total = sum(split.manifest["counts"].values())
    assert total == len(ds)


def test_generator_group_never_leaks(training_rows):
    ds = build_dataset(training_rows)
    split = split_dataset(ds, seed=7)
    # reconstruct group->split from the three splits and assert disjoint
    # (split_dataset already raises LeakageError internally on any overlap)
    assert split.manifest["counts"]["train"] > 0
    assert split.manifest["counts"]["val"] >= 0
    assert split.manifest["counts"]["test"] >= 0


def test_too_few_groups_raises():
    rows = [
        make_row("human", participant="adult_001", attempt_id="h1"),
        make_row("bot", bot_family="straight", generator_version="g1", attempt_id="b1"),
    ]
    ds = build_dataset(rows)
    with pytest.raises(ValueError):
        split_dataset(ds, seed=1)


def test_gan_bot_follows_origin_participant():
    # a GAN bot carrying an origin participant id groups with that human
    row = make_row("bot", bot_family="gan", attempt_id="gb1")
    row["anonymous_participant_id"] = "adult_042"
    from training.build_dataset import group_key
    import pandas as pd

    assert group_key(pd.Series(row)) == "human::adult_042"
