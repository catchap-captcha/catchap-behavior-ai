"""Tests for prospective known-Bot lockbox reservation."""

from tools.create_revalidation_bot_lockbox import reserve_lockbox


def _row(family: str, generator: str, attempt_id: str) -> dict:
    return {
        "attempt_id": attempt_id,
        "label": "bot",
        "bot_family": family,
        "generator_version": generator,
    }


def test_bot_lockbox_uses_only_previous_train_groups_and_excludes_test_rows():
    rows = [
        _row("a", "a_train_1", "a1"),
        _row("a", "a_train_2", "a2"),
        _row("a", "a_test", "a3"),
        _row("b", "b_train_1", "b1"),
        _row("b", "b_train_2", "b2"),
        _row("b", "b_test", "b3"),
    ]
    split = {
        "attempt_to_split": {
            "a1": "train", "a2": "train", "a3": "test",
            "b1": "train", "b2": "train", "b3": "test",
        }
    }

    development, lockbox, manifest = reserve_lockbox(
        rows, split, seed="test", excluded_families=set()
    )

    chosen = manifest["selection"]["lockbox_generator_by_family"]
    assert {row["generator_version"] for row in lockbox} == set(chosen.values())
    assert all(row["generator_version"] not in {"a_test", "b_test"} for row in development)
    assert manifest["counts"]["lockbox_families"] == 2
