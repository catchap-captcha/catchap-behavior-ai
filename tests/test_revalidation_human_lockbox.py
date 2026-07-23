"""Tests for the prospective Human lockbox reservation."""

from tools.create_revalidation_human_lockbox import reserve_lockbox


def _row(participant: str | None, attempt_id: str) -> dict:
    return {"attempt_id": attempt_id, "label": "human", "anonymous_participant_id": participant}


def test_lockbox_excludes_previous_test_and_keeps_anonymous_development_rows():
    rows = [
        *[_row("a", f"a_{index}") for index in range(4)],
        *[_row("b", f"b_{index}") for index in range(4)],
        *[_row("c", f"c_{index}") for index in range(4)],
        *[_row("old_test", f"old_{index}") for index in range(4)],
        _row(None, "anonymous"),
    ]
    split = {
        "group_to_split": {
            "human::a": "train",
            "human::b": "train",
            "human::c": "train",
            "human::old_test": "test",
        }
    }

    development, lockbox, manifest = reserve_lockbox(
        rows, split, seed="test", target_fraction=0.3
    )

    lockbox_participants = set(manifest["selection"]["lockbox_participants"])
    assert lockbox_participants <= {"a", "b", "c"}
    assert {row["anonymous_participant_id"] for row in lockbox} == lockbox_participants
    assert all(row["anonymous_participant_id"] != "old_test" for row in development)
    assert any(row["anonymous_participant_id"] is None for row in development)
    assert manifest["training_usage"] == "external_holdout_only"
