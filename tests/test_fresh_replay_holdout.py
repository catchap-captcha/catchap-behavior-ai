from __future__ import annotations

import json

import pytest

from tools.generate_fresh_replay_holdout import assess_readiness, generate_holdout


def _attempt(attempt_id: str, participant: str) -> dict:
    return {
        "attempt_id": attempt_id,
        "anonymous_participant_id": participant,
        "captcha": {"width": 400, "height": 240},
        "events": [
            {"seq": 0, "t_ms": 0, "x": 20.0, "y": 80.0},
            {"seq": 1, "t_ms": 40, "x": 110.0, "y": 85.0},
            {"seq": 2, "t_ms": 100, "x": 240.0, "y": 110.0},
            {"seq": 3, "t_ms": 180, "x": 360.0, "y": 135.0},
        ],
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_readiness_requires_participants_absent_from_known_model_data(tmp_path):
    known = tmp_path / "known.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(known, [{"anonymous_participant_id": "known"}])
    _write_jsonl(attempts, [_attempt("a0", "known"), _attempt("a1", "fresh")])

    readiness = assess_readiness(
        attempts, known, min_fresh_participants=2, count=2
    )

    assert readiness["fresh_participants"] == 1
    assert readiness["ready"] is False


def test_fresh_holdout_is_sealed_and_uses_only_new_participants(tmp_path):
    known = tmp_path / "known.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    output = tmp_path / "fresh.jsonl"
    _write_jsonl(known, [{"anonymous_participant_id": "known"}])
    _write_jsonl(
        attempts,
        [_attempt("known", "known"), _attempt("fresh_a", "fresh_a"), _attempt("fresh_b", "fresh_b")],
    )

    manifest = generate_holdout(
        attempts,
        known,
        output,
        count=2,
        seed=7,
        min_fresh_participants=2,
    )

    assert manifest["training_usage"] == "external_holdout_only"
    assert manifest["fresh_participant_count"] == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["collection"]["evaluation_role"] for row in rows} == {
        "fresh_participant_external_holdout"
    }
    assert all(row["collection"]["training_usage"] == "external_holdout_only" for row in rows)


def test_fresh_holdout_refuses_insufficient_new_participants(tmp_path):
    known = tmp_path / "known.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(known, [{"anonymous_participant_id": "known"}])
    _write_jsonl(attempts, [_attempt("a0", "known")])

    with pytest.raises(ValueError, match="not ready"):
        generate_holdout(
            attempts,
            known,
            tmp_path / "fresh.jsonl",
            count=1,
            seed=7,
            min_fresh_participants=1,
        )
