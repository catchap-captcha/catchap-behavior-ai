"""The audit record has to survive the round trip that broke reproduction before."""

from __future__ import annotations

import json

from app.services.scoring_audit import build_audit


def events(n: int = 5, x0: float = 10.0):
    return [
        {"seq": i, "event_type": "pointermove", "t_ms": i * 20,
         "x": x0 + i * 1.5, "y": 20.0 + i}
        for i in range(n)
    ]


def audit(**over):
    base = dict(
        events=events(),
        features={"event_count": 5.0, "avg_speed": 0.123456789},
        captcha_width=500,
        captcha_height=331,
        scoring_unit="session",
        session_human_score=0.9999500521595306,
        per_drag=None,
    )
    base.update(over)
    return build_audit(**base)


def test_same_input_same_digest():
    assert audit()["input_digest"] == audit()["input_digest"]


def test_changed_coordinate_changes_the_digest():
    moved = events()
    moved[2]["x"] += 1.0
    assert audit(events=moved)["input_digest"] != audit()["input_digest"]


def test_digest_survives_a_json_round_trip():
    """A float that goes through a JSON column comes back subtly different.

    That is exactly how `behavior_batch_payload_invalid` silently discarded every
    batch for a day: the re-hash of stored events no longer matched. Rounding
    before hashing is what makes a stored input verifiable at all.
    """
    original = audit()
    revived = json.loads(json.dumps({"events": events()}))["events"]
    assert audit(events=revived)["input_digest"] == original["input_digest"]


def test_geometry_is_recorded():
    # /predict requires width and height, and nothing stored them — which is why
    # replaying a stored attempt could not reconstruct the request.
    rec = audit()
    assert rec["captcha"] == {"width": 500, "height": 331}


def test_features_are_recorded_and_digested():
    rec = audit()
    assert rec["features"]["avg_speed"] == 0.123457      # rounded, not raw
    assert rec["feature_digest"]
    other = audit(features={"event_count": 5.0, "avg_speed": 0.2})
    assert other["feature_digest"] != rec["feature_digest"]


def test_extra_event_fields_do_not_move_the_digest():
    # Only the fields the extractor reads are digested, so adding a column later
    # does not invalidate every past record.
    decorated = [dict(e, pressure=0.5, pointer_type="mouse") for e in events()]
    assert audit(events=decorated)["input_digest"] == audit()["input_digest"]


def test_record_is_json_serialisable():
    # It goes into a JSON column; a non-serialisable value would fail at write
    # time, i.e. exactly when the evidence is needed.
    json.dumps(audit())
