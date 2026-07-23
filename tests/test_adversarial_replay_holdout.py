"""Tests for the unseen combined replay holdout generator."""

from __future__ import annotations

import random

import numpy as np

from tools.generate_adversarial_replay_holdout import adversarial_replay_warp


def _source() -> dict:
    width, height = 400, 240
    progress = np.linspace(0.0, 1.0, 50)
    return {
        "captcha": {"width": width, "height": height},
        "events": [
            {
                "seq": index,
                "t_ms": int(round((progress[index] ** 1.4) * 900)),
                "x": float(35 + progress[index] * 300),
                "y": float(80 + np.sin(progress[index] * np.pi) * 28),
            }
            for index in range(len(progress))
        ],
    }


def test_adversarial_holdout_combines_multiple_trace_changes():
    events, width, height, transform = adversarial_replay_warp(_source(), random.Random(9))

    assert width == 400
    assert height == 240
    assert len(events) != transform["source_event_count"]
    assert 4 <= len(events)
    assert 4.0 <= abs(transform["rotation_degrees"]) <= 13.0
    assert transform["slow_strength"] > 1.0
    assert transform["fast_strength"] < 1.0
    assert [event["t_ms"] for event in events] == sorted(event["t_ms"] for event in events)
    assert all(0.0 <= event["x_normalized"] <= 1.0 for event in events)
    assert all(0.0 <= event["y_normalized"] <= 1.0 for event in events)
