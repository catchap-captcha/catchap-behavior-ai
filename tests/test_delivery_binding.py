"""Regression for the delivery-timing binding prototype (protocol §6).

Self-contained (no data/ dependency): builds honest-streamed vs burst vs
fast-chunk vs paced-replay delivery metadata and asserts the loop result —
burst/fast offline replay of a long trajectory is flagged, honest streaming
and short drags pass, and a paced real-time replay evades (documented gap).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.delivery_binding import evaluate_delivery

BATCH_MAX = 32


def _rel_timeline(span_ms, n):
    return [round(span_ms * i / (n - 1)) for i in range(n)]


def honest_stream(rel, win=150, jitter=20.0):
    batches, cur, start = [], [], rel[0]
    for t in rel:
        if t - start > win or len(cur) >= BATCH_MAX:
            batches.append(cur); cur = []; start = t
        cur.append(t)
    if cur:
        batches.append(cur)
    return [{"server_received_at_ms": b[-1] + jitter, "client_t_ms": b} for b in batches]


def burst(rel):
    return [{"server_received_at_ms": 0.0, "client_t_ms": rel}]


def fast_chunk(rel, gap=15.0):
    out, srv, i = [], 0.0, 0
    while i < len(rel):
        out.append({"server_received_at_ms": srv, "client_t_ms": rel[i:i + BATCH_MAX]})
        srv += gap
        i += BATCH_MAX
    return out


def test_long_trajectory_burst_is_flagged():
    rel = _rel_timeline(6000, 104)          # 6s, 104 events — like a passing evader
    ok, reason, _ = evaluate_delivery(burst(rel))
    # a one-shot burst trips the batch-count gate; both reasons mean "blocked"
    assert not ok and reason in {"too_few_batches_for_timeline",
                                 "events_delivered_in_burst"}


def test_long_trajectory_fast_chunk_is_flagged():
    rel = _rel_timeline(6000, 104)
    ok, reason, _ = evaluate_delivery(fast_chunk(rel))
    assert not ok  # server span ~ tiny vs 6s client span


def test_honest_stream_passes():
    rel = _rel_timeline(6000, 104)
    ok, reason, m = evaluate_delivery(honest_stream(rel))
    assert ok, (reason, m)


def test_short_drag_is_exempt():
    rel = _rel_timeline(500, 8)             # genuine fast human drag
    ok, reason, m = evaluate_delivery(burst(rel))
    assert ok and m.get("short_exempt") is True


def test_paced_replay_evades_documented_gap():
    # A patient attacker reproduces honest pacing -> passes at this layer.
    rel = _rel_timeline(6000, 104)
    ok, _, _ = evaluate_delivery(honest_stream(rel))
    assert ok  # residual gap: needs per-event sequencing + receipt chain


def test_missing_batches_is_flagged():
    ok, reason, _ = evaluate_delivery([])
    assert not ok and reason == "no_server_batches"
