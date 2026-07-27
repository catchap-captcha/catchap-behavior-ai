"""End-to-end policy composition (design sketch): protocol gate + model band."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.online_policy_sketch import decide
from tools.receipt_chain import CollectionServer

BATCH_MAX = 32
HUMAN_PASS = 0.99999999   # above step-up threshold (model says human)


def _events(span_ms, n):
    return [{"t_ms": round(span_ms * i / (n - 1)), "x": 45 + 300 * i / (n - 1),
             "y": 128 - 40 * i / (n - 1)} for i in range(n)]


def _windows(ev, win=150):
    out, cur, start = [], [], ev[0]["t_ms"]
    for e in ev:
        if e["t_ms"] - start > win or len(cur) >= BATCH_MAX:
            out.append(cur); cur = []; start = e["t_ms"]
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def _honest_server(ev, sid="h"):
    srv = CollectionServer(f"ch_{sid}", f"s_{sid}")
    prev, seq = srv.nonce, 0
    for w in _windows(ev):
        r, err = srv.submit_batch(seq, w, prev, w[-1]["t_ms"] + 15, f"s_{sid}")
        assert err is None, err
        prev, seq = r, seq + 1
    return srv


def _burst_server(ev, sid="b"):
    srv = CollectionServer(f"ch_{sid}", f"s_{sid}")
    srv.submit_batch(0, ev, srv.nonce, 0.0, f"s_{sid}")
    return srv


def test_honest_human_with_human_model_score_allows():
    srv = _honest_server(_events(6000, 104))
    action, risk, reasons = decide(srv, HUMAN_PASS)
    assert action == "allow", (action, risk, reasons)


def test_bulk_burst_replay_is_step_up_even_with_human_model_score():
    srv = _burst_server(_events(6000, 104))
    action, risk, reasons = decide(srv, HUMAN_PASS)
    assert action == "step_up"
    assert "invalid_event_telemetry" in reasons


def test_low_model_score_is_step_up_regardless():
    srv = _honest_server(_events(6000, 104))
    action, risk, reasons = decide(srv, 0.0)
    assert action == "step_up" and "ml_bot_score" in reasons


def test_protocol_gate_only_lowers_never_raises():
    # A failed gate on an otherwise-allow model score must not stay allow.
    srv = _burst_server(_events(6000, 104))
    action, _, _ = decide(srv, HUMAN_PASS)
    assert action != "allow"
