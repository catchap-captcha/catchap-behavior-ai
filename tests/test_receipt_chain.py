"""Regression for the receipt-chain collection server (protocol §3-4).

Self-contained. Asserts the integrity attacks are closed (receipt reuse,
sequence gap, verify replay) and that an honest streamed session passes, while
documenting — not fixing — the residual gaps (short-record burst and paced
real-time replay pass, because they are indistinguishable from a live human).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.receipt_chain import CollectionServer, protocol_gate

BATCH_MAX = 32


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


def _honest(ev, sid="h"):
    srv = CollectionServer(f"ch_{sid}", f"s_{sid}")
    prev, seq = srv.nonce, 0
    for w in _windows(ev):
        recv = w[-1]["t_ms"] + 15
        r, err = srv.submit_batch(seq, w, prev, recv, f"s_{sid}")
        assert err is None, err
        prev, seq = r, seq + 1
    return srv


def test_honest_session_passes():
    srv = _honest(_events(6000, 104))
    assert srv.verify() == "ok"
    ok, reasons = protocol_gate(srv)
    assert ok, reasons


def test_receipt_reuse_rejected():
    ev = _events(6000, 104)
    srv = CollectionServer("ch", "s")
    r, err = srv.submit_batch(0, ev[:32], "attacker-guess", 0.0, "s")
    assert err == "receipt_chain_broken"


def test_sequence_gap_rejected():
    ev = _events(6000, 104)
    srv = CollectionServer("ch", "s")
    r, _ = srv.submit_batch(0, ev[:32], srv.nonce, 0.0, "s")
    r2, err = srv.submit_batch(5, ev[32:64], r, 200.0, "s")
    assert err == "batch_sequence_gap"


def test_verify_replay_rejected():
    srv = _honest(_events(6000, 104))
    assert srv.verify() == "ok"
    assert srv.verify() == "nonce_reused"


def test_session_mismatch_rejected():
    srv = CollectionServer("ch", "s")
    _, err = srv.submit_batch(0, _events(6000, 40)[:32], srv.nonce, 0.0, "other-session")
    assert err == "session_mismatch"


def test_bulk_burst_replay_blocked_by_delivery():
    ev = _events(6000, 104)
    srv = CollectionServer("ch", "s")
    srv.submit_batch(0, ev, srv.nonce, 0.0, "s")   # whole thing, one batch
    srv.verify()
    ok, reasons = protocol_gate(srv)
    assert not ok


def test_residual_paced_replay_passes_documented_gap():
    # A patient attacker reproduces honest pacing with a valid chain -> passes.
    # This is the acknowledged ceiling of client-submitted-events protocols;
    # recorded here so a future change that claims to close it must update this.
    srv = _honest(_events(6000, 104))
    srv.verify()
    ok, _ = protocol_gate(srv)
    assert ok
