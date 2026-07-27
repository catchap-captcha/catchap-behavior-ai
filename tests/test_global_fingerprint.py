"""Regression for the cross-session exact-fingerprint tracker (red-team R13)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.global_fingerprint import GlobalFingerprintTracker


def test_same_fingerprint_across_sessions_is_caught():
    t = GlobalFingerprintTracker()
    # first submission from "session A" — new, not a replay
    assert t.check_and_record("fp_abc", now=1000.0) is False
    # same exact path from a fresh "session B" — caught despite session rotation
    assert t.check_and_record("fp_abc", now=1005.0) is True


def test_distinct_fingerprints_never_collide():
    t = GlobalFingerprintTracker()
    for i in range(500):
        assert t.check_and_record(f"fp_{i}", now=1000.0 + i) is False


def test_expired_fingerprint_is_forgotten():
    t = GlobalFingerprintTracker(ttl_seconds=60.0)
    assert t.check_and_record("fp_x", now=1000.0) is False
    # same path well after the TTL — treated as new (bounded memory)
    assert t.check_and_record("fp_x", now=1000.0 + 61.0) is False


def test_none_fingerprint_is_ignored():
    t = GlobalFingerprintTracker()
    assert t.check_and_record(None, now=1000.0) is False
    assert t.check_and_record(None, now=1001.0) is False


def test_capacity_bound_is_enforced():
    t = GlobalFingerprintTracker(max_entries=10)
    for i in range(50):
        t.check_and_record(f"fp_{i}", now=1000.0 + i)
    assert len(t._seen) <= 10
