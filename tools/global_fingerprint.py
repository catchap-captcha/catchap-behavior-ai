"""Cross-session exact-fingerprint tracker — closes the session-scoping gap.

`recent_session_history` is deliberately session-scoped (so one student's path
never matches another's), which means an attacker who rotates `session_id` per
attempt evades replay detection entirely (red-team R13). A GLOBAL check on the
*exact* path fingerprint (SHA-256 of the rounded path, not DTW similarity)
catches literal replay across sessions while staying FRR-safe: across 3000
distinct human recordings there were 0 exact-fingerprint collisions, because two
genuine drags never produce a byte-identical path.

Privacy: stores only path hashes with a TTL, no session/participant identifiers.
This is a prototype of the shared-state logic; production backs it with a DB
table or cache (Redis) keyed by fingerprint. It intentionally does NOT catch
geometric transforms (mirror/rotate/resample) — those change the fingerprint and
are the job of captcha-side object-endpoint binding (R12).
"""
from __future__ import annotations

from collections import OrderedDict


class GlobalFingerprintTracker:
    """Bounded, TTL'd set of recently-seen exact path fingerprints."""

    def __init__(self, ttl_seconds: float = 900.0, max_entries: int = 100_000):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._seen: "OrderedDict[str, float]" = OrderedDict()

    def _evict(self, now: float) -> None:
        # drop expired
        while self._seen:
            fp, ts = next(iter(self._seen.items()))
            if now - ts > self.ttl:
                self._seen.popitem(last=False)
            else:
                break
        # drop oldest over capacity
        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)

    def check_and_record(self, fingerprint: str | None, now: float) -> bool:
        """Return True if this exact fingerprint was seen recently (any session).

        Records it either way. A None fingerprint (degenerate path) is never a
        cross-session replay and is not tracked.
        """
        if fingerprint is None:
            return False
        self._evict(now)
        hit = fingerprint in self._seen
        self._seen[fingerprint] = now
        self._seen.move_to_end(fingerprint)
        # enforce the capacity bound after inserting the new entry
        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)
        return hit


__all__ = ["GlobalFingerprintTracker"]
