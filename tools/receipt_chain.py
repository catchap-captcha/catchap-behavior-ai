"""Receipt-chain collection server — prototype for BEHAVIOR_EVENT_TRUST_PROTOCOL §3-4.

Simulates the server-side state machine that collects behavior-event batches
during a challenge: a one-time nonce seeds a chain, each batch must carry the
previous server-issued receipt and the next sequence number, and verify
consumes the nonce exactly once. The receipt is an HMAC the browser cannot
forge (the secret never leaves the server).

Combined with tools.delivery_binding, this closes the protocol-integrity
attacks (receipt reuse, sequence gap/reversal, verify replay) and the bulk
offline replay (burst / fast-chunk). It does NOT close a patient real-time
paced replay of a genuine human recording, nor a short (<stream-expectation)
recording replayed as a burst — those are indistinguishable from a live human
at every protocol layer (see protocol doc §1). Not wired into the online path;
the collection transport does not exist yet.
"""
from __future__ import annotations

import hashlib
import hmac

from tools.delivery_binding import evaluate_delivery

_SERVER_SECRET = b"server-only-secret-never-in-browser"


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _receipt(challenge_id, session_hash, seq, payload_hash, prev_receipt, recv_at) -> str:
    msg = f"{challenge_id}|{session_hash}|{seq}|{payload_hash}|{prev_receipt}|{recv_at}"
    return hmac.new(_SERVER_SECRET, msg.encode(), hashlib.sha256).hexdigest()


class CollectionServer:
    """Per-challenge collection state. All timestamps in ms."""

    def __init__(self, challenge_id: str, session_id: str):
        self.challenge_id = challenge_id
        self.session_hash = _h(session_id)
        self.nonce = _h(challenge_id, session_id, "nonce")
        self.nonce_live = True
        self.batches: list[tuple[int, str, float, list[float]]] = []
        self.last_receipt = self.nonce

    def submit_batch(self, seq, events, prev_receipt, recv_at, session_id):
        """Store one batch; return (receipt, None) or (None, error_reason)."""
        if _h(session_id) != self.session_hash:
            return None, "session_mismatch"
        if not self.nonce_live:
            return None, "challenge_consumed"
        if seq != len(self.batches):
            return None, "batch_sequence_gap"
        if prev_receipt != self.last_receipt:
            return None, "receipt_chain_broken"
        payload_hash = _h(*(f"{e['t_ms']}:{e['x']}:{e['y']}" for e in events))
        recp = _receipt(self.challenge_id, self.session_hash, seq, payload_hash,
                        prev_receipt, recv_at)
        self.batches.append((seq, payload_hash, recv_at, [e["t_ms"] for e in events]))
        self.last_receipt = recp
        return recp, None

    def delivery_metadata(self):
        return [{"server_received_at_ms": b[2], "client_t_ms": b[3]} for b in self.batches]

    def verify(self):
        """Consume the nonce exactly once."""
        if not self.nonce_live:
            return "nonce_reused"
        self.nonce_live = False
        return "ok"


def protocol_gate(server: CollectionServer):
    """Return (ok, reasons). ok=False -> caller floors recommendation at step_up."""
    if not server.batches:
        return False, ["no_server_batches"]
    ok, reason, _ = evaluate_delivery(server.delivery_metadata())
    return (ok, [] if ok else [reason])


__all__ = ["CollectionServer", "protocol_gate"]
