"""Delivery-timing binding — a prototype for BEHAVIOR_EVENT_TRUST_PROTOCOL §6.

Pure function over the batch metadata the server records during collection
(server_received_at + the client-embedded timestamps of each batch). It does
NOT judge whether the motion is human; it checks only that the events were
delivered progressively, consistent with their own client timeline.

A trajectory that is pre-computed offline and dumped (or fast-chunked) at
verify time fails this check even when the model scores it as human. A drag
that is genuinely streamed passes. A patient real-time paced replay is
indistinguishable at this layer and still passes — see the protocol doc §1;
closing that needs per-event server sequencing plus the nonce/receipt chain,
not this signal alone.

Not yet wired into the online predict path: the online API does not yet
receive per-batch arrival metadata. This module encodes the design and its
validated thresholds so the loop result is reproducible (tests) and ready to
wire once the collection transport exists.
"""
from __future__ import annotations

from statistics import mean
from typing import Sequence, TypedDict


class Batch(TypedDict):
    server_received_at_ms: float
    client_t_ms: Sequence[float]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def evaluate_delivery(batches: Sequence[Batch],
                      min_delivery_ratio: float = 0.35,
                      min_corr: float = 0.80,
                      min_batches: int = 3,
                      stream_expectation_ms: float = 800.0):
    """Return (ok, reason, metrics).

    ok=False means the delivery shape is inconsistent with progressive
    streaming and the caller should floor the recommendation at step_up,
    regardless of model score.

    A drag whose whole client timeline is shorter than stream_expectation_ms
    is too brief to have produced several 150ms batches, so the burst / batch
    checks are skipped for it (they would false-positive on fast human drags).
    This exemption is a known residual: a short recorded human drag replayed
    as a burst passes this layer — quantified in the protocol doc.
    """
    batches = [b for b in batches if b.get("client_t_ms")]
    if not batches:
        return False, "no_server_batches", {}
    n_batches = len(batches)

    client_all = [t for b in batches for t in b["client_t_ms"]]
    client_span = max(client_all) - min(client_all)
    srv = [b["server_received_at_ms"] for b in batches]
    server_span = max(srv) - min(srv)

    if client_span < stream_expectation_ms:
        return True, None, {"n_batches": n_batches,
                            "client_span_ms": round(client_span),
                            "short_exempt": True}

    if n_batches < min_batches:
        return False, "too_few_batches_for_timeline", {
            "n_batches": n_batches, "client_span_ms": round(client_span)}

    delivery_ratio = server_span / client_span if client_span > 0 else 0.0
    corr = _pearson([mean(b["client_t_ms"]) for b in batches], srv)
    metrics = {"n_batches": n_batches, "client_span_ms": round(client_span),
               "server_span_ms": round(server_span),
               "delivery_ratio": round(delivery_ratio, 3), "corr": round(corr, 3)}

    if delivery_ratio < min_delivery_ratio:
        return False, "events_delivered_in_burst", metrics
    if corr < min_corr:
        return False, "delivery_timeline_uncorrelated", metrics
    return True, None, metrics


__all__ = ["evaluate_delivery", "Batch"]
