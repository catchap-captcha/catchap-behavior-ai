"""How the trust-protocol layers compose with the model in the online path.

A design sketch (not wired into app/) showing the decision order once the
batch-collection transport exists. The ordering is the point: the protocol
gate runs FIRST and can only *lower* the recommendation, never raise it above
what the model+risk policy would allow. A passing protocol gate does not by
itself allow — the model and risk fusion still decide.

    protocol_gate(collection)      # delivery binding + receipt chain (J-6)
      fail  -> floor at step_up, reason=invalid_event_telemetry
      pass  -> model score -> risk_fusion -> allow / step_up / step_up+rl

This closes: bulk offline replay, receipt/sequence/nonce integrity attacks.
This does NOT close (documented root limits, J-6/J-7): short (<800ms) recorded
drags replayed as burst, and patient paced real-time replay. Those are handled
operationally (session risk accrual, attempt caps, rate limits), not here.
"""
from __future__ import annotations

from tools.receipt_chain import CollectionServer, protocol_gate

_MEDIUM = 40.0


def decide(server: CollectionServer, model_human_score: float,
           step_up_human_threshold: float = 0.9999955746270602,
           human_threshold: float = 0.9999500521595306):
    """Return (recommended_action, risk_score, reasons).

    Mirrors risk_fusion's bands but with the protocol gate applied first.
    """
    reasons: list[str] = []
    ok, gate_reasons = protocol_gate(server)

    # model/risk banding (simplified from app/services/risk_fusion.py)
    if model_human_score < human_threshold:
        reasons.append("ml_bot_score")
        risk = _MEDIUM
    elif model_human_score < step_up_human_threshold:
        reasons.append("ml_step_up_band")
        risk = _MEDIUM
    else:
        risk = 0.0

    # protocol gate can only lower the outcome (floor at step_up)
    if not ok:
        reasons.append("invalid_event_telemetry")
        reasons.extend(gate_reasons)
        risk = max(risk, _MEDIUM)

    if risk >= _MEDIUM:
        action = "step_up"
    else:
        action = "allow"
    return action, risk, reasons


__all__ = ["decide"]
