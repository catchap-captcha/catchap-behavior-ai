"""Advisory risk scoring for model, replay, and session-rate signals.

This module intentionally does not decide whether a request is accepted or
blocked.  It returns an explainable policy score and a recommended next step;
the CAPTCHA backend owns the final decision and challenge lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.replay_detector import ReplayFeatures


@dataclass(frozen=True)
class RiskFusionPolicy:
    """Auditable initial thresholds for an advisory risk assessment.

    The weights make an exact replay a strong signal, while a low ML score by
    itself can only request a step-up challenge. They are policy values, not a
    claim that ``risk_score`` is a calibrated bot probability.
    """

    model_human_threshold: float
    dtw_similarity_threshold: float
    max_attempts_per_minute: float
    # Scores in this band are not automatically rejected. They request a
    # second verification step before an otherwise automatic allow decision.
    # It is calibrated from development OOF human groups, never red-team data.
    step_up_human_threshold: float | None = None
    exact_replay_enabled: bool = True
    model_weight: float = 50.0
    # DTW similarity and request bursts are each insufficient on their own,
    # but together should reach high-risk step-up handling.
    dtw_weight: float = 35.0
    session_rate_weight: float = 30.0
    exact_replay_floor: float = 80.0
    medium_risk_threshold: float = 40.0
    high_risk_threshold: float = 70.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.model_human_threshold <= 1.0:
            raise ValueError("model_human_threshold must be between 0 and 1")
        if self.step_up_human_threshold is not None and not (
            self.model_human_threshold <= self.step_up_human_threshold <= 1.0
        ):
            raise ValueError(
                "step_up_human_threshold must be between model_human_threshold and 1"
            )
        if not 0.0 <= self.dtw_similarity_threshold <= 1.0:
            raise ValueError("dtw_similarity_threshold must be between 0 and 1")
        if self.max_attempts_per_minute <= 0:
            raise ValueError("max_attempts_per_minute must be positive")
        if not 0.0 <= self.medium_risk_threshold < self.high_risk_threshold <= 100.0:
            raise ValueError("risk thresholds must satisfy 0 <= medium < high <= 100")


@dataclass(frozen=True)
class RiskFusionDecision:
    risk_score: float
    risk_level: str
    recommended_action: str
    reasons: tuple[str, ...]
    human_score: float
    bot_risk_score: float
    exact_replay_detected: bool
    path_similarity_score: float
    attempts_per_minute: float


def fuse_behavior_risk(
    human_score: float,
    replay: ReplayFeatures,
    policy: RiskFusionPolicy,
) -> RiskFusionDecision:
    """Return an explainable, non-blocking risk assessment for one attempt."""
    score = float(human_score)
    if not 0.0 <= score <= 1.0:
        raise ValueError("human_score must be between 0 and 1")

    reasons: list[str] = []
    # A model score is evidence, not a final verdict. On its own it cannot
    # raise risk above medium, because the adversarial replay evaluation showed
    # the first-stage model is not sufficient for hard enforcement.
    risk_score = (1.0 - score) * policy.model_weight
    if score < policy.model_human_threshold:
        reasons.append("ml_bot_score")
        risk_score = max(risk_score, policy.medium_risk_threshold)
    elif (
        policy.step_up_human_threshold is not None
        and score < policy.step_up_human_threshold
    ):
        reasons.append("ml_step_up_band")
        risk_score = max(risk_score, policy.medium_risk_threshold)
    if policy.exact_replay_enabled and replay.exact_replay_detected:
        reasons.append("exact_trace_fingerprint")
        risk_score = max(risk_score, policy.exact_replay_floor)
    if replay.path_similarity_score >= policy.dtw_similarity_threshold:
        reasons.append("dtw_similar_trace")
        risk_score += policy.dtw_weight
    if replay.attempts_per_minute >= policy.max_attempts_per_minute:
        reasons.append("session_rate_exceeded")
        risk_score += policy.session_rate_weight

    risk_score = round(min(100.0, max(0.0, risk_score)), 2)
    if risk_score >= policy.high_risk_threshold:
        risk_level = "high"
        recommended_action = "step_up_and_rate_limit"
    elif risk_score >= policy.medium_risk_threshold:
        risk_level = "medium"
        recommended_action = "step_up"
    else:
        risk_level = "low"
        recommended_action = "allow"

    return RiskFusionDecision(
        risk_score=risk_score,
        risk_level=risk_level,
        recommended_action=recommended_action,
        reasons=tuple(reasons),
        human_score=score,
        bot_risk_score=1.0 - score,
        exact_replay_detected=replay.exact_replay_detected,
        path_similarity_score=float(replay.path_similarity_score),
        attempts_per_minute=float(replay.attempts_per_minute),
    )


__all__ = ["RiskFusionDecision", "RiskFusionPolicy", "fuse_behavior_risk"]
