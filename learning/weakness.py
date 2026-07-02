"""Weakness scoring and weakest-concept selection.

weakness = 0.45*(1-mastery) + 0.30*recent_wrong + 0.15*review_urgency + 0.10*hard_fail

Every term is normalized to [0,1]. Mastery is the primary driver; "review need"
(time since last study) is kept as a small, separate term so a merely-forgotten
concept is not confused with a genuinely weak one. Concepts with too few valid
attempts are not ranked as weak — they are reported as "diagnosis needed".
"""

from __future__ import annotations

from datetime import datetime

from learning.config import (
    HARD_DIFFICULTY,
    MIN_VALID_ATTEMPTS_FOR_CONFIDENCE,
    RECENT_WINDOW,
    REVIEW_SATURATION_DAYS,
    WEIGHT_HARD_FAIL,
    WEIGHT_MASTERY,
    WEIGHT_RECENT,
    WEIGHT_REVIEW,
)
from learning.models import CountedOutcome, MasteryResult, WeakConcept


def _recent_wrong_ratio(outcomes: list[CountedOutcome]) -> float:
    recent = sorted(outcomes, key=lambda o: o.answered_at)[-RECENT_WINDOW:]
    if not recent:
        return 0.0
    wrong = sum(1 for o in recent if not o.is_correct)
    return wrong / len(recent)


def _review_urgency(outcomes: list[CountedOutcome], now: datetime) -> float:
    if not outcomes:
        return 0.0
    last = max(o.answered_at for o in outcomes)
    days = (now - last).total_seconds() / 86400.0
    return max(0.0, min(days / REVIEW_SATURATION_DAYS, 1.0))


def _hard_fail_ratio(outcomes: list[CountedOutcome]) -> float:
    hard = [o for o in outcomes if o.difficulty >= HARD_DIFFICULTY]
    if not hard:
        return 0.0
    wrong = sum(1 for o in hard if not o.is_correct)
    return wrong / len(hard)


def weakness_for_concept(
    mastery: MasteryResult, outcomes: list[CountedOutcome], now: datetime
) -> WeakConcept:
    """Compute the weakness score and a human-readable reason for one concept."""
    inv_mastery = 1.0 - mastery.mastery
    recent = _recent_wrong_ratio(outcomes)
    review = _review_urgency(outcomes, now)
    hard = _hard_fail_ratio(outcomes)

    contributions = {
        "숙련도 낮음": WEIGHT_MASTERY * inv_mastery,
        f"최근 {RECENT_WINDOW}문제 중 오답 많음": WEIGHT_RECENT * recent,
        "마지막 학습 후 오래됨": WEIGHT_REVIEW * review,
        "어려운 문제 실패": WEIGHT_HARD_FAIL * hard,
    }
    score = sum(contributions.values())

    # reason = the dominant contributor(s)
    top = max(contributions, key=contributions.get)
    reason = top
    last_days = int((now - max(o.answered_at for o in outcomes)).total_seconds() / 86400) if outcomes else 0
    if top == "마지막 학습 후 오래됨":
        reason = f"마지막 학습 후 {last_days}일 경과 (복습 필요)"
    elif top.startswith("최근"):
        recent_n = min(len(outcomes), RECENT_WINDOW)
        wrong = sum(1 for o in sorted(outcomes, key=lambda o: o.answered_at)[-RECENT_WINDOW:] if not o.is_correct)
        reason = f"최근 {recent_n}문제 중 {wrong}문제 오답"

    return WeakConcept(
        concept_id=mastery.concept_id,
        mastery_score=round(mastery.mastery, 4),
        weakness_score=round(score, 4),
        reason=reason,
    )


def top_weak_concepts(
    mastery: dict[str, MasteryResult],
    outcomes_by_concept: dict[str, list[CountedOutcome]],
    now: datetime,
    k: int,
) -> tuple[list[WeakConcept], list[str]]:
    """Return (top-k weak concepts, concept_ids needing more diagnosis).

    Concepts with fewer than the confidence threshold of valid attempts are not
    ranked; they are returned separately as "diagnosis needed".
    """
    ranked: list[WeakConcept] = []
    diagnostic: list[str] = []
    for cid, m in mastery.items():
        if m.diagnostic_needed:
            diagnostic.append(cid)
            continue
        ranked.append(weakness_for_concept(m, outcomes_by_concept.get(cid, []), now))

    ranked.sort(key=lambda w: w.weakness_score, reverse=True)
    return ranked[:k], diagnostic
