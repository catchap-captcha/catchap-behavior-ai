"""Rule-based problem recommendation.

For each weak concept, pick questions whose difficulty matches the student's
mastery band (zone of proximal development — not too easy, not too hard):

    mastery < 0.40   -> easy
    0.40 .. 0.70     -> medium
    >= 0.70          -> hard (or move on to the next concept)

Extra rules: skip recently-solved questions, avoid stacking only-hard problems,
cap the set at N. This is the interim stand-in for the later XGBoost/IRT-driven
"pick the item with ~appropriate success probability" selection.
"""

from __future__ import annotations

from learning.config import (
    DEFAULT_N_RECOMMEND,
    EASY_MAX,
    MASTERED_THRESHOLD,
    MEDIUM_MAX,
)
from learning.models import MasteryResult, Question, Recommendation, WeakConcept


def difficulty_band(difficulty: float) -> str:
    if difficulty < EASY_MAX:
        return "easy"
    if difficulty < MEDIUM_MAX:
        return "medium"
    return "hard"


def target_band(mastery: float) -> str:
    """Which difficulty band suits this mastery level."""
    if mastery < EASY_MAX:
        return "easy"
    if mastery < MEDIUM_MAX:
        return "medium"
    return "hard"


def recommend(
    weak_concepts: list[WeakConcept],
    mastery: dict[str, MasteryResult],
    question_bank: list[Question],
    recently_solved: set[str] | None = None,
    n: int = DEFAULT_N_RECOMMEND,
) -> list[Recommendation]:
    """Recommend up to ``n`` questions across the weak concepts.

    Args:
        weak_concepts: ranked weakest concepts (from :mod:`learning.weakness`).
        mastery: mastery per concept.
        question_bank: candidate questions to choose from.
        recently_solved: question_ids to exclude (already practised recently).
        n: how many questions to return (design: 3~5).

    Returns:
        A recommendation list, roughly ordered by concept weakness, difficulty
        matched to mastery, with at most one "stretch" (harder) item in a row.
    """
    recently_solved = recently_solved or set()
    by_concept: dict[str, list[Question]] = {}
    for q in question_bank:
        by_concept.setdefault(q.concept_id, []).append(q)

    recs: list[Recommendation] = []
    last_was_hard = False

    for w in weak_concepts:
        if len(recs) >= n:
            break
        m = mastery.get(w.concept_id)
        band = target_band(m.mastery) if m else "easy"

        candidates = [
            q for q in by_concept.get(w.concept_id, [])
            if q.question_id not in recently_solved
        ]
        # prefer the target band, then adjacent bands (never jump straight to hard)
        order = _band_preference(band)
        candidates.sort(key=lambda q: order.index(difficulty_band(q.difficulty))
                        if difficulty_band(q.difficulty) in order else len(order))

        for q in candidates:
            if len(recs) >= n:
                break
            qb = difficulty_band(q.difficulty)
            if qb == "hard" and last_was_hard:
                continue  # do not stack consecutive hard problems
            recs.append(Recommendation(
                question_id=q.question_id,
                concept_id=w.concept_id,
                target_band=band,
                reason=f"취약 개념 '{w.concept_id}' ({w.reason}), 숙련도 {m.mastery:.0%} → {band} 문제",
                mastery_before=round(m.mastery, 4) if m else 0.0,
            ))
            recently_solved = recently_solved | {q.question_id}
            last_was_hard = qb == "hard"

    return recs


def _band_preference(band: str) -> list[str]:
    """Preferred difficulty order given a target band (avoid over-shooting)."""
    return {
        "easy": ["easy", "medium", "hard"],
        "medium": ["medium", "easy", "hard"],
        "hard": ["hard", "medium", "easy"],
    }[band]


def next_concept_ready(mastery: MasteryResult) -> bool:
    """True if the student has effectively mastered a concept (move on)."""
    return not mastery.diagnostic_needed and mastery.mastery >= MASTERED_THRESHOLD
