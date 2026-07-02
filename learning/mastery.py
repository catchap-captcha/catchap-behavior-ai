"""Rule-based mastery estimation (guessing-corrected).

Two corrections stacked, because they fix different problems:
  1. additive smoothing  -> guards against overconfidence from FEW attempts
  2. guessing correction -> guards against the illusion from FEW answer options

Guess baseline is derived per question from ``answer_options_count`` (1/N), so a
2-choice item (50% chance) and a 4-choice item (25% chance) are handled
correctly, and open-ended items (no choices) get no guess correction. This is the
interim rule-based stand-in for BKT's guess parameter / IRT's guessing modelled
later.
"""

from __future__ import annotations

from statistics import mean

from learning.config import (
    MIN_VALID_ATTEMPTS_FOR_CONFIDENCE,
    SMOOTHING_CORRECT,
    SMOOTHING_TOTAL,
)
from learning.models import CountedOutcome, MasteryResult


def _guess_rate(options: int) -> float:
    """Chance of a correct guess for one item. Open-ended (<=1) -> 0."""
    return 1.0 / options if options and options >= 2 else 0.0


def concept_mastery(concept_id: str, outcomes: list[CountedOutcome]) -> MasteryResult:
    """Compute guessing-corrected mastery for one concept.

        observed = (correct + 2) / (valid + 4)                 # smoothing
        mastery  = (observed - guess) / (1 - guess), clamped>=0 # guess correction

    where ``guess`` is the mean per-item guess rate over the attempted questions.
    """
    n = len(outcomes)
    correct = sum(1 for o in outcomes if o.is_correct)

    observed = (correct + SMOOTHING_CORRECT) / (n + SMOOTHING_TOTAL)
    guess = mean(_guess_rate(o.answer_options_count) for o in outcomes) if n else 0.0

    if guess >= 1.0:
        mastery = observed
    else:
        mastery = (observed - guess) / (1.0 - guess)
    mastery = max(0.0, min(1.0, mastery))

    return MasteryResult(
        concept_id=concept_id,
        mastery=round(mastery, 4),
        valid_attempts=n,
        correct=correct,
        guess_baseline=round(guess, 4),
        diagnostic_needed=n < MIN_VALID_ATTEMPTS_FOR_CONFIDENCE,
    )


def mastery_by_concept(outcomes: list[CountedOutcome]) -> dict[str, MasteryResult]:
    """Group counted outcomes by concept and compute mastery for each."""
    grouped: dict[str, list[CountedOutcome]] = {}
    for o in outcomes:
        grouped.setdefault(o.concept_id, []).append(o)
    return {cid: concept_mastery(cid, items) for cid, items in grouped.items()}
