"""End-to-end weak-problem recommendation (rule-based, phase 1).

    raw attempts
        -> resolve operation errors + retries   (operation_error)
        -> mastery per concept (guess-corrected) (mastery)
        -> weakest concepts Top-K               (weakness)
        -> recommend questions                  (recommender)

One call, no DB, no ML — works from the very first solved problem. The closed
loop (design step 6) happens by re-running :func:`diagnose` after the student
solves the recommended questions: mastery is simply recomputed from the larger
set of attempts, so mastery_before (in the recommendation) vs the new mastery
gives the before/after change.
"""

from __future__ import annotations

from datetime import datetime

from learning.config import DEFAULT_N_RECOMMEND, DEFAULT_TOP_K_WEAK
from learning.mastery import mastery_by_concept
from learning.models import CountedOutcome, DiagnoseResult, Question, RawAttempt
from learning.operation_error import resolve_all
from learning.recommender import recommend
from learning.weakness import top_weak_concepts


def diagnose(
    student_id: str,
    attempts: list[RawAttempt],
    question_bank: list[Question],
    now: datetime,
    recently_solved: set[str] | None = None,
    k_weak: int = DEFAULT_TOP_K_WEAK,
    n_recommend: int = DEFAULT_N_RECOMMEND,
) -> DiagnoseResult:
    """Run the full rule-based diagnosis + recommendation for one student.

    Args:
        student_id: the learner.
        attempts: all of this student's raw attempts.
        question_bank: candidate questions to recommend from.
        now: current time (for review-urgency; passed in for determinism/testing).
        recently_solved: question_ids to exclude from recommendation.
        k_weak: number of weak concepts to target.
        n_recommend: number of questions to recommend.

    Returns:
        :class:`DiagnoseResult` with mastery, weak concepts, diagnosis-needed
        concepts, and recommendations.
    """
    # 1) raw -> counted concept-level outcomes (operation errors excluded)
    counted = resolve_all(attempts)

    # 2) mastery per concept
    mastery = mastery_by_concept(counted)

    # 3) weakest concepts (+ concepts that still need diagnosis)
    outcomes_by_concept: dict[str, list[CountedOutcome]] = {}
    for o in counted:
        outcomes_by_concept.setdefault(o.concept_id, []).append(o)
    weak, diagnostic = top_weak_concepts(mastery, outcomes_by_concept, now, k_weak)

    # 4) recommend
    recs = recommend(weak, mastery, question_bank, recently_solved, n_recommend)

    return DiagnoseResult(
        student_id=student_id,
        mastery=mastery,
        weak_concepts=weak,
        diagnostic_needed=diagnostic,
        recommendations=recs,
    )
