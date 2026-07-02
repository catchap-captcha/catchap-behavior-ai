"""Tests for the rule-based weak-problem recommendation package."""

from __future__ import annotations

from datetime import datetime, timedelta

from learning.mastery import concept_mastery, mastery_by_concept
from learning.models import CountedOutcome, Outcome, Question, RawAttempt
from learning.operation_error import classify_attempt, resolve_all, resolve_presentation
from learning.recommender import difficulty_band, recommend, target_band
from learning.service import diagnose
from learning.weakness import top_weak_concepts, weakness_for_concept

BASE = datetime(2026, 1, 1, 9, 0, 0)


def mk(qid="q1", concept="C1", correct="A", grabbed="A", released="slot",
       difficulty=0.3, options=3, t=0, presentation=None, **kw) -> RawAttempt:
    return RawAttempt(
        attempt_id=f"a{t}", student_id="s1", question_id=qid, concept_id=concept,
        difficulty=difficulty, answer_options_count=options, correct_answer_id=correct,
        answered_at=BASE + timedelta(seconds=t), grabbed_answer_id=grabbed,
        released_target_id=released, presentation_id=presentation, **kw,
    )


def mkc(concept="C1", correct=True, difficulty=0.3, options=3, t=0) -> CountedOutcome:
    return CountedOutcome(
        concept_id=concept, question_id=f"q{t}", difficulty=difficulty,
        answer_options_count=options, is_correct=correct,
        answered_at=BASE + timedelta(seconds=t), outcome=Outcome.CORRECT,
    )


# --------------------------------------------------------------------------- #
# operation-error judgment
# --------------------------------------------------------------------------- #
def test_correct_clean_drop():
    j = classify_attempt(mk(grabbed="A", correct="A", released="slot"))
    assert j.outcome is Outcome.CORRECT and j.valid_for_learning and j.is_correct


def test_concept_error_wrong_tile_clean_drop():
    j = classify_attempt(mk(grabbed="B", correct="A", released="slot"))
    assert j.outcome is Outcome.CONCEPT_ERROR and j.valid_for_learning and j.is_correct is False


def test_operation_error_on_cancel():
    j = classify_attempt(mk(pointercancel_count=1))
    assert j.outcome is Outcome.OPERATION_ERROR and not j.valid_for_learning


def test_operation_error_on_failed_drop():
    j = classify_attempt(mk(released=None))
    assert j.outcome is Outcome.OPERATION_ERROR and not j.valid_for_learning


def test_operation_error_on_no_grab():
    j = classify_attempt(mk(grabbed=None))
    assert j.outcome is Outcome.OPERATION_ERROR


def test_ambiguous_on_wrong_location():
    j = classify_attempt(mk(grabbed="A", correct="A", released="somewhere_else"))
    assert j.outcome is Outcome.AMBIGUOUS and not j.valid_for_learning


def test_system_error_excluded():
    j = classify_attempt(mk(system_error=True))
    assert j.outcome is Outcome.SYSTEM_ERROR and not j.valid_for_learning


# --------------------------------------------------------------------------- #
# retry resolution (anti-gaming)
# --------------------------------------------------------------------------- #
def test_operation_then_correct_counts_as_correct():
    # 1st: grabbed correct but drop failed (operation error) -> excluded
    # 2nd: correct clean -> this counts
    attempts = [
        mk(grabbed="A", correct="A", released=None, t=0, presentation="p1"),
        mk(grabbed="A", correct="A", released="slot", t=3, presentation="p1"),
    ]
    counted = resolve_presentation(attempts)
    assert counted is not None and counted.is_correct is True


def test_wrong_then_right_retry_still_counts_wrong():
    # first concept-level outcome (wrong) is what counts; retry-correct is ignored
    attempts = [
        mk(grabbed="B", correct="A", released="slot", t=0, presentation="p2"),
        mk(grabbed="A", correct="A", released="slot", t=2, presentation="p2"),
    ]
    counted = resolve_presentation(attempts)
    assert counted is not None and counted.is_correct is False


def test_retry_cap_excludes_when_only_operation_errors():
    # 3 operation errors within the cap, concept-level only on the 4th -> excluded
    attempts = [
        mk(released=None, t=0, presentation="p3"),
        mk(released=None, t=1, presentation="p3"),
        mk(released=None, t=2, presentation="p3"),
        mk(grabbed="A", correct="A", released="slot", t=3, presentation="p3"),
    ]
    assert resolve_presentation(attempts) is None


# --------------------------------------------------------------------------- #
# mastery (smoothing + guessing correction)
# --------------------------------------------------------------------------- #
def test_smoothing_prevents_one_shot_full_mastery():
    m = concept_mastery("C1", [mkc(correct=True, options=100, t=0)])
    assert m.mastery < 0.7            # not 100% from a single correct
    assert m.diagnostic_needed        # 1 < 3 attempts


def test_guess_correction_two_options_lower_than_four():
    # same 50% observed accuracy, but 2-option guessing baseline is higher
    two = concept_mastery("C1", [mkc(correct=(i < 5), options=2, t=i) for i in range(10)])
    four = concept_mastery("C1", [mkc(correct=(i < 5), options=4, t=i) for i in range(10)])
    assert two.mastery < four.mastery
    assert two.guess_baseline == 0.5 and four.guess_baseline == 0.25


def test_open_ended_has_no_guess_correction():
    m = concept_mastery("C1", [mkc(correct=True, options=0, t=i) for i in range(5)])
    assert m.guess_baseline == 0.0


# --------------------------------------------------------------------------- #
# weakness
# --------------------------------------------------------------------------- #
def test_low_mastery_scores_more_weak_than_high():
    now = BASE + timedelta(days=1)
    low_outcomes = [mkc(correct=(i < 2), t=i) for i in range(10)]     # 20% correct
    high_outcomes = [mkc(correct=(i < 9), t=i) for i in range(10)]    # 90% correct
    low_m = concept_mastery("C1", low_outcomes)
    high_m = concept_mastery("C2", high_outcomes)
    w_low = weakness_for_concept(low_m, low_outcomes, now)
    w_high = weakness_for_concept(high_m, high_outcomes, now)
    assert w_low.weakness_score > w_high.weakness_score


def test_diagnostic_concepts_excluded_from_ranking():
    now = BASE + timedelta(days=1)
    outcomes = {"C1": [mkc(concept="C1", correct=False, t=0)]}  # only 1 attempt
    mastery = mastery_by_concept(outcomes["C1"])
    weak, diagnostic = top_weak_concepts(mastery, outcomes, now, k=3)
    assert weak == [] and "C1" in diagnostic


def test_review_urgency_reason_when_stale():
    # well-mastered concept (all correct, 10 attempts) but studied long ago:
    # the "review needed" term should dominate, not "low mastery".
    now = BASE + timedelta(days=25)
    outcomes = [mkc(correct=True, options=4, t=i) for i in range(10)]
    m = concept_mastery("C1", outcomes)
    w = weakness_for_concept(m, outcomes, now)
    assert "복습" in w.reason


# --------------------------------------------------------------------------- #
# recommender
# --------------------------------------------------------------------------- #
def test_bands():
    assert difficulty_band(0.2) == "easy"
    assert difficulty_band(0.5) == "medium"
    assert difficulty_band(0.9) == "hard"
    assert target_band(0.3) == "easy" and target_band(0.6) == "medium" and target_band(0.8) == "hard"


def test_recommend_matches_band_and_excludes_solved():
    now = BASE + timedelta(days=1)
    outcomes = [mkc(concept="C1", correct=(i < 2), t=i) for i in range(10)]  # weak, low mastery
    mastery = mastery_by_concept(outcomes)
    weak, _ = top_weak_concepts(mastery, {"C1": outcomes}, now, k=3)
    bank = [
        Question("easy1", "C1", 0.2, 3, "A"),
        Question("hard1", "C1", 0.9, 3, "A"),
        Question("solved", "C1", 0.2, 3, "A"),
    ]
    recs = recommend(weak, mastery, bank, recently_solved={"solved"}, n=3)
    ids = [r.question_id for r in recs]
    assert "solved" not in ids
    assert ids[0] == "easy1"  # low mastery -> easy preferred first


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_diagnose_end_to_end():
    now = BASE + timedelta(days=1)
    attempts = []
    # C1: weak (mostly wrong), C2: strong (mostly right)
    for i in range(6):
        attempts.append(mk(qid=f"c1_{i}", concept="C1", grabbed=("A" if i < 1 else "B"),
                           correct="A", released="slot", difficulty=0.3, t=i))
    for i in range(6):
        attempts.append(mk(qid=f"c2_{i}", concept="C2", grabbed=("A" if i < 5 else "B"),
                           correct="A", released="slot", difficulty=0.3, t=100 + i))
    bank = [Question(f"new_c1_{i}", "C1", 0.2, 3, "A") for i in range(3)]

    result = diagnose("s1", attempts, bank, now)
    assert "C1" in result.mastery and "C2" in result.mastery
    assert result.mastery["C1"].mastery < result.mastery["C2"].mastery
    assert result.weak_concepts and result.weak_concepts[0].concept_id == "C1"
    assert result.recommendations and all(r.concept_id == "C1" for r in result.recommendations)
