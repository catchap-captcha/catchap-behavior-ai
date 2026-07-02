"""Operation-error vs concept-error judgment (+ retry disambiguation).

Because the student drags a *labelled* tile, the tile they grabbed reveals the
answer they intended. Combined with where they dropped it, most cases resolve
with simple rules; the rest are held as ambiguous and settled by a retry.

Key safeguard: an operation-error attempt is fully excluded from mastery, and a
question presentation contributes only its FIRST concept-level outcome — so a
retry cannot inflate mastery (a wrong-then-right retry still counts as wrong).
"""

from __future__ import annotations

from learning.config import RETRY_MAX, RETRY_WINDOW_S
from learning.models import (
    CONCEPT_LEVEL,
    CountedOutcome,
    Judgment,
    Outcome,
    RawAttempt,
)


def operation_error_probability(a: RawAttempt) -> float:
    """Soft 0..1 signal that a slip occurred, from drag mechanics.

    Used only as a secondary hint for AMBIGUOUS cases — never to override the
    retry evidence. Reuses the same mechanics the CAPTCHA behavioral pipeline
    already extracts (regrab / failed drop / cancel / drop error).
    """
    signals = []
    signals.append(min(a.pointercancel_count, 3) / 3.0)
    signals.append(min(a.failed_drop_count, 3) / 3.0)
    signals.append(min(a.regrab_count, 3) / 3.0)
    if a.final_drop_error_px is not None:
        # 0px perfect .. >=60px clearly off
        signals.append(min(max(a.final_drop_error_px, 0.0), 60.0) / 60.0)
    return round(sum(signals) / len(signals), 4) if signals else 0.0


def classify_attempt(a: RawAttempt) -> Judgment:
    """Judge one raw attempt into an :class:`Outcome`.

    Rules (in order):
      1. system error                          -> SYSTEM_ERROR (excluded)
      2. input cancelled / nothing grabbed /
         dropped outside any valid area (None)  -> OPERATION_ERROR (excluded)
      3. dropped in the answer slot:
           grabbed correct tile -> CORRECT
           grabbed wrong tile   -> CONCEPT_ERROR
      4. anything else (e.g. dropped in a
         non-slot spot)                         -> AMBIGUOUS (held, retry)
    """
    if a.system_error:
        return Judgment(Outcome.SYSTEM_ERROR, False, None, 1.0, ["시스템 오류"])

    if a.pointercancel_count > 0:
        return Judgment(Outcome.OPERATION_ERROR, False, None, 1.0, ["입력 취소(pointercancel)"])

    if a.grabbed_answer_id is None:
        return Judgment(Outcome.OPERATION_ERROR, False, None, 1.0, ["아무 타일도 잡지 않음"])

    if a.released_target_id is None:
        return Judgment(Outcome.OPERATION_ERROR, False, None, 1.0, ["유효 영역 밖 드롭(드롭 실패)"])

    if a.released_target_id == a.answer_slot_id:
        if a.intended_correct:
            return Judgment(Outcome.CORRECT, True, True, 1.0, ["정답 타일을 슬롯에 정상 드롭"])
        return Judgment(Outcome.CONCEPT_ERROR, True, False, 1.0, ["오답 타일을 슬롯에 정상 드롭"])

    # dropped somewhere that is neither the slot nor a total miss
    return Judgment(
        Outcome.AMBIGUOUS, False, None,
        operation_error_probability(a),
        ["엉뚱한 위치에 드롭 — 판정 보류, 재시도 필요"],
    )


def resolve_presentation(attempts: list[RawAttempt], max_retries: int = RETRY_MAX) -> CountedOutcome | None:
    """Collapse all attempts of one question presentation into ONE counted result.

    Attempts are ordered by time. Operation/system/ambiguous attempts are
    skipped; the first concept-level (CORRECT or CONCEPT_ERROR) attempt within
    the retry cap is what counts. Returns None if the presentation yields no
    concept-level attempt (fully excluded from mastery).
    """
    ordered = sorted(attempts, key=lambda a: a.answered_at)
    considered = ordered[: max_retries + 1]

    for idx, a in enumerate(considered):
        j = classify_attempt(a)
        if j.outcome in CONCEPT_LEVEL:
            reasons = list(j.reasons)
            if idx > 0:
                reasons.append(f"{idx}회 재시도 후 개념 판정")
            return CountedOutcome(
                concept_id=a.concept_id,
                question_id=a.question_id,
                difficulty=a.difficulty,
                answer_options_count=a.answer_options_count,
                is_correct=bool(j.is_correct),
                answered_at=a.answered_at,
                outcome=j.outcome,
                reasons=reasons,
            )
    return None


def group_presentations(attempts: list[RawAttempt]) -> list[list[RawAttempt]]:
    """Group attempts by presentation_id (retries of the same shown question).

    Attempts without a presentation_id are each their own group.
    """
    groups: dict[str, list[RawAttempt]] = {}
    singletons: list[list[RawAttempt]] = []
    for a in attempts:
        if a.presentation_id is None:
            singletons.append([a])
        else:
            groups.setdefault(a.presentation_id, []).append(a)
    return list(groups.values()) + singletons


def resolve_all(attempts: list[RawAttempt]) -> list[CountedOutcome]:
    """Judge + retry-resolve every presentation, dropping fully-excluded ones."""
    counted = [resolve_presentation(g) for g in group_presentations(attempts)]
    return [c for c in counted if c is not None]
