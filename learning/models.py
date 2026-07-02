"""Domain data models for weak-problem recommendation.

Pure dataclasses — no DB, no framework. The whole rule-based pipeline runs on
these, so it is fully testable without a database or any collected data.

The design is unified with the drag CAPTCHA: one drag both authenticates a human
(bot detection) and records a learning answer. The fields below capture the
"WHAT they chose" side (answer content) plus just enough drag mechanics to tell
an operation slip apart from a concept mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Outcome(str, Enum):
    """Result of judging one raw attempt."""

    CORRECT = "correct"                 # right answer, clean drop
    CONCEPT_ERROR = "concept_error"     # wrong answer, clean drop -> counts as wrong
    OPERATION_ERROR = "operation_error"  # drag/drop slip -> excluded from mastery
    SYSTEM_ERROR = "system_error"       # app/hardware error -> excluded
    AMBIGUOUS = "ambiguous"             # unclear -> held, retry offered


# Outcomes that reflect the student's actual understanding (feed mastery).
CONCEPT_LEVEL = {Outcome.CORRECT, Outcome.CONCEPT_ERROR}


@dataclass
class Question:
    """Static metadata for one question (assumed to already exist in content)."""

    question_id: str
    concept_id: str
    difficulty: float                   # 0.0 (easy) .. 1.0 (hard)
    answer_options_count: int           # tiles/choices; 0 or 1 => open-ended (no guessing)
    correct_answer_id: str
    answer_slot_id: str = "slot"        # id representing the valid answer drop area


@dataclass
class RawAttempt:
    """One raw problem-solving attempt from the drag interaction."""

    attempt_id: str
    student_id: str
    question_id: str
    concept_id: str
    difficulty: float
    answer_options_count: int
    correct_answer_id: str
    answered_at: datetime

    # what the student did
    grabbed_answer_id: str | None       # which tile they picked up == intended answer
    released_target_id: str | None      # where dropped; None == drop failed (out of area)
    answer_slot_id: str = "slot"

    # drag mechanics (reused from the CAPTCHA behavioral signals)
    pointercancel_count: int = 0
    regrab_count: int = 0
    failed_drop_count: int = 0
    retry_count: int = 0
    final_drop_error_px: float | None = None
    response_time_ms: int | None = None
    system_error: bool = False

    # groups retries of the SAME presented question together
    presentation_id: str | None = None

    @property
    def intended_correct(self) -> bool:
        """True if the tile they grabbed is the correct answer."""
        return self.grabbed_answer_id is not None and self.grabbed_answer_id == self.correct_answer_id


@dataclass
class Judgment:
    """Verdict for one raw attempt."""

    outcome: Outcome
    valid_for_learning: bool
    is_correct: bool | None             # None when excluded (operation/system/ambiguous)
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class CountedOutcome:
    """The single concept-level result that a question presentation contributes."""

    concept_id: str
    question_id: str
    difficulty: float
    answer_options_count: int
    is_correct: bool
    answered_at: datetime
    outcome: Outcome
    reasons: list[str] = field(default_factory=list)


@dataclass
class MasteryResult:
    concept_id: str
    mastery: float                      # 0..1, guessing-corrected
    valid_attempts: int
    correct: int
    guess_baseline: float
    diagnostic_needed: bool             # too few attempts to trust


@dataclass
class WeakConcept:
    concept_id: str
    mastery_score: float
    weakness_score: float
    reason: str


@dataclass
class Recommendation:
    question_id: str
    concept_id: str
    target_band: str                    # easy | medium | hard
    reason: str
    mastery_before: float


@dataclass
class DiagnoseResult:
    student_id: str
    mastery: dict[str, MasteryResult]
    weak_concepts: list[WeakConcept]
    diagnostic_needed: list[str]        # concept_ids with too little data
    recommendations: list[Recommendation]
