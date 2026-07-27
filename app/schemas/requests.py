"""Inbound request schemas (Pydantic v2).

The CAPTCHA payload is web-only: no device model, OS, screen orientation or
device fingerprint is accepted. CAPTCHA pixel size and normalized coordinates
are preserved because the behavioral features depend on them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Upper bound on pointer events per attempt. The observed human maximum is
# ~1500; 5000 blocks oversized/DoS payloads (red-team R10) with a wide margin.
MAX_EVENTS_PER_ATTEMPT = 5000

EventType = Literal["pointerdown", "pointermove", "pointerup", "pointercancel"]
Label = Literal["human", "bot", "unknown"]
LabelSource = Literal[
    "controlled_collection",
    "playwright",
    "selenium",
    "rule_bot",
    "gan_bot",
    "replay_bot",
]
AgeGroup = Literal["adult", "child", "unknown"]


class CaptchaInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Timing(BaseModel):
    presented_at: datetime | None = None
    submitted_at: datetime | None = None


class PointerEventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0)
    event_type: EventType
    t_ms: int
    x: float
    y: float
    x_normalized: float | None = None
    y_normalized: float | None = None
    target_role: str | None = None


class InteractionIn(BaseModel):
    regrab_count: int = 0
    retry_count: int = 0
    pointercancel_count: int = 0
    empty_click_count: int = 0
    failed_drop_count: int = 0


class CollectionContext(BaseModel):
    """Trusted labelling context, supplied ONLY by the CAPTCHA backend.

    This block is honoured because /collect is API-key protected. The end-user
    frontend cannot reach /collect and therefore cannot self-assign a label.
    """

    label: Label = "unknown"
    label_source: LabelSource | None = None
    bot_family: str | None = None
    generator_version: str | None = None
    age_group: AgeGroup = "unknown"
    consent_version: str | None = None


class LearningContext(BaseModel):
    """Answer-semantics for the weak-problem recommendation (the 'WHAT').

    Supplied by the trusted CAPTCHA/learning backend (which looks up the correct
    answer, concept and difficulty from the question bank — never trusts the
    frontend for these). Bot detection needs only the pointer events (HOW);
    recommendation needs this block (WHAT). Optional, so bot-only collection
    still works without it.
    """

    question_id: str = Field(min_length=1, max_length=64)
    concept_id: str = Field(min_length=1, max_length=64)
    difficulty: float = Field(ge=0.0, le=1.0)
    answer_options_count: int = Field(ge=0)          # 0/1 => open-ended (no guessing)
    correct_answer_id: str
    answer_slot_id: str = "slot"                     # id of the valid answer drop area
    grabbed_answer_id: str | None = None             # tile the student picked up (= intended answer)
    released_target_id: str | None = None            # where dropped; None => drop failed
    game_type: str | None = None


class _AttemptBase(BaseModel):
    schema_version: str
    attempt_id: str = Field(min_length=1, max_length=64)
    challenge_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    anonymous_participant_id: str | None = Field(default=None, max_length=64)
    captcha: CaptchaInfo
    timing: Timing = Field(default_factory=Timing)
    events: list[PointerEventIn]
    interaction: InteractionIn = Field(default_factory=InteractionIn)

    @field_validator("events")
    @classmethod
    def _events_present(cls, v: list[PointerEventIn]) -> list[PointerEventIn]:
        # Structural floor only; content quality is judged by quality_validator.
        if len(v) == 0:
            raise ValueError("events must not be empty")
        # Upper bound guards the scorer against oversized payloads (red-team
        # R10: 100k events = 18.7MB / 1.7s CPU with no cap). The observed human
        # maximum is ~1500 events, so 5000 leaves a wide margin and never
        # rejects a real drag.
        if len(v) > MAX_EVENTS_PER_ATTEMPT:
            raise ValueError(
                f"events exceed {MAX_EVENTS_PER_ATTEMPT} (got {len(v)})"
            )
        return v


class CollectRequest(_AttemptBase):
    """Controlled-collection payload. May carry a trusted labelling context.

    Optional pass/fail signals (position_correct, interaction_success,
    final_drop_error) are stored but never used as model features.
    """

    collection: CollectionContext = Field(default_factory=CollectionContext)
    position_correct: bool | None = None
    interaction_success: bool | None = None
    final_drop_error: float | None = None
    # answer-semantics for weak-problem recommendation; None => bot-only collection
    learning: LearningContext | None = None


class PredictRequest(_AttemptBase):
    """Production inference payload. Deliberately has NO label fields."""


class ShadowOutcomeRequest(BaseModel):
    """Final CAPTCHA result recorded by the trusted backend during shadow mode.

    The AI does not receive an answer or any answer text. Shadow mode must not
    change the user's main CAPTCHA result, so the two verdicts must agree.
    """

    attempt_id: str = Field(min_length=1, max_length=64)
    main_captcha_verdict: Literal["passed", "failed"]
    final_verdict: Literal["passed", "failed"]

    @field_validator("final_verdict")
    @classmethod
    def _shadow_preserves_main_verdict(cls, value: str, info) -> str:
        main_verdict = info.data.get("main_captcha_verdict")
        if main_verdict is not None and value != main_verdict:
            raise ValueError("shadow mode must preserve the main CAPTCHA verdict")
        return value


class ChallengeIssueRequest(BaseModel):
    """Trusted CAPTCHA-backend request to create a one-time challenge."""

    session_id: str = Field(min_length=1, max_length=64)
    site_key: str = Field(min_length=1, max_length=128)
    purpose: str = Field(min_length=1, max_length=64)
    problem_binding: str = Field(min_length=1, max_length=512)
    ttl_seconds: int | None = Field(default=None, ge=30, le=600)


class ChallengeConsumeRequest(BaseModel):
    """Trusted CAPTCHA-backend request to consume one issued challenge."""

    challenge_id: str = Field(min_length=1, max_length=64)
    nonce: str = Field(min_length=32, max_length=128)
    session_id: str = Field(min_length=1, max_length=64)
    site_key: str = Field(min_length=1, max_length=128)
    purpose: str = Field(min_length=1, max_length=64)
    problem_binding: str = Field(min_length=1, max_length=512)
    verdict: Literal["passed", "failed"]
