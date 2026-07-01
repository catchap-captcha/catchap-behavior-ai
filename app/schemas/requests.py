"""Inbound request schemas (Pydantic v2).

The CAPTCHA payload is web-only: no device model, OS, screen orientation or
device fingerprint is accepted. CAPTCHA pixel size and normalized coordinates
are preserved because the behavioral features depend on them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class PredictRequest(_AttemptBase):
    """Production inference payload. Deliberately has NO label fields."""
