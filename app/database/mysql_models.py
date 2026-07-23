"""SQLAlchemy 2.x ORM models mapping the DB team's MySQL tables.

These declarations MUST stay in lock-step with ``db/schema_mysql.sql`` (the DDL
handed to the DB team). The application never issues DDL from these models; they
exist only so the ORM can read/write existing tables.

The 29 behavioral feature columns on :class:`AttemptFeatures` are generated from
``feature_extractor.FEATURE_NAMES`` so the table and the extractor cannot drift.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.services.feature_extractor import FEATURE_NAMES


class Base(DeclarativeBase):
    pass


class CaptchaChallenge(Base):
    """Server-side, single-use CAPTCHA challenge state.

    The nonce and problem binding are stored only as SHA-256 digests.  This
    table is deliberately independent of behavior attempts because a challenge
    must be consumed even when a user submits no pointer trace or fails it.
    """

    __tablename__ = "ai_captcha_challenges"

    challenge_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    nonce_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    site_key: Mapped[str] = mapped_column(String(128), nullable=False)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    problem_binding_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="issued", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BehaviorAttempt(Base):
    __tablename__ = "ai_behavior_attempts"

    attempt_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # challenge_id / session_id kept as strings until the DB team confirms the
    # other teams' table names + types and wires real FKs (see docs/DB_REQUEST.md).
    challenge_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    anonymous_participant_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    schema_version: Mapped[str] = mapped_column(String(16))

    captcha_width: Mapped[int] = mapped_column(Integer)
    captcha_height: Mapped[int] = mapped_column(Integer)
    presented_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    position_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    interaction_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    final_drop_error: Mapped[float | None] = mapped_column(Float, nullable=True)

    label: Mapped[str] = mapped_column(String(16), default="unknown", index=True)
    label_source: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    bot_family: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generator_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    age_group: Mapped[str] = mapped_column(String(16), default="unknown")
    consent_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    quality_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    events: Mapped[list["PointerEvent"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )


class PointerEvent(Base):
    __tablename__ = "ai_pointer_events"
    __table_args__ = (UniqueConstraint("attempt_id", "seq", name="uq_attempt_seq"),)

    event_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    attempt_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_behavior_attempts.attempt_id"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(16))
    t_ms: Mapped[int] = mapped_column(Integer)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    x_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    attempt: Mapped[BehaviorAttempt] = relationship(back_populates="events")


class InteractionSummary(Base):
    __tablename__ = "ai_interaction_summaries"

    attempt_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_behavior_attempts.attempt_id"), primary_key=True
    )
    regrab_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    pointercancel_count: Mapped[int] = mapped_column(Integer, default=0)
    empty_click_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_drop_count: Mapped[int] = mapped_column(Integer, default=0)


def _feature_columns() -> dict[str, Mapped[float]]:
    """Build one Float column per behavioral feature, generated from the spec."""
    return {name: mapped_column(Float, nullable=True) for name in FEATURE_NAMES}


# Generate AttemptFeatures dynamically so its columns == FEATURE_NAMES exactly.
AttemptFeatures = type(
    "AttemptFeatures",
    (Base,),
    {
        "__tablename__": "ai_attempt_features",
        "__annotations__": {name: "Mapped[float]" for name in FEATURE_NAMES},
        "attempt_id": mapped_column(
            String(64), ForeignKey("ai_behavior_attempts.attempt_id"), primary_key=True
        ),
        "feature_schema_version": mapped_column(String(16)),
        "calculated_at": mapped_column(DateTime),
        **_feature_columns(),
    },
)


class SecurityFeatures(Base):
    __tablename__ = "ai_security_features"

    attempt_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_behavior_attempts.attempt_id"), primary_key=True
    )
    path_similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    exact_replay_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    repeated_duration_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempts_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    recent_attempt_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    repeated_endpoint_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calculated_at: Mapped[datetime] = mapped_column(DateTime)


class ModelPrediction(Base):
    __tablename__ = "ai_model_predictions"

    prediction_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    attempt_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_behavior_attempts.attempt_id"), index=True
    )
    human_score: Mapped[float] = mapped_column(Float)
    bot_risk_score: Mapped[float] = mapped_column(Float)
    bot_decision: Mapped[str] = mapped_column(String(16))
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    recommended_action: Mapped[str] = mapped_column(String(32))
    policy_mode: Mapped[str] = mapped_column(String(16), default="shadow", index=True)
    risk_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    feature_schema_version: Mapped[str] = mapped_column(String(16))
    predicted_at: Mapped[datetime] = mapped_column(DateTime)


class ShadowOutcome(Base):
    """Observed final result for a prediction while enforcement is shadowed.

    No answer text, question content, or client-controlled risk action is
    stored here. The would-have action is copied from the saved AI prediction.
    """

    __tablename__ = "ai_shadow_outcomes"

    attempt_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_behavior_attempts.attempt_id"), primary_key=True
    )
    main_captcha_verdict: Mapped[str] = mapped_column(String(16))
    final_verdict: Mapped[str] = mapped_column(String(16))
    would_have_action: Mapped[str] = mapped_column(String(32), index=True)
    risk_level: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64))
    recorded_at: Mapped[datetime] = mapped_column(DateTime)


class LearningAttempt(Base):
    """Maps ``learning_attempts`` (see db/schema_learning_mysql.sql).

    FK columns are plain strings here (no ORM ForeignKey) — the real FKs live in
    the DDL; the app only reads/writes. Stores the answer-semantics (WHAT) plus
    the operation-error judgment computed at collect time.
    """

    __tablename__ = "learning_attempts"

    attempt_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(String(64), index=True)
    question_id: Mapped[str] = mapped_column(String(64), index=True)
    concept_id: Mapped[str] = mapped_column(String(64), index=True)
    difficulty: Mapped[float] = mapped_column(Float)
    answer_options_count: Mapped[int] = mapped_column(Integer)
    correct_answer_id: Mapped[str] = mapped_column(String(64))

    grabbed_answer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    released_target_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    answer_slot_id: Mapped[str] = mapped_column(String(64), default="slot")

    pointercancel_count: Mapped[int] = mapped_column(Integer, default=0)
    regrab_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_drop_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    final_drop_error_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    system_error: Mapped[bool] = mapped_column(Boolean, default=False)

    presentation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # operation-error judgment (computed from learning.operation_error)
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)
    valid_for_learning: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    captcha_attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    answered_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)
