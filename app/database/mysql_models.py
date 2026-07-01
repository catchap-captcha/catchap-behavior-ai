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
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.services.feature_extractor import FEATURE_NAMES


class Base(DeclarativeBase):
    pass


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
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    feature_schema_version: Mapped[str] = mapped_column(String(16))
    predicted_at: Mapped[datetime] = mapped_column(DateTime)
