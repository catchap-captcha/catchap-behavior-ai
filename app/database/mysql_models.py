"""SQLAlchemy 2.x ORM models mapping the DB team's MySQL tables.

These declarations MUST stay in lock-step with the **deployed** schema in
``catchap_dev_db``, a snapshot of which lives in
``local-test-data/prod_ai_schema.sql``. The application never issues DDL from
these models; they exist only so the ORM can read/write existing tables.

Why the attribute names differ from the column names
----------------------------------------------------
The DB team built a richer schema than ``db/schema_mysql.sql`` (the DDL this
repo originally handed over), and the two drifted apart: different primary
keys, different column names, extra CHECK constraints. Rather than rename the
attributes everywhere in the service, each model maps its existing attribute
name onto the deployed column name, e.g. ``mapped_column("participant_id", ...)``.
Read the second positional argument as "the real column".

Values the deployed schema has no column for are packed into that table's JSON
column (``metadata`` / ``extra_features`` / ``security_flags`` /
``model_metadata`` / ``event_metadata``) by ``repositories.py``. Nothing is
dropped silently.

``db/schema_mysql.sql`` is NO LONGER the source of truth. See
``docs/DB_REQUEST_20260728.md``.
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


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Inference path — these six are mapped to the deployed schema.
# ---------------------------------------------------------------------------


class BehaviorAttempt(Base):
    """One scored attempt.

    ``attempt_id`` is a UUIDv5 derived from the CAPTCHA's own attempt id (see
    ``repositories.attempt_uuid``); the deployed PK is ``id CHAR(36)`` and the
    CAPTCHA's 42-character ``ms-{challenge_id}-a{n}`` string does not fit. The
    original string is preserved in ``extra_metadata["source_attempt_id"]``.
    """

    __tablename__ = "ai_behavior_attempts"
    __table_args__ = (
        UniqueConstraint("challenge_id", "attempt_number", name="uq_challenge_attempt"),
    )

    attempt_id: Mapped[str] = mapped_column("id", String(36), primary_key=True)
    challenge_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    anonymous_participant_id: Mapped[str | None] = mapped_column(
        "participant_id", String(128), nullable=True, index=True
    )
    # Part of the deployed UNIQUE KEY. Assigned by the repository, not the caller.
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    # The deployed CHECK allows NULL / 'human' / 'bot' only — never 'unknown'.
    label: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    label_source: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    bot_family: Mapped[str | None] = mapped_column("bot_type", String(32), nullable=True)

    # The deployed CHECK allows pending/valid/invalid/review/corrupted.
    quality_status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    rejection_reason: Mapped[str | None] = mapped_column(
        "quality_reason", String(255), nullable=True
    )
    position_correct: Mapped[bool | None] = mapped_column("is_correct", Boolean, nullable=True)

    # The CAPTCHA canvas is the viewport as far as the deployed schema is concerned.
    captcha_width: Mapped[int | None] = mapped_column("viewport_width", Integer, nullable=True)
    captcha_height: Mapped[int | None] = mapped_column("viewport_height", Integer, nullable=True)

    presented_at: Mapped[datetime] = mapped_column("started_at", DateTime)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    consent_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # schema_version, age_group, generator_version, interaction_success,
    # final_drop_error and source_attempt_id live here — no deployed columns.
    extra_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    events: Mapped[list["PointerEvent"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )
    interaction: Mapped["InteractionSummary | None"] = relationship(
        back_populates="attempt", cascade="all, delete-orphan", uselist=False
    )
    features: Mapped["AttemptFeatures | None"] = relationship(
        back_populates="attempt", cascade="all, delete-orphan", uselist=False
    )
    security: Mapped["SecurityFeatures | None"] = relationship(
        back_populates="attempt", cascade="all, delete-orphan", uselist=False
    )
    predictions: Mapped[list["ModelPrediction"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )


class PointerEvent(Base):
    __tablename__ = "ai_pointer_events"
    __table_args__ = (UniqueConstraint("attempt_id", "seq", name="uq_attempt_seq"),)

    event_id: Mapped[int] = mapped_column(
        "id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_behavior_attempts.id"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(32))
    t_ms: Mapped[int] = mapped_column(
        "client_timestamp_ms", BigInteger().with_variant(Integer, "sqlite")
    )
    x: Mapped[float | None] = mapped_column("x_pixel", Float, nullable=True)
    y: Mapped[float | None] = mapped_column("y_pixel", Float, nullable=True)
    # Deployed CHECK constrains both to [0, 1]; the repository clamps.
    x_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    object_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # target_role has no deployed column.
    event_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    attempt: Mapped[BehaviorAttempt] = relationship(back_populates="events")


class InteractionSummary(Base):
    """The AI's own five counters got dedicated columns on 2026-07-28.

    The deployed table's existing counters (``wrong_click_count``,
    ``wrong_drag_count``, ``object_revisit_count``) measure adjacent but
    different things, so the DB team added these five rather than have us guess
    a mapping — see ``docs/DB_REQUEST_20260728.md`` request C.
    """

    __tablename__ = "ai_interaction_summaries"

    summary_id: Mapped[str] = mapped_column("id", String(36), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_behavior_attempts.id"), unique=True, index=True
    )
    total_event_count: Mapped[int] = mapped_column(Integer, default=0)
    pointer_move_count: Mapped[int] = mapped_column(Integer, default=0)
    pointer_down_count: Mapped[int] = mapped_column(Integer, default=0)
    pointer_up_count: Mapped[int] = mapped_column(Integer, default=0)

    regrab_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    pointercancel_count: Mapped[int] = mapped_column(Integer, default=0)
    empty_click_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_drop_count: Mapped[int] = mapped_column(Integer, default=0)

    extra_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    calculated_at: Mapped[datetime] = mapped_column(DateTime)

    attempt: Mapped[BehaviorAttempt] = relationship(back_populates="interaction")


class AttemptFeatures(Base):
    """Model features.

    The authoritative copy of every extracted feature is the ``extra_features``
    JSON blob, keyed by the extractor's own names. The named columns below are
    populated as well, but only where the deployed column measures exactly the
    same quantity under a different name — no approximations.
    """

    __tablename__ = "ai_attempt_features"
    __table_args__ = (
        UniqueConstraint("attempt_id", "feature_version", name="uq_attempt_feature_version"),
    )

    features_id: Mapped[str] = mapped_column("id", String(36), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_behavior_attempts.id"), index=True
    )
    feature_schema_version: Mapped[str | None] = mapped_column(
        "feature_version", String(32), nullable=True
    )
    # Deployed CHECK allows pending/completed/failed.
    extraction_status: Mapped[str] = mapped_column(String(16), default="completed")
    extraction_error: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # 1:1 renames of the same quantity.
    event_count: Mapped[float | None] = mapped_column("pointer_event_count", Float, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column("total_duration_ms", Float, nullable=True)
    total_distance: Mapped[float | None] = mapped_column("total_path_length", Float, nullable=True)
    displacement: Mapped[float | None] = mapped_column(
        "straight_line_distance", Float, nullable=True
    )
    avg_speed: Mapped[float | None] = mapped_column("average_speed", Float, nullable=True)
    speed_std: Mapped[float | None] = mapped_column("speed_stddev", Float, nullable=True)
    avg_acceleration: Mapped[float | None] = mapped_column(
        "average_acceleration", Float, nullable=True
    )
    jerk_mean: Mapped[float | None] = mapped_column("average_jerk", Float, nullable=True)
    direction_changes: Mapped[float | None] = mapped_column(
        "direction_change_count", Float, nullable=True
    )
    # Deployed CHECK constrains this to [0, 1]; the repository clamps.
    linearity: Mapped[float | None] = mapped_column("straightness_ratio", Float, nullable=True)

    extra_features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    calculated_at: Mapped[datetime] = mapped_column("extracted_at", DateTime)

    attempt: Mapped[BehaviorAttempt] = relationship(back_populates="features")


class SecurityFeatures(Base):
    __tablename__ = "ai_security_features"

    security_id: Mapped[str] = mapped_column("id", String(36), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_behavior_attempts.id"), unique=True, index=True
    )
    # Deployed CHECK constrains this to [0, 1]; the repository clamps.
    path_similarity_score: Mapped[float | None] = mapped_column(
        "replay_similarity_score", Float, nullable=True
    )
    exact_replay_detected: Mapped[bool] = mapped_column(
        "exact_sequence_match", Boolean, default=False
    )
    attempts_per_minute: Mapped[float | None] = mapped_column(
        "session_frequency_score", Float, nullable=True
    )
    recent_attempt_count: Mapped[int] = mapped_column(
        "recent_session_attempt_count", Integer, default=0
    )
    # repeated_duration_count / repeated_endpoint_count have no deployed column.
    security_flags: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    calculated_at: Mapped[datetime] = mapped_column(DateTime)

    attempt: Mapped[BehaviorAttempt] = relationship(back_populates="security")


class ModelPrediction(Base):
    __tablename__ = "ai_model_predictions"

    prediction_id: Mapped[str] = mapped_column("id", String(36), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_behavior_attempts.id"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    feature_schema_version: Mapped[str | None] = mapped_column(
        "feature_version", String(32), nullable=True
    )
    # Deployed CHECK allows human/bot/uncertain — not the "<level>_risk" string
    # the service used to store. The repository derives it from risk_level.
    bot_decision: Mapped[str] = mapped_column("predicted_label", String(16))
    # Deployed CHECKs constrain both probabilities to [0, 1].
    human_score: Mapped[float | None] = mapped_column("human_probability", Float, nullable=True)
    bot_risk_score: Mapped[float | None] = mapped_column("bot_probability", Float, nullable=True)
    model_score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column("decision_threshold", Float, nullable=True)
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    recommended_action: Mapped[str] = mapped_column(String(32))
    risk_reasons: Mapped[list] = mapped_column(JSON, default=list)
    inference_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # policy_mode has no deployed column (the 20260723 migration was never applied).
    model_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime)

    attempt: Mapped[BehaviorAttempt] = relationship(back_populates="predictions")


# ---------------------------------------------------------------------------
# NOT DEPLOYED — do not use against catchap_dev_db until the DB team acts.
# Kept so the code that references them still imports; every write will fail.
# See docs/DB_REQUEST_20260728.md.
# ---------------------------------------------------------------------------


class CaptchaChallenge(Base):
    """⚠️ Shape mismatch. ``ai_captcha_challenges`` exists but is a different
    design (question_id, client_ip_hash, status enum, attempt_count/max_attempts,
    expires_at > issued_at CHECK). The AI's issue/consume protocol is not wired
    to the CAPTCHA today, so this mapping is left as-is pending a decision.
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


class ShadowOutcome(Base):
    """⚠️ Table does not exist in catchap_dev_db. ``/api/v1/behavior/shadow/outcomes``
    returns 500 until the DB team applies the DDL in the request doc.
    """

    __tablename__ = "ai_shadow_outcomes"

    attempt_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    main_captcha_verdict: Mapped[str] = mapped_column(String(16))
    final_verdict: Mapped[str] = mapped_column(String(16))
    would_have_action: Mapped[str] = mapped_column(String(32), index=True)
    risk_level: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64))
    recorded_at: Mapped[datetime] = mapped_column(DateTime)


class LearningAttempt(Base):
    """⚠️ Table does not exist in catchap_dev_db (see db/schema_learning_mysql.sql)."""

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

    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)
    valid_for_learning: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    captcha_attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    answered_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)
