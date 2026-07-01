"""Data-access layer.

All MySQL reads/writes go through these repositories so the API and training
code never build SQL inline. Writes that must be atomic (an attempt plus its
pointer events) are done inside a single transaction by the caller's session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.database.mysql_models import (
    AttemptFeatures,
    BehaviorAttempt,
    InteractionSummary,
    ModelPrediction,
    PointerEvent,
    SecurityFeatures,
)
from app.services.feature_extractor import FEATURE_NAMES


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AttemptRepository:
    """Reads/writes for attempts, events, summaries and features."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # --- reads ---
    def get_attempt(self, attempt_id: str) -> BehaviorAttempt | None:
        return self.session.get(BehaviorAttempt, attempt_id)

    def exists(self, attempt_id: str) -> bool:
        return self.get_attempt(attempt_id) is not None

    # --- writes ---
    def save_attempt_bundle(
        self,
        *,
        attempt: dict[str, Any],
        events: list[dict[str, Any]],
        interaction: dict[str, Any] | None,
    ) -> BehaviorAttempt:
        """Persist an attempt + its pointer events + interaction summary.

        Must be called inside a transaction (the caller commits). The frontend
        can never set ``label`` here — the API layer fixes label/label_source
        from the authenticated collection context before calling this.
        """
        now = _utcnow()
        row = BehaviorAttempt(
            attempt_id=attempt["attempt_id"],
            challenge_id=attempt["challenge_id"],
            session_id=attempt["session_id"],
            anonymous_participant_id=attempt.get("anonymous_participant_id"),
            schema_version=attempt["schema_version"],
            captcha_width=attempt["captcha_width"],
            captcha_height=attempt["captcha_height"],
            presented_at=attempt.get("presented_at"),
            submitted_at=attempt.get("submitted_at"),
            position_correct=attempt.get("position_correct"),
            interaction_success=attempt.get("interaction_success"),
            final_drop_error=attempt.get("final_drop_error"),
            label=attempt.get("label", "unknown"),
            label_source=attempt.get("label_source"),
            bot_family=attempt.get("bot_family"),
            generator_version=attempt.get("generator_version"),
            age_group=attempt.get("age_group", "unknown"),
            consent_version=attempt.get("consent_version"),
            quality_status=attempt.get("quality_status", "pending"),
            rejection_reason=attempt.get("rejection_reason"),
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        for e in events:
            self.session.add(
                PointerEvent(
                    attempt_id=row.attempt_id,
                    seq=e["seq"],
                    event_type=e["event_type"],
                    t_ms=e["t_ms"],
                    x=e["x"],
                    y=e["y"],
                    x_normalized=e.get("x_normalized"),
                    y_normalized=e.get("y_normalized"),
                    target_role=e.get("target_role"),
                    created_at=now,
                )
            )
        if interaction is not None:
            self.session.add(
                InteractionSummary(
                    attempt_id=row.attempt_id,
                    regrab_count=interaction.get("regrab_count", 0),
                    retry_count=interaction.get("retry_count", 0),
                    pointercancel_count=interaction.get("pointercancel_count", 0),
                    empty_click_count=interaction.get("empty_click_count", 0),
                    failed_drop_count=interaction.get("failed_drop_count", 0),
                )
            )
        return row

    def save_features(
        self, attempt_id: str, features: dict[str, float], feature_schema_version: str
    ) -> None:
        """Upsert the 29-feature row for an attempt."""
        existing = self.session.get(AttemptFeatures, attempt_id)
        payload = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
        if existing is None:
            row = AttemptFeatures(
                attempt_id=attempt_id,
                feature_schema_version=feature_schema_version,
                calculated_at=_utcnow(),
                **payload,
            )
            self.session.add(row)
        else:
            for name, value in payload.items():
                setattr(existing, name, value)
            existing.feature_schema_version = feature_schema_version
            existing.calculated_at = _utcnow()

    def save_security_features(self, attempt_id: str, feats: dict[str, Any]) -> None:
        existing = self.session.get(SecurityFeatures, attempt_id)
        if existing is None:
            self.session.add(
                SecurityFeatures(attempt_id=attempt_id, calculated_at=_utcnow(), **feats)
            )
        else:
            for k, v in feats.items():
                setattr(existing, k, v)
            existing.calculated_at = _utcnow()


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save_prediction(self, **kwargs: Any) -> ModelPrediction:
        row = ModelPrediction(predicted_at=_utcnow(), **kwargs)
        self.session.add(row)
        return row


class TrainingDatasetRepository:
    """Read-only access to the ``ai_training_dataset`` view for training."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def fetch_all(self) -> list[dict[str, Any]]:
        """Return every valid, labelled row from the training view as dicts."""
        result = self.session.execute(text("SELECT * FROM ai_training_dataset"))
        return [dict(r) for r in result.mappings().all()]

    def view_exists(self) -> bool:
        try:
            self.session.execute(text("SELECT 1 FROM ai_training_dataset LIMIT 1"))
            return True
        except Exception:
            return False
