"""Data-access layer.

All MySQL reads/writes go through these repositories so the API and training
code never build SQL inline. Writes that must be atomic (an attempt plus its
pointer events) are done inside a single transaction by the caller's session.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, func, select, text, update
from sqlalchemy.orm import Session, selectinload

from app.database.mysql_models import (
    AttemptFeatures,
    BehaviorAttempt,
    CaptchaChallenge,
    InteractionSummary,
    LearningAttempt,
    ModelPrediction,
    PointerEvent,
    SecurityFeatures,
    ShadowOutcome,
)
from app.services.feature_extractor import FEATURE_NAMES
from app.services.replay_detector import HistoricalAttempt, path_from_events, trace_fingerprint


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# --- deployed-schema adapters -------------------------------------------------
# The deployed tables use a CHAR(36) surrogate key and a stricter vocabulary than
# the service does. Everything that translates between the two lives here so the
# API layer keeps speaking in its own terms. See mysql_models for the mapping.

_ATTEMPT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# quality_validator emits valid / pending / rejected (QUALITY_* constants).
# The deployed CHECK allows pending / valid / invalid / review, so only
# "rejected" actually needs renaming. An earlier version of this map keyed on
# "accepted", which does not exist upstream — every genuinely valid attempt
# fell through to "review" and the CHECK accepted it silently, so nothing
# selecting quality_status='valid' would have found a single row.
_QUALITY_STATUS = {"valid": "valid", "rejected": "invalid", "pending": "pending"}


def attempt_uuid(attempt_id: str) -> str:
    """Map a caller's attempt id onto the deployed CHAR(36) primary key.

    The CAPTCHA sends ``ms-{challenge_id}-a{n}`` (42 chars), which does not fit
    ``id CHAR(36)``. UUIDv5 keeps the mapping deterministic, so retries and the
    idempotency check in ``exists()`` still line up. The original string is kept
    in ``metadata.source_attempt_id``.
    """
    try:
        return str(uuid.UUID(attempt_id))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(_ATTEMPT_NAMESPACE, attempt_id))


def _row_id() -> str:
    return str(uuid.uuid4())


def _clamp01(value: float | None) -> float | None:
    """Deployed CHECKs reject anything outside [0, 1] on normalized columns."""
    if value is None:
        return None
    return min(1.0, max(0.0, float(value)))


def _quality_status(status: str | None) -> str:
    return _QUALITY_STATUS.get(status or "pending", "review")


def _label(label: str | None) -> str | None:
    """Deployed CHECK allows NULL / 'human' / 'bot'. Inference traffic is NULL."""
    return label if label in ("human", "bot") else None


def _predicted_label(risk_level: str) -> str:
    """Deployed CHECK allows human / bot / uncertain."""
    return {"low": "human", "high": "bot"}.get(risk_level, "uncertain")


@dataclass(frozen=True)
class IssuedChallenge:
    challenge_id: str
    nonce: str
    expires_at: datetime


@dataclass(frozen=True)
class ChallengeConsumeResult:
    accepted: bool
    reason: str
    verdict: str | None = None


class AttemptRepository:
    """Reads/writes for attempts, events, summaries and features."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # --- reads ---
    def get_attempt(self, attempt_id: str) -> BehaviorAttempt | None:
        return self.session.get(BehaviorAttempt, attempt_uuid(attempt_id))

    def exists(self, attempt_id: str) -> bool:
        return self.get_attempt(attempt_id) is not None

    def _next_attempt_number(self, challenge_id: str) -> int:
        """Deployed UNIQUE KEY is (challenge_id, attempt_number), so retries of the
        same challenge must not both claim 1.
        """
        highest = self.session.scalar(
            select(func.max(BehaviorAttempt.attempt_number)).where(
                BehaviorAttempt.challenge_id == challenge_id
            )
        )
        return int(highest or 0) + 1

    def recent_session_history(
        self,
        *,
        session_id: str,
        now: datetime,
        window_seconds: int,
        limit: int,
    ) -> list[HistoricalAttempt]:
        """Return recent, already-stored paths for same-session risk signals.

        This is deliberately session scoped. It gives the risk assessor useful
        replay and burst evidence without treating a similar path from a
        different student's long-term history as an automatic match.
        """
        if window_seconds <= 0 or limit <= 0:
            return []
        cutoff = now - timedelta(seconds=window_seconds)
        rows = self.session.scalars(
            select(BehaviorAttempt)
            .options(selectinload(BehaviorAttempt.events))
            .where(
                BehaviorAttempt.session_id == session_id,
                BehaviorAttempt.created_at >= cutoff,
                BehaviorAttempt.created_at <= now,
            )
            .order_by(BehaviorAttempt.created_at.desc())
            .limit(limit)
        ).all()

        history: list[HistoricalAttempt] = []
        for row in rows:
            events = [
                {
                    "seq": event.seq,
                    "x": event.x,
                    "y": event.y,
                    "x_normalized": event.x_normalized,
                    "y_normalized": event.y_normalized,
                }
                for event in row.events
            ]
            path = path_from_events(events)
            if path.shape[0] < 2:
                continue
            times = [event.t_ms for event in row.events]
            duration_ms = float(max(times) - min(times)) if times else 0.0
            created_at = row.submitted_at or row.created_at
            created_epoch = created_at.replace(tzinfo=timezone.utc).timestamp()
            history.append(
                HistoricalAttempt(
                    path=path,
                    duration_ms=duration_ms,
                    endpoint=(float(path[-1][0]), float(path[-1][1])),
                    created_at_epoch_s=created_epoch,
                    path_fingerprint=trace_fingerprint(path),
                )
            )
        return history

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
        source_id = attempt["attempt_id"]
        row = BehaviorAttempt(
            attempt_id=attempt_uuid(source_id),
            challenge_id=attempt["challenge_id"],
            session_id=attempt["session_id"],
            anonymous_participant_id=attempt.get("anonymous_participant_id"),
            attempt_number=self._next_attempt_number(attempt["challenge_id"]),
            captcha_width=attempt.get("captcha_width"),
            captcha_height=attempt.get("captcha_height"),
            # started_at is NOT NULL in the deployed schema.
            presented_at=attempt.get("presented_at") or now,
            submitted_at=attempt.get("submitted_at"),
            position_correct=attempt.get("position_correct"),
            label=_label(attempt.get("label")),
            label_source=attempt.get("label_source"),
            bot_family=attempt.get("bot_family"),
            consent_version=attempt.get("consent_version"),
            quality_status=_quality_status(attempt.get("quality_status")),
            rejection_reason=attempt.get("rejection_reason"),
            extra_metadata={
                "source_attempt_id": source_id,
                "schema_version": attempt.get("schema_version"),
                "age_group": attempt.get("age_group", "unknown"),
                "generator_version": attempt.get("generator_version"),
                "interaction_success": attempt.get("interaction_success"),
                "final_drop_error": attempt.get("final_drop_error"),
                # 'unknown' is not a legal label in the deployed CHECK.
                "raw_label": attempt.get("label"),
            },
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        # The deployed FKs point at ai_behavior_attempts.id, and the ORM only
        # orders inserts it can see a relationship for. Flush the parent before
        # anything references it. Without this MySQL raises errno 1452 and the
        # whole bundle is rolled back.
        self.session.flush()

        for e in events:
            self.session.add(
                PointerEvent(
                    attempt_id=row.attempt_id,
                    seq=e["seq"],
                    event_type=e["event_type"],
                    t_ms=e["t_ms"],
                    x=e.get("x"),
                    y=e.get("y"),
                    x_normalized=_clamp01(e.get("x_normalized")),
                    y_normalized=_clamp01(e.get("y_normalized")),
                    pointer_type=e.get("pointer_type"),
                    pressure=e.get("pressure"),
                    buttons_mask=e.get("buttons"),
                    is_trusted=e.get("is_trusted"),
                    is_primary=e.get("is_primary"),
                    coalesced_count=e.get("coalesced_count"),
                    event_metadata=(
                        {"target_role": e["target_role"]} if e.get("target_role") else None
                    ),
                    created_at=now,
                )
            )
        if interaction is not None:
            counts = {
                "pointer_move_count": sum(1 for e in events if e["event_type"] == "pointermove"),
                "pointer_down_count": sum(1 for e in events if e["event_type"] == "pointerdown"),
                "pointer_up_count": sum(1 for e in events if e["event_type"] == "pointerup"),
            }
            self.session.add(
                InteractionSummary(
                    summary_id=_row_id(),
                    attempt_id=row.attempt_id,
                    total_event_count=len(events),
                    calculated_at=now,
                    # Dedicated columns since 2026-07-28 (DB request C).
                    **{
                        name: int(interaction.get(name, 0))
                        for name in (
                            "regrab_count",
                            "retry_count",
                            "pointercancel_count",
                            "empty_click_count",
                            "failed_drop_count",
                        )
                    },
                    **counts,
                )
            )
        return row

    # Deployed columns that measure exactly the same quantity as one of ours,
    # only under a different name. Anything not listed here stays JSON-only.
    _NAMED_FEATURES = (
        "event_count", "duration_ms", "total_distance", "displacement",
        "avg_speed", "speed_std", "avg_acceleration", "jerk_mean",
        "direction_changes", "linearity",
    )

    def save_features(
        self, attempt_id: str, features: dict[str, float], feature_schema_version: str
    ) -> None:
        """Upsert one feature row.

        ``extra_features`` is the authoritative copy — the deployed named columns
        are a different feature set from this extractor's, so only exact-synonym
        columns are filled alongside it. Nothing the extractor produced is lost.
        """
        key = attempt_uuid(attempt_id)
        payload = {name: float(value) for name, value in features.items()}
        named = {
            name: (_clamp01(payload.get(name)) if name == "linearity" else payload.get(name))
            for name in self._NAMED_FEATURES
            if name in payload
        }
        existing = self.session.scalar(
            select(AttemptFeatures).where(
                AttemptFeatures.attempt_id == key,
                AttemptFeatures.feature_schema_version == feature_schema_version,
            )
        )
        if existing is None:
            self.session.add(
                AttemptFeatures(
                    features_id=_row_id(),
                    attempt_id=key,
                    feature_schema_version=feature_schema_version,
                    extraction_status="completed",
                    extra_features=payload,
                    calculated_at=_utcnow(),
                    **named,
                )
            )
            return
        for name, value in named.items():
            setattr(existing, name, value)
        existing.extra_features = payload
        existing.extraction_status = "completed"
        existing.calculated_at = _utcnow()

    def learning_exists(self, attempt_id: str) -> bool:
        return self.session.get(LearningAttempt, attempt_id) is not None

    def save_learning_attempt(self, fields: dict[str, Any]) -> None:
        """Insert one learning_attempts row (answer-semantics + judgment)."""
        self.session.add(LearningAttempt(created_at=_utcnow(), **fields))

    def save_security_features(self, attempt_id: str, feats: dict[str, Any]) -> None:
        key = attempt_uuid(attempt_id)
        mapped = {
            "path_similarity_score": _clamp01(feats.get("path_similarity_score")),
            "exact_replay_detected": bool(feats.get("exact_replay_detected", False)),
            "attempts_per_minute": feats.get("attempts_per_minute"),
            "recent_attempt_count": int(feats.get("recent_attempt_count") or 0),
            # No deployed column for these two.
            "security_flags": {
                "repeated_duration_count": feats.get("repeated_duration_count"),
                "repeated_endpoint_count": feats.get("repeated_endpoint_count"),
            },
        }
        existing = self.session.scalar(
            select(SecurityFeatures).where(SecurityFeatures.attempt_id == key)
        )
        if existing is None:
            self.session.add(
                SecurityFeatures(
                    security_id=_row_id(), attempt_id=key, calculated_at=_utcnow(), **mapped
                )
            )
            return
        for name, value in mapped.items():
            setattr(existing, name, value)
        existing.calculated_at = _utcnow()


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save_prediction(
        self,
        *,
        attempt_id: str,
        human_score: float,
        bot_risk_score: float,
        bot_decision: str,
        risk_score: float,
        risk_level: str,
        recommended_action: str,
        policy_mode: str,
        risk_reasons: list[str],
        threshold: float,
        model_name: str,
        model_version: str,
        feature_schema_version: str,
        inference_latency_ms: int | None = None,
    ) -> ModelPrediction:
        row = ModelPrediction(
            prediction_id=_row_id(),
            attempt_id=attempt_uuid(attempt_id),
            model_name=model_name,
            model_version=model_version,
            feature_schema_version=feature_schema_version,
            # Deployed CHECK allows human/bot/uncertain only — the caller's
            # "<level>_risk" string is kept in model_metadata.
            bot_decision=_predicted_label(risk_level),
            human_score=_clamp01(human_score),
            bot_risk_score=_clamp01(bot_risk_score),
            model_score=float(human_score),
            threshold=threshold,
            risk_score=max(0.0, float(risk_score)),
            risk_level=risk_level,
            recommended_action=recommended_action,
            risk_reasons=list(risk_reasons),
            inference_latency_ms=inference_latency_ms,
            # policy_mode has no deployed column (20260723 migration not applied).
            model_metadata={"policy_mode": policy_mode, "bot_decision": bot_decision},
            predicted_at=_utcnow(),
        )
        self.session.add(row)
        return row


class ShadowOutcomeRepository:
    """Idempotent shadow-mode outcome recording for trusted CAPTCHA backends."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record(
        self,
        *,
        attempt_id: str,
        main_captcha_verdict: str,
        final_verdict: str,
    ) -> tuple[ShadowOutcome, bool]:
        key = attempt_uuid(attempt_id)
        existing = self.session.get(ShadowOutcome, key)
        if existing is not None:
            return existing, False

        prediction = self.session.scalar(
            select(ModelPrediction)
            .where(ModelPrediction.attempt_id == key)
            .order_by(ModelPrediction.predicted_at.desc())
            .limit(1)
        )
        if prediction is None:
            raise LookupError("prediction_not_found")
        # policy_mode moved into model_metadata — no deployed column for it.
        if (prediction.model_metadata or {}).get("policy_mode") != "shadow":
            raise ValueError("prediction_was_not_shadowed")

        outcome = ShadowOutcome(
            attempt_id=key,
            main_captcha_verdict=main_captcha_verdict,
            final_verdict=final_verdict,
            would_have_action=prediction.recommended_action,
            risk_level=prediction.risk_level,
            model_version=prediction.model_version,
            recorded_at=_utcnow(),
        )
        self.session.add(outcome)
        return outcome, True

    def summary(self) -> dict[str, Any]:
        """Aggregate observed CAPTCHA results without exposing raw attempts."""
        rows = self.session.execute(
            select(
                ShadowOutcome.would_have_action,
                func.count(ShadowOutcome.attempt_id),
                func.coalesce(
                    func.sum(case((ShadowOutcome.main_captcha_verdict == "passed", 1), else_=0)),
                    0,
                ),
            ).group_by(ShadowOutcome.would_have_action)
        ).all()
        actions = []
        total = 0
        would_step_up_count = 0
        for action, attempts, main_passed in rows:
            attempts_int = int(attempts)
            passed_int = int(main_passed)
            total += attempts_int
            if action != "allow":
                would_step_up_count += attempts_int
            actions.append(
                {
                    "would_have_action": action,
                    "attempts": attempts_int,
                    "main_passed": passed_int,
                    "main_failed": attempts_int - passed_int,
                    "main_pass_rate": round(passed_int / attempts_int, 6),
                }
            )
        return {
            "total_outcomes": total,
            "would_step_up_count": would_step_up_count,
            "would_step_up_rate": round(would_step_up_count / total, 6) if total else 0.0,
            "actions": sorted(actions, key=lambda item: item["would_have_action"]),
        }


class ChallengeRepository:
    """One-time challenge issuance and atomic server-side consumption."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def issue(
        self,
        *,
        session_id: str,
        site_key: str,
        purpose: str,
        problem_binding: str,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> IssuedChallenge:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = now or _utcnow()
        nonce = secrets.token_urlsafe(32)
        challenge_id = f"ch_{secrets.token_urlsafe(32)}"
        expires_at = now + timedelta(seconds=ttl_seconds)
        self.session.add(
            CaptchaChallenge(
                challenge_id=challenge_id,
                nonce_hash=_sha256(nonce),
                session_id=session_id,
                site_key=site_key,
                purpose=purpose,
                problem_binding_hash=_sha256(problem_binding),
                status="issued",
                expires_at=expires_at,
                consumed_at=None,
                verdict=None,
                created_at=now,
            )
        )
        return IssuedChallenge(challenge_id=challenge_id, nonce=nonce, expires_at=expires_at)

    def consume(
        self,
        *,
        challenge_id: str,
        nonce: str,
        session_id: str,
        site_key: str,
        purpose: str,
        problem_binding: str,
        verdict: str,
        now: datetime | None = None,
    ) -> ChallengeConsumeResult:
        """Consume exactly once when every trusted binding matches.

        The conditional update is the critical replay defense. Concurrent valid
        requests race on ``status='issued'`` and only one can change it to
        ``consumed``. Both passed and failed verdicts consume the challenge.
        """
        if verdict not in {"passed", "failed"}:
            raise ValueError("verdict must be passed or failed")
        now = now or _utcnow()
        statement = (
            update(CaptchaChallenge)
            .where(
                CaptchaChallenge.challenge_id == challenge_id,
                CaptchaChallenge.nonce_hash == _sha256(nonce),
                CaptchaChallenge.session_id == session_id,
                CaptchaChallenge.site_key == site_key,
                CaptchaChallenge.purpose == purpose,
                CaptchaChallenge.problem_binding_hash == _sha256(problem_binding),
                CaptchaChallenge.status == "issued",
                CaptchaChallenge.expires_at > now,
            )
            .values(status="consumed", consumed_at=now, verdict=verdict)
        )
        if self.session.execute(statement).rowcount == 1:
            return ChallengeConsumeResult(accepted=True, reason="consumed", verdict=verdict)

        challenge = self.session.get(CaptchaChallenge, challenge_id)
        if challenge is None:
            return ChallengeConsumeResult(accepted=False, reason="challenge_not_found")
        if challenge.status == "consumed":
            return ChallengeConsumeResult(accepted=False, reason="challenge_already_consumed")
        if challenge.expires_at <= now:
            return ChallengeConsumeResult(accepted=False, reason="challenge_expired")
        return ChallengeConsumeResult(accepted=False, reason="challenge_binding_invalid")


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
