"""Data-access layer.

All MySQL reads/writes go through these repositories so the API and training
code never build SQL inline. Writes that must be atomic (an attempt plus its
pointer events) are done inside a single transaction by the caller's session.
"""

from __future__ import annotations

import hashlib
import secrets
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
        return self.session.get(BehaviorAttempt, attempt_id)

    def exists(self, attempt_id: str) -> bool:
        return self.get_attempt(attempt_id) is not None

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

    def learning_exists(self, attempt_id: str) -> bool:
        return self.session.get(LearningAttempt, attempt_id) is not None

    def save_learning_attempt(self, fields: dict[str, Any]) -> None:
        """Insert one learning_attempts row (answer-semantics + judgment)."""
        self.session.add(LearningAttempt(created_at=_utcnow(), **fields))

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
        existing = self.session.get(ShadowOutcome, attempt_id)
        if existing is not None:
            return existing, False

        prediction = self.session.scalar(
            select(ModelPrediction)
            .where(ModelPrediction.attempt_id == attempt_id)
            .order_by(ModelPrediction.predicted_at.desc(), ModelPrediction.prediction_id.desc())
            .limit(1)
        )
        if prediction is None:
            raise LookupError("prediction_not_found")
        if prediction.policy_mode != "shadow":
            raise ValueError("prediction_was_not_shadowed")

        outcome = ShadowOutcome(
            attempt_id=attempt_id,
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
