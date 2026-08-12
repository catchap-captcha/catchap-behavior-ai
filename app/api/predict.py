"""POST /api/v1/behavior/predict — advisory production risk assessment.

Validates the raw events, computes features, and scores with the loaded
production model. If no model is loaded it returns HTTP 503 with
``model_not_ready`` — never a fabricated score. The trusted CAPTCHA backend
receives an advisory risk level and recommended follow-up action; it remains
the sole owner of allow/block and challenge-token decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.challenge import require_captcha_backend_key
from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import AttemptRepository, PredictionRepository
from app.schemas.requests import PredictRequest
from app.schemas.responses import ModelNotReadyResponse, PredictResponse
from app.services.feature_profiles import get_feature_profile
from app.services.model_service import model_service
from app.services.scoring_audit import build_audit
from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from app.services.aim_segment import trim_aim, without_aim
from app.services.replay_detector import compute_replay_features
from app.services.risk_fusion import RiskFusionPolicy, fuse_behavior_risk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


def _to_naive_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _duration_ms(events: list[dict]) -> float:
    timestamps = [event.get("t_ms") for event in events if event.get("t_ms") is not None]
    return float(max(timestamps) - min(timestamps)) if timestamps else 0.0


@router.post(
    "/api/v1/behavior/predict",
    dependencies=[Depends(require_captcha_backend_key)],
)
def predict(payload: PredictRequest, session: Session = Depends(get_session)):
    # No model -> 503, never a fake score.
    if not model_service.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ModelNotReadyResponse().model_dump(),
        )

    # 조준(`aimmove`)을 누가 보는지는 번들이 정한다.
    #
    # 재생 탐지기는 항상 본다 — 길이가 곧 성능이라, 드래그만 13점이면 변형 재생을
    # 9.3% 잡지만 조준을 이으면 27점이 되어 95.7% 를 잡는다(2026-08-12, 실사용
    # 1,323개). `_to_path` 는 좌표만 있으면 유형을 안 가리므로 그대로 넘기면 된다.
    #
    # 분류기는 `uses_aim` 인 번들만 본다. 세션 특징 추출기가 이벤트 유형을 안 가려서
    # (`feature_extractor_v23.extract_features` 가 받은 것을 전부 쓴다), 조준 없이
    # 학습된 모델에 넘기면 학습 때 본 적 없는 분포가 되고 `scoring_unit="session"`
    # 이라 그게 그대로 판정이 된다.
    #
    # 조준을 보는 번들에는 **학습과 같은 규칙으로 잘라서** 넘긴다. 자르지 않으면
    # 문제를 읽는 동안 커서가 돌아다닌 것까지 궤적이 되어, 미지 계열 통과율이
    # 7.1% 에서 8.4% 로 올라간다.
    all_events = [e.model_dump() for e in payload.events]
    #
    # 걸러낸 쪽은 **복사본에 `seq` 를 다시 매긴다.** 품질 검사가 seq 가 0부터
    # 빈틈없이 이어질 것을 요구하기 때문이다(`quality_validator`: seq_not_sequential
    # → invalid_event_telemetry → 위험도 상승 → step_up). 조준을 빼면 그 자리에
    # 구멍이 남는다. 0812 에 실제로 이렇게 나갔다 — shadow 라 무해했지만 active 였다면
    # 조준이 있는 사용자가 전부 캡차를 한 번 더 받았을 것이다.
    #
    # 복사본이어야 하는 이유: 자른 결과는 같은 dict 객체를 가리키므로 제자리에서
    # 손대면 저장·재생 경로가 쓰는 `all_events` 의 seq 까지 바뀐다.
    kept = trim_aim(all_events) if model_service.uses_aim else without_aim(all_events)
    events = [{**event, "seq": index} for index, event in enumerate(kept)]

    # Validate raw events (result is advisory here; we still score, but a broken
    # payload is recorded with its quality status when persisted).
    quality = validate_attempt(
        events,
        captcha_width=payload.captcha.width,
        captcha_height=payload.captcha.height,
        presented_at=_to_naive_utc(payload.timing.presented_at),
        submitted_at=_to_naive_utc(payload.timing.submitted_at),
        enforce_server_time_window=True,
    )

    profile = get_feature_profile(
        model_service.feature_schema_version,
        trajectory_only=model_service.feature_input_scope == "pointer_trajectory_only",
    )
    features = profile.extractor(events, payload.interaction.model_dump())
    result = model_service.score(features)
    session_score = result["human_score"]

    # Score per drag as well, always. The two numbers disagree in exactly the
    # cases that matter — a straight path repeated five times scores 1.0000 as a
    # session and ~0.006 per drag — so recording both is what lets us flip the
    # unit later on evidence instead of on argument.
    settings = get_settings()
    per_drag = model_service.score_per_drag(
        events,
        profile.extractor,
        payload.interaction.model_dump(),
        settings.per_drag_threshold,
    )
    if settings.scoring_unit == "per_drag" and per_drag is not None:
        # Keep the session score in the record; only the decision moves.
        result = {**result, "session_human_score": result["human_score"],
                  "human_score": per_drag["human_score"],
                  "bot_risk_score": round(1.0 - per_drag["human_score"], 6),
                  "threshold": per_drag["threshold"],
                  "bot_decision": "low_risk" if per_drag["prediction"] == "human" else "high_risk",
                  "prediction": per_drag["prediction"]}

    # Compare only with recent attempts from this trusted backend session. A
    # browser never calls this route, so session_id is backend-derived rather
    # than an untrusted client claim.
    now = _utcnow()
    attempts = AttemptRepository(session)
    try:
        history = attempts.recent_session_history(
            session_id=payload.session_id,
            now=now,
            window_seconds=settings.risk_history_window_seconds,
            limit=settings.risk_history_max_attempts,
        )
    except SQLAlchemyError:
        # Shadow mode must remain observable during a DB outage. The model can
        # still score the current trajectory, but replay-history signals are
        # deliberately unavailable until storage recovers.
        session.rollback()
        history = []
    # 이력(`history`)은 저장된 이벤트 전부로 경로를 만든다 — 유형을 안 가린다. 그래서
    # 현재 시도도 조준을 포함한 전체 경로로 비교해야 같은 모양끼리 맞붙는다.
    # 소요시간도 같은 이유로 전체 구간에서 읽는다.
    replay = compute_replay_features(
        all_events,
        duration_ms=_duration_ms(all_events),
        now_epoch_s=now.replace(tzinfo=timezone.utc).timestamp(),
        history=history,
        recent_window_s=float(settings.risk_history_window_seconds),
    )
    assessment = fuse_behavior_risk(
        result["human_score"],
        replay,
        RiskFusionPolicy(
            model_human_threshold=result["threshold"],
            step_up_human_threshold=result.get("step_up_threshold"),
            dtw_similarity_threshold=settings.risk_dtw_similarity_threshold,
            max_attempts_per_minute=settings.risk_max_attempts_per_minute,
        ),
        quality_rejected=quality.status == QUALITY_REJECTED,
    )

    # best-effort persistence of raw attempt + features + prediction
    _persist(
        session,
        payload,
        # 저장도 전체 경로여야 한다 — 다음 시도가 이걸 `recent_session_history` 로
        # 되읽어 현재 경로와 맞붙는다. 조준을 빼고 저장하면 27점짜리 현재 경로가
        # 13점짜리 이력과 비교돼 비슷함이 무너진다.
        all_events,
        features,
        profile.version,
        quality,
        result,
        replay,
        assessment,
        settings.risk_policy_mode,
        build_audit(
            events=events,
            features=features,
            captcha_width=payload.captcha.width,
            captcha_height=payload.captcha.height,
            scoring_unit=settings.scoring_unit,
            session_human_score=float(session_score),
            per_drag=per_drag,
        ),
    )

    return PredictResponse(
        attempt_id=payload.attempt_id,
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        recommended_action=assessment.recommended_action,
        policy_mode=settings.risk_policy_mode,
        reasons=list(assessment.reasons),
        human_score=result["human_score"],
        bot_risk_score=result["bot_risk_score"],
        path_similarity_score=assessment.path_similarity_score,
        exact_replay_detected=assessment.exact_replay_detected,
        attempts_per_minute=assessment.attempts_per_minute,
        threshold=result["threshold"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        feature_schema_version=result["feature_schema_version"],
    )


def _persist(
    session: Session,
    payload,
    events,
    features,
    feature_schema_version: str,
    quality,
    result,
    replay,
    assessment,
    policy_mode: str,
    scoring_detail: dict | None = None,
) -> None:
    """Store the inference attempt. Failure here never breaks the response."""
    try:
        repo = AttemptRepository(session)
        if not repo.exists(payload.attempt_id):
            repo.save_attempt_bundle(
                attempt={
                    "attempt_id": payload.attempt_id,
                    "challenge_id": payload.challenge_id,
                    "session_id": payload.session_id,
                    "anonymous_participant_id": payload.anonymous_participant_id,
                    "schema_version": payload.schema_version,
                    "captcha_width": payload.captcha.width,
                    "captcha_height": payload.captcha.height,
                    "presented_at": _to_naive_utc(payload.timing.presented_at),
                    "submitted_at": _to_naive_utc(payload.timing.submitted_at),
                    # inference traffic is unlabelled
                    "label": "unknown",
                    "label_source": None,
                    "quality_status": quality.status,
                    "rejection_reason": quality.reason,
                },
                events=events,
                interaction=payload.interaction.model_dump(),
            )
            repo.save_features(payload.attempt_id, features, feature_schema_version)
            repo.save_security_features(
                payload.attempt_id,
                {
                    "path_similarity_score": replay.path_similarity_score,
                    "exact_replay_detected": replay.exact_replay_detected,
                    "repeated_duration_count": replay.repeated_duration_count,
                    "attempts_per_minute": replay.attempts_per_minute,
                    "recent_attempt_count": replay.recent_attempt_count,
                    "repeated_endpoint_count": replay.repeated_endpoint_count,
                },
            )
        PredictionRepository(session).save_prediction(
            attempt_id=payload.attempt_id,
            human_score=result["human_score"],
            bot_risk_score=result["bot_risk_score"],
            bot_decision=f"{assessment.risk_level}_risk",
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            recommended_action=assessment.recommended_action,
            policy_mode=policy_mode,
            risk_reasons=list(assessment.reasons),
            threshold=result["threshold"],
            scoring_detail=scoring_detail,
            model_name=result["model_name"],
            model_version=result["model_version"],
            feature_schema_version=result["feature_schema_version"],
        )
        session.commit()
    except Exception:
        # Persistence is best-effort so a storage fault never fails a live
        # CAPTCHA, but it must never be invisible: a silent rollback here once
        # left /predict storing nothing at all while /health still read "ok",
        # which in turn left the replay detector with an empty history.
        session.rollback()
        logger.exception(
            "predict: failed to persist attempt %s — scoring returned, nothing stored",
            payload.attempt_id,
        )
