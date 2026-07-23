# CAPTCHA 팀 API 규격

ai-service는 웹 드래그 CAPTCHA의 **원본 포인터 이벤트**를 받습니다. 웹 전용이므로
**기기 모델·OS·화면 방향·디바이스 지문은 수집하지 않습니다.** CAPTCHA 픽셀 크기와
정규화 좌표는 유지합니다.

- Base 경로: `/api/v1/behavior`
- `schema_version`: 현재 `"1.0"`

## 1) 수집 — `POST /api/v1/behavior/collect`

**통제된 데이터 수집 전용.** CAPTCHA **백엔드만** 호출합니다.

- 인증: 헤더 `X-API-Key: <COLLECT_API_KEY>` (없거나 틀리면 `401`)
- 라벨은 **신뢰된 `collection` 컨텍스트**로만 설정됩니다. 프론트엔드는 이 엔드포인트에
  접근할 수 없으므로 라벨을 임의로 정할 수 없습니다.
- 동일 `attempt_id` 재요청은 **idempotent** (재저장하지 않고 기존 상태 반환).

### 요청 본문

```json
{
  "schema_version": "1.0",
  "attempt_id": "att_000001",
  "challenge_id": "captcha_000123",
  "session_id": "session_001",
  "anonymous_participant_id": "adult_001",
  "captcha": { "width": 420, "height": 220 },
  "timing": {
    "presented_at": "2026-07-01T10:00:00Z",
    "submitted_at": "2026-07-01T10:00:02Z"
  },
  "events": [
    { "seq": 0, "event_type": "pointerdown", "t_ms": 0,   "x": 10,  "y": 31, "x_normalized": 0.024, "y_normalized": 0.141, "target_role": "slider_handle" },
    { "seq": 1, "event_type": "pointermove", "t_ms": 16,  "x": 15,  "y": 32, "x_normalized": 0.036, "y_normalized": 0.145, "target_role": "slider_handle" },
    { "seq": 2, "event_type": "pointerup",   "t_ms": 840, "x": 278, "y": 30, "x_normalized": 0.662, "y_normalized": 0.136, "target_role": "slider_handle" }
  ],
  "interaction": {
    "regrab_count": 0, "retry_count": 0, "pointercancel_count": 0,
    "empty_click_count": 0, "failed_drop_count": 0
  },
  "collection": {
    "label": "human",
    "label_source": "controlled_collection",
    "bot_family": null,
    "generator_version": null,
    "age_group": "adult",
    "consent_version": "v1"
  },
  "position_correct": true,
  "interaction_success": true,
  "final_drop_error": 1.2
}
```

- `event_type` ∈ `pointerdown | pointermove | pointerup | pointercancel`
- `collection.label` ∈ `human | bot | unknown`
- `collection.label_source` ∈ `controlled_collection | playwright | selenium | rule_bot | gan_bot | replay_bot`
- `collection.age_group` ∈ `adult | child | unknown`
- `position_correct` / `interaction_success` / `final_drop_error` 는 저장되지만
  **행동 모델 Feature로 쓰지 않습니다** (CAPTCHA 통과 판정용).

### 취약문제 추천용 `learning` 블록 (선택)

봇탐지는 위 포인터 이벤트(HOW)만으로 되지만, **취약문제 추천/진단**에는 아래
`learning` 블록(WHAT)이 필요합니다. 있으면 collect가 학습 데이터도 함께 저장하고
**조작실수 판정(정답/개념오답/조작실수)까지 계산**해 둡니다. 없으면 봇탐지만 저장.

```json
"learning": {
  "question_id": "q_add_3plus2",
  "concept_id": "ADD_WITHIN_5",
  "difficulty": 0.3,
  "answer_options_count": 3,
  "correct_answer_id": "5",
  "grabbed_answer_id": "5",       // 집은 타일 (= 의도한 답)
  "released_target_id": "slot",   // 놓은 곳; null이면 드롭 실패
  "answer_slot_id": "slot",
  "game_type": "number_add"
}
```

- `correct_answer_id` / `concept_id` / `difficulty` 는 **백엔드가 문제은행에서 조회**해
  넣습니다 (프론트 값 신뢰 금지 — 치팅 방지).
- 응답의 `learning_stored: true` 로 저장 여부 확인.

### 응답 `200`

```json
{
  "attempt_id": "att_000001",
  "stored": true,
  "idempotent": false,
  "quality_status": "valid",
  "rejection_reason": null,
  "feature_schema_version": "1.0"
}
```

- `quality_status` ∈ `valid | pending | rejected` (품질 낮아도 원본은 저장됨)

## 2) 위험 평가 — `POST /api/v1/behavior/predict`

**CAPTCHA 백엔드 전용입니다.** 브라우저는 호출하면 안 되며,
`X-Captcha-Backend-Key`가 필요합니다. 요청의 `session_id`도 백엔드가 자신의
세션에서 조회해 채워야 합니다.

라벨을 받지 않고, 단일 궤적 ML 점수와 같은 세션의 최근 Trace Fingerprint,
DTW 유사도, 요청 빈도를 합쳐 **정책 위험 점수**를 반환합니다. 이 점수는
`P(bot)`이 아니며, AI 서비스는 통과·차단을 결정하지 않습니다. 백엔드가
`recommended_action`을 사용해 추가 캡차나 요청 제한 여부를 결정합니다.

요청 본문은 `collect`와 동일하되
`collection` / `position_correct` 등 라벨·판정 필드는 **없습니다.**

### 응답 `200`

```json
{
  "attempt_id": "att_000001",
  "risk_score": 5.0,
  "risk_level": "low",
  "recommended_action": "allow",
  "policy_mode": "shadow",
  "reasons": [],
  "human_score": 0.87,
  "bot_risk_score": 0.13,
  "path_similarity_score": 0.18,
  "exact_replay_detected": false,
  "attempts_per_minute": 1.0,
  "threshold": 0.55,
  "model_name": "xgboost",
  "model_version": "xgboost_v1",
  "feature_schema_version": "1.0"
}
```

`recommended_action` 값:

- `allow`: 백엔드의 일반 흐름 유지
- `step_up`: 새 캡차 또는 추가 검증 권장
- `step_up_and_rate_limit`: 추가 검증과 세션 요청 제한 권장

어떤 값도 AI 서비스 단독 차단을 뜻하지 않습니다.

`policy_mode`는 backend가 권고를 실제 적용할지 나타냅니다.

- `shadow`: `recommended_action`을 기록만 하고 현재 CAPTCHA의 통과·실패는 바꾸지 않습니다.
- `active`: shadow 검증을 마친 뒤에만 backend가 권고를 적용할 수 있습니다.

### Shadow 결과 기록 — `POST /api/v1/behavior/shadow/outcomes`

`RISK_POLICY_MODE=shadow`일 때만 CAPTCHA backend가 호출합니다. 반드시 먼저
`/predict`를 호출한 같은 `attempt_id`만 기록할 수 있습니다. 답안 원문이나 문제 내용은
전송하지 않습니다.

```json
{
  "attempt_id": "att_000001",
  "main_captcha_verdict": "passed",
  "final_verdict": "passed"
}
```

shadow에서는 AI 권고가 실제 결과를 바꾸지 않으므로 `main_captcha_verdict`와
`final_verdict`는 반드시 같습니다. 같은 `attempt_id`의 재전송은 idempotent입니다.

```json
{
  "attempt_id": "att_000001",
  "stored": true,
  "idempotent": false,
  "policy_mode": "shadow",
  "would_have_action": "step_up",
  "risk_level": "medium",
  "model_version": "revalidation_two_view_participant_safe_20260722"
}
```

### Shadow 요약 — `GET /api/v1/admin/shadow/summary`

관리자 전용(`X-Admin-Key`)입니다. 원본 답안이나 개별 궤적을 반환하지 않고,
`allow`/`step_up`/`step_up_and_rate_limit`별 시도 수와 main CAPTCHA 통과·실패 수,
AI가 추가 인증을 권고했을 비율만 반환합니다.

### 모델이 없을 때 `503`

가짜 점수를 반환하지 않습니다.

```json
{ "reason": "model_not_ready", "detail": "No production model is loaded. ..." }
```

## 3) 상태 — `GET /health`

```json
{
  "status": "ok",
  "mysql_connected": true,
  "model_loaded": false,
  "model_name": null,
  "model_version": null,
  "feature_schema_version": "1.0",
  "policy_mode": "shadow"
}
```

## 4) 일회성 Challenge — CAPTCHA 백엔드 전용

이 API는 **브라우저가 호출하면 안 됩니다.** CAPTCHA 백엔드만
`X-Captcha-Backend-Key: <CAPTCHA_BACKEND_API_KEY>`를 사용합니다. 백엔드는
브라우저가 보낸 임의 session ID를 신뢰하지 않고, 자신의 pre-auth 세션에서
session ID를 조회해 요청에 넣어야 합니다.

### 발급 — `POST /api/v1/captcha/challenges`

```json
{
  "session_id": "preauth_session_001",
  "site_key": "catchap-web",
  "purpose": "login_guard",
  "problem_binding": "question-batch-v3:sha256:...",
  "ttl_seconds": 120
}
```

응답의 `challenge_id`, `nonce`, `expires_at`을 CAPTCHA 백엔드의 현재 세션과
문제 배치에 연결한다. nonce와 문제 바인딩 원문은 AI DB에 저장하지 않고 SHA-256
해시만 저장한다.

### 소비 — `POST /api/v1/captcha/challenges/consume`

CAPTCHA 백엔드가 문제 정답을 자체 검증한 후 호출한다. `verdict`가 `passed`든
`failed`든, 바인딩이 맞는 첫 요청은 challenge를 소비한다.

```json
{
  "challenge_id": "ch_...",
  "nonce": "발급 때 받은 nonce",
  "session_id": "preauth_session_001",
  "site_key": "catchap-web",
  "purpose": "login_guard",
  "problem_binding": "question-batch-v3:sha256:...",
  "verdict": "passed"
}
```

| 상황 | HTTP | reason |
|---|---:|---|
| 최초 바인딩 일치 소비 | 200 | 응답 `consumed: true` |
| 같은 challenge 재전송 | 409 | `challenge_already_consumed` |
| 만료 후 소비 | 410 | `challenge_expired` |
| nonce·세션·site key·purpose·문제 바인딩 불일치 | 403 | `challenge_binding_invalid` |

소비는 `status='issued' AND expires_at > now` 조건부 UPDATE로 처리한다. 따라서
동시 요청도 최초 한 건만 상태를 `consumed`로 바꿀 수 있다.

## 참고

- 좌표 `x,y`는 CAPTCHA 영역 픽셀 기준, `x_normalized,y_normalized`는 0~1.
- `t_ms`는 드래그 시작(첫 `pointerdown`) 이후 경과 시간(ms), **감소하면 안 됨**.
- 이벤트는 최소 2개 이상, `seq`는 0부터 1씩 증가해야 `valid` 판정에 유리합니다.
