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

## 2) 추론 — `POST /api/v1/behavior/predict`

**운영 추론용. 라벨을 받지 않습니다.** 요청 본문은 `collect`와 동일하되
`collection` / `position_correct` 등 라벨·판정 필드는 **없습니다.**

### 응답 `200`

```json
{
  "attempt_id": "att_000001",
  "prediction": "human",
  "human_score": 0.87,
  "bot_risk_score": 0.13,
  "bot_decision": "low_risk",
  "threshold": 0.55,
  "model_name": "xgboost",
  "model_version": "xgboost_v1",
  "feature_schema_version": "1.0"
}
```

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
  "feature_schema_version": "1.0"
}
```

## 참고

- 좌표 `x,y`는 CAPTCHA 영역 픽셀 기준, `x_normalized,y_normalized`는 0~1.
- `t_ms`는 드래그 시작(첫 `pointerdown`) 이후 경과 시간(ms), **감소하면 안 됨**.
- 이벤트는 최소 2개 이상, `seq`는 0부터 1씩 증가해야 `valid` 판정에 유리합니다.
