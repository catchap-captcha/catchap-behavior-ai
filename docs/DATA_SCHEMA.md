# 데이터 스키마 (원본 vs Feature)

## 원본(raw)과 Feature의 차이

| 구분 | 무엇 | 어디에 |
|------|------|--------|
| **원본** | 포인터 이벤트 그 자체 (`seq, t_ms, x, y, ...`) | `ai_pointer_events` |
| **Feature** | 원본에서 계산한 29개 요약 수치 | `ai_attempt_features` |

핵심 원칙: **Feature는 원본으로 언제든 다시 계산할 수 있습니다.** Feature 정의가
바뀌면 `feature_schema_version`을 올리고, 저장된 원본으로 전체 재계산합니다.
서로 다른 버전의 Feature는 **섞어서 학습하지 않습니다.**

- Feature 계산 로직의 단일 출처: `app/services/feature_extractor.py`
- 현재 `feature_schema_version = "1.0"`

## 행동 Feature 29개

### A. 기본 (15)
`event_count, duration_ms, total_distance, displacement, avg_speed, max_speed,
speed_std, avg_acceleration, max_acceleration, jerk_mean, direction_changes,
pause_count, pause_ratio, linearity, y_deviation`

### B. 이벤트 간격 (4)
`interval_mean_ms, interval_std_ms, interval_cv, duplicate_interval_ratio`

### C. 목표 근처 보정 (5)
`overshoot_count, overshoot_distance, correction_count, endpoint_adjustment_time,
final_segment_speed`

### D. 조작 (5)
`regrab_count, retry_count, pointercancel_count, empty_click_count, failed_drop_count`

**단위:** 거리=CAPTCHA px, 시간=ms, 속도=px/ms, 가속도=px/ms², jerk=px/ms³.
0으로 나누기·동일 timestamp·누락값은 모두 방어 처리되어 결과는 항상 유한값입니다.

## 모델 입력에서 제외되는 열

식별자·출처·라벨·판정 신호는 **Feature가 아닙니다.** 학습 시 아래 열은 입력에서
제거합니다 (`feature_extractor.MODEL_INPUT_EXCLUDE_COLUMNS`):

```
attempt_id, challenge_id, session_id, anonymous_participant_id,
label, label_source, bot_family, generator_version,
schema_version, feature_schema_version,
position_correct, interaction_success, final_drop_error,
human_score, bot_risk_score, bot_decision, model_version
```

`label`은 정답(Human=1, Bot=0)이며 Feature가 아닙니다.

## Replay / 세션 보안 Feature (별도)

`path_similarity_score, exact_replay_detected, repeated_duration_count,
attempts_per_minute, recent_attempt_count, repeated_endpoint_count`

- 1차 행동 모델 입력과 **분리**되어 있으며, 최종 위험도 결합 단계에서 사용합니다.
- 유사도 백엔드는 `PathComparator` 인터페이스로 분리되어 나중에 DTW/ANN으로
  교체 가능합니다. (`app/services/replay_detector.py`)

## 라벨 & 익명화 & 아동 데이터 주의

- 참여자는 **익명 ID(`anonymous_participant_id`)**로만 저장합니다. 이름·이메일 등
  개인 식별정보는 저장하지 않습니다.
- `age_group`(adult/child/unknown), `consent_version`(동의서 버전)을 기록합니다.
- **아동 데이터는 보호자 동의를 받은 것만** 수집합니다.
- 성인 데이터로만 학습한 모델은 아동의 조작 패턴을 봇으로 **오탐**할 수 있습니다.
  배포 대상 연령대를 확인하고, 필요하면 연령대별 검증을 수행하세요.

## 임의 생성 금지

- Human 데이터를 **임의로 생성하거나** 실제 Human 데이터인 것처럼 저장하지
  않습니다.
- 테스트 fixture는 `tests/fixtures` 및 테스트 코드 안에서만 사용하며, 실데이터
  경로(`data/processed`)나 `models/production`에 저장하지 않습니다.
