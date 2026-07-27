# 행동 이벤트 신뢰 프로토콜 보완안

## 1. 목적

현재 브라우저는 CAPTCHA 검증 시점에 행동 이벤트 전체를 한 번에 제출한다.
따라서 정답 객체만 실제로 선택하고, 행동 궤적은 별도로 만든 사람 유사
데이터로 바꿔 넣을 수 있다.

목표는 브라우저를 신뢰하는 것이 아니라 서버가 다음 사실을 직접 기록하는
것이다.

- 어떤 challenge와 세션에서 이벤트를 받았는가
- 이벤트 배치를 어떤 순서로 받았는가
- 각 배치를 서버가 언제 받았는가
- 이전에 받은 배치가 나중에 교체되지 않았는가
- 검증 전에 실제로 이벤트가 점진적으로 도착했는가

이 프로토콜은 행동이 진짜 사람이라는 사실까지 증명하지는 않는다. 공격자가
실시간으로 사람 유사 이벤트를 생성할 수는 있다. 다만 검증 순간에 완성된
궤적을 한꺼번에 바꿔 끼우는 현재 공격은 막고 공격 비용을 높인다.

## 2. 권장 흐름

```text
문제 발급
  -> 서버가 challenge_id, session binding, 일회성 nonce 생성
  -> 브라우저가 드래그 이벤트를 100~250ms 단위의 작은 배치로 전송
  -> 서버가 batch_seq, received_at, payload_hash, previous_hash 저장
  -> 브라우저는 서버가 확인한 다음 batch_seq와 receipt만 보관
  -> 정답 제출
  -> 서버가 저장한 이벤트 배치만 행동 AI 입력으로 사용
  -> 브라우저가 verify에 다시 보낸 원시 events는 판정에 사용하지 않음
  -> 성공·실패와 무관하게 challenge를 소비하거나 시도 횟수를 증가
```

## 3. 이벤트 배치 API

### 요청

`POST /api/captcha/challenges/{challenge_id}/behavior-events`

```json
{
  "session_id": "bounded-session-id",
  "batch_seq": 3,
  "previous_receipt": "server-issued-receipt",
  "events": [
    {
      "type": "pointer_move",
      "object_id": "temporary-object-id",
      "x": 0.51,
      "y": 0.42,
      "timestamp_ms": 1784790000123
    }
  ]
}
```

제약:

- 한 배치 최대 `32개` 이벤트
- 요청 본문 최대 크기 제한
- `batch_seq`는 `0`부터 하나씩 증가
- challenge와 session이 일치해야 함
- 만료·소비·최대 시도 초과 challenge는 수집 거부
- 객체 ID는 해당 challenge의 임시 ID만 허용
- 좌표와 이벤트 종류는 기존 Pydantic 스키마로 검증

### 응답

```json
{
  "accepted": true,
  "batch_seq": 3,
  "server_received_at": "2026-07-23T07:00:04.123Z",
  "receipt": "server-generated-opaque-value"
}
```

`receipt`는 서버 비밀키로 다음 값을 인증한다.

```text
challenge_id | session_hash | batch_seq | payload_hash |
previous_receipt_hash | server_received_at
```

브라우저 안에 서버 비밀키를 넣지 않는다. 브라우저는 receipt를 다음 배치에
되돌려 줄 뿐이며 직접 새 receipt를 만들 수 없다.

## 4. 서버 저장 구조

권장 테이블:

```text
captcha_behavior_event_batches
  challenge_id
  batch_seq
  event_count
  payload_json
  payload_hash
  previous_hash
  receipt_hash
  server_received_at
  PRIMARY KEY(challenge_id, batch_seq)
```

필수 규칙:

- 같은 `challenge_id + batch_seq` 재전송은 payload hash가 같을 때만 멱등 처리
- 같은 순번에 다른 payload가 오면 `batch_payload_conflict`로 거부
- 다음 순번이 아니면 `batch_sequence_gap`으로 거부
- DB 트랜잭션과 행 잠금을 사용해 동시 요청 순서를 고정
- 원본 IP나 세션 원문 대신 기존 해시 정책 사용
- 보존 기간이 지나면 원시 이벤트를 삭제하고 집계·감사 정보만 유지

## 5. 검증 시 행동 AI 입력

verify 서버는 다음 순서로 처리한다.

1. challenge, session, 만료, 시도 횟수를 확인한다.
2. DB에 저장된 배치를 순서대로 읽는다.
3. 순번 단절, receipt chain 오류, 이벤트 부족을 검사한다.
4. 서버 도착시각과 클라이언트 이벤트 간격을 함께 사용해 모델 입력을 만든다.
5. verify 요청의 `events`는 로그 비교용으로만 두거나 제거한다.
6. 행동 AI의 `allow`, `step_up`, `step_up_and_rate_limit`을 정책에 반영한다.
7. 정답 여부와 무관하게 시도 횟수와 최종 상태를 원자적으로 갱신한다.

첫 단계에서는 클라이언트 시각을 완전히 버리지 않고, 각 배치 내부의 상대
간격만 제한적으로 사용할 수 있다. 배치 사이의 시간은 서버 도착시각을
기준으로 계산한다.

## 6. 위험 판정

다음 조건은 모델 점수가 높아도 최소 `step_up`으로 처리한다.

- 서버 시각 또는 이벤트 배치가 없음
- batch sequence가 끊기거나 역전됨
- receipt chain이 일치하지 않음
- 검증 직전에 대부분의 이벤트가 한꺼번에 도착
- 문제 발급 전 또는 제출 후 시각
- 동일 receipt, payload hash, trace fingerprint 재사용
- 세션 또는 IP 해시 단위 요청 빈도 초과

다음 조건이 함께 나타나면 `step_up_and_rate_limit`을 검토한다.

- 프로토콜 이상 + 낮은 모델 Human score
- 프로토콜 이상 + 반복 challenge 실패
- DTW 유사 궤적 + 높은 세션 요청 빈도
- 일회성 nonce 또는 receipt 재사용

## 7. 장애 정책

- 행동 AI 장애: shadow 단계에서는 메인 CAPTCHA 결과를 유지하고 장애율 기록
- 이벤트 저장 DB 장애: 운영 active 단계에서는 조용히 `allow`하지 않고
  별도 CAPTCHA 또는 재시도로 전환
- 일부 배치 전송 실패: 동일 payload와 batch sequence로 제한된 횟수만 재전송
- 브라우저 네트워크 불안정: 짧은 유예와 멱등 재전송을 허용하되 순서 변경은 금지

## 8. 검증 항목

### 정상 흐름

- 마우스·터치·트랙패드에서 배치 순서와 receipt chain 정상
- 느린 네트워크에서 멱등 재전송 정상
- 마지막 배치와 verify 경쟁 조건 없음
- Human FRR `3% 이하`

### 공격 회귀

- 기존 상대시각 고정 후보
- 현재 challenge 시간으로 이동한 후보
- 동일 배치 재전송
- 같은 순번의 payload 교체
- 순번 누락·역전
- 검증 직전 전체 이벤트 일괄 전송
- 다른 challenge·session의 receipt 재사용
- 동일 궤적의 좌표·시간 소폭 변형

### 배포 기준

- 알려진 Bot ASR `5% 이하`
- 미지 family 최악 ASR `10% 이하`
- Human FRR `3% 이하`
- 프로토콜 오류가 정상 사용자 환경별로 편향되지 않음
- shadow mode에서 충분한 로그를 확인한 뒤에만 active 전환

## 9. 구현 순서

1. 배치 테이블과 DB 메서드를 추가한다.
2. 이벤트 수집 API와 receipt 발급을 구현한다.
3. 프론트에서 `100~250ms`마다 최대 `32개`를 순차 전송한다.
4. verify가 DB 저장 이벤트를 우선 사용하도록 shadow 비교한다.
5. 브라우저 제출 이벤트와 서버 저장 이벤트의 불일치율을 측정한다.
6. 정상 환경별 누락률과 FRR을 확인한다.
7. 기존 고정 holdout과 시간 이동 holdout을 다시 실행한다.
8. 기준을 만족할 때 verify 입력을 서버 저장 이벤트로 완전히 전환한다.
