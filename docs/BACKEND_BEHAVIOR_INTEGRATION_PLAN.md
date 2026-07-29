# 백엔드 캡차에 행동 신뢰 프로토콜 이식 — 계획 (경로 B)

작성: 최성우 (행동 AI 담당) / 2026-07-29
대상: 백엔드 `catchap-backend` (`210.109.52.124`, compose 프로젝트 `catchap-backend`)
협의 대상: 김태형(백엔드), DB팀, 김민서(캡차 — 경로 A 보류 확인)

---

## 0. 왜 백엔드인가 (경로 A와 비교)

| | A. 김민서 드래그 캡차 | **B. 백엔드 자체 캡차** |
|---|---|---|
| 방어 구현 상태 | 완성 (테스트 23 통과) | 이식 필요 (1~2일) |
| **인강 체크포인트 연결** | ❌ 안 됨 | ✅ **이미 연결됨** |
| 배포 결정권 | 김민서 | 김태형 |
| AI 위치 | 크로스 호스트 노출 필요 | ✅ **같은 호스트 `127.0.0.1:8010`** |
| 궤적 수집 코드 | 있음 | ✅ **이미 있음** (`catchap-widget.js`) |

2026-07-28 16:46 김민서가 메인 캡차를 재배포했으나 행동 방어는 포함되지 않았다
(`/api/captcha/challenges/{id}/behavior-batches` 부재 확인). A는 통제 밖의 배포 결정에
계속 묶이고, 붙여도 인강 경로와 연결되지 않는다.

---

## 1. 지금 백엔드 캡차가 위조에 취약한 이유 (코드 근거)

`app/static/catchap-widget.js`:

```javascript
var trace = [], TRACE_MAX = 1500;
function tracePoint(e, force) {
  if (e && e.pointerType) inputType = e.pointerType;      // mouse|touch|pen
  var x = (e.clientX - r.left) / r.width;                 // 정규화 좌표
  if (!force && t - traceLastT < 16) return;              // ~16ms 스로틀
}
...
var behavior = { ... };                                    // trace 포함
api(base, '/captcha/v1/verify', key, { challenge_token, answer, behavior })
```

**궤적 전체가 verify 시점에 클라이언트에서 한 번에 도착한다.** 서버는 그것을
`_normalize_trace`로 형식 검증하고 `_behavior_risk_level`로 채점한 뒤
`record_behavior_event`로 적재한다. 형식은 검증하지만 **출처는 검증하지 않는다.**

→ 봇은 그럴듯한 궤적을 오프라인에서 합성해 넣으면 된다. 실제 사람 궤적을 재생하면
규칙 기반 위험도에 걸리지 않는다(레드팀에서 재생 97.5% 확인).

## 2. 이미 있어서 재사용할 것

| 자산 | 위치 | 재사용 방식 |
|---|---|---|
| 포인터 이벤트 수집 | `catchap-widget.js` `tracePoint` | 전송부만 교체 |
| 궤적 형식 검증 | `captcha_service._normalize_trace` | 그대로 |
| 규칙 기반 위험도 | `captcha_service._behavior_risk_level` | 그대로 (AI와 병행) |
| 행동데이터 적재 | `record_behavior_event` → `behavior_summaries`/`behavior_traces` | 입력만 "저장 배치"로 교체 |
| nonce 1회성 패턴 | challenge nonce·verdict jti DB UNIQUE | 배치 nonce도 같은 패턴 |
| 행동 AI | `127.0.0.1:8010` (동일 호스트, 가동중) | HTTP 호출 |

**핵심: 백엔드가 이미 하는 일을 바꾸는 게 아니라, 입력의 출처를 신뢰 가능하게 만드는 것.**

---

## 3. 단계별 계획

### Phase 0 — 합의 (코드 전)

- [ ] 김태형: 수정 대상 3파일 동의 — `captcha_service.py`, `captcha_api.py`, `catchap-widget.js`
- [ ] 김태형: `/captcha/v1/*`가 **외부 고객사에 공개된 API인지** 확인
      (공개 API면 기존 필드 제거 금지, 추가만 — 하위호환 필수)
- [ ] DB팀: 테이블 2개 추가 (Phase 1 DDL)
- [ ] 김민서: 경로 A 보류 확인 (같은 방어를 두 곳에 유지하지 않기 위해)

### Phase 1 — DB 테이블 2개 (alembic)

배포 스키마 관례에 맞춤: `CHAR(36)` PK, `DATETIME(6)`, `utf8mb4_unicode_ci`, CHECK 제약.

```sql
CREATE TABLE captcha_behavior_sessions (
    challenge_id         CHAR(36)     NOT NULL,   -- 챌린지당 1행
    nonce_hash           CHAR(64)     NOT NULL,   -- 배치 인증용 nonce의 SHA-256
    next_batch_seq       INT UNSIGNED NOT NULL DEFAULT 0,
    last_receipt_hash    CHAR(64)     NULL,       -- 체인의 현재 끝
    received_event_count INT UNSIGNED NOT NULL DEFAULT 0,
    created_at           DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (challenge_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE captcha_behavior_batches (
    id                     CHAR(36)         NOT NULL,
    challenge_id           CHAR(36)         NOT NULL,
    batch_seq              INT UNSIGNED     NOT NULL,
    event_count            SMALLINT UNSIGNED NOT NULL,
    previous_receipt_hash  CHAR(64)         NULL,     -- 첫 배치는 NULL
    payload_hash           CHAR(64)         NOT NULL,
    receipt_hash           CHAR(64)         NOT NULL,
    events_json            JSON             NOT NULL,
    received_at            DATETIME(6)      NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_cbb_challenge_seq (challenge_id, batch_seq),  -- 순번 재사용 차단
    KEY idx_cbb_challenge (challenge_id),
    CONSTRAINT fk_cbb_session FOREIGN KEY (challenge_id)
        REFERENCES captcha_behavior_sessions (challenge_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

`UNIQUE(challenge_id, batch_seq)`가 순번 재사용을, FK CASCADE가 고아 배치를 막는다.

### Phase 2 — 서버: 신뢰 프로토콜

**2-1. 챌린지 발급에 nonce 추가** (`captcha_api.py` `/challenge`)

- 랜덤 nonce 생성 → 응답에 평문 1회 반환, DB에는 SHA-256만 저장
- 기존 Fernet 챌린지 토큰 구조는 건드리지 않는다 (필드 추가만)

**2-2. 배치 수집 엔드포인트 신설**

```
POST /captcha/v1/behavior-batches
  { challenge_token, nonce, batch_seq, previous_receipt, events[] }
  → { accepted: true, receipt }
```

검증 순서 (하나라도 실패 시 거부):
1. `nonce` 해시가 세션의 `nonce_hash`와 일치
2. `batch_seq == next_batch_seq` (순번 강제)
3. `previous_receipt == last_receipt_hash` (체인 연결)
4. `events` 개수·좌표 범위 상한 (기존 `_normalize_trace` 재사용)
5. `receipt = HMAC(server_key, previous_receipt || payload_hash || batch_seq)` 계산·저장·반환

**2-3. verify 변경 — 여기가 방어의 본체**

```
현재:  클라이언트 behavior.trace 를 채점
변경:  captcha_behavior_batches 에서 조립한 궤적만 채점.
       클라이언트 behavior.trace 는 받되 무시 (하위호환용으로 필드는 유지)
```

추가 검증:
- **체인 무결성** — 배치들의 receipt 체인이 끊기지 않았는지, `received_event_count` 일치
- **좌표-객체 결속** — `pointerdown`이 정답 객체의 hit region에서 시작했는지 (드래그 유형만)
- **전송 타이밍** — 클라이언트가 주장한 시간 폭 vs 서버가 실제로 받은 시간 폭의 괴리
  (긴 궤적을 마지막에 한꺼번에 올리는 패턴 탐지)
- **lifecycle** — 이벤트 종류·순서가 실제 상호작용과 맞는지

**2-4. `record_behavior_event` 입력 교체**

`behavior_summaries`/`behavior_traces` 적재는 그대로 두고, 입력만 클라이언트 trace →
서버 저장 배치로 바꾼다. **기존 28,957행과 계보가 이어지되 이후 데이터는 신뢰 가능해진다.**

### Phase 3 — 위젯 (`catchap-widget.js`)

- `tracePoint`가 모은 이벤트를 **200ms마다 배치 전송** (`TRACE_MAX` 대신 배치 단위)
- 응답 `receipt`를 보관해 다음 배치의 `previous_receipt`로 사용
- verify 시 trace 덤프는 유지(무시되지만 하위호환) 또는 제거 — Phase 0의 공개 API 여부로 결정
- 전송 실패 시: 재시도 1회 후 포기. **shadow 단계에서는 사용자를 막지 않는다**

⚠️ **미결정 사항**: 위젯은 드래그 유형과 클릭 유형(그림 다중선택·간단 셈)을 모두 다룬다.
클릭 유형은 궤적이 빈약해 AI 판별력이 낮다. 두 안 중 선택 필요:
- (i) 드래그 유형에만 배치 프로토콜 적용 — 안전하지만 커버리지 부분적
- (ii) 전 유형 적용, 클릭 유형은 AI 점수 대신 체인 무결성만 사용
→ 권고: **(ii)**. 체인·타이밍·결속은 클릭에도 유효하고, AI 점수만 유형별로 가중치를 다르게 둔다.

### Phase 4 — AI 연결

- `behavior_client.py`(423줄) 이식 → `http://127.0.0.1:8010/api/v1/behavior/predict`
- `CAPTCHA_BACKEND_API_KEY`를 백엔드 `.env.production`에 추가
  (AI 쪽 값은 `~/catchap-behavior-ai/.env`에 이미 있음 — GPU 것을 승계한 64자 키)
- **fail-open 유지**: AI가 죽어도 캡차는 동작. 단 실패를 **로그·카운터로 남긴다**
  (조용히 꺼지는 것이 가장 위험한 실패 모드 — 2026-07-28에 실제로 겪음)
- shadow 모드: AI 점수를 기록만 하고 통과/실패에 반영하지 않는다

### Phase 5 — 검증 (로컬)

2026-07-28에 만든 환경 재사용: 프로덕션 스키마 복제본 `catchap_prodlike` (로컬 MySQL 3307).

확인 항목:
- [ ] 정상 사용자: 배치 정상 전송 → verify 통과, 오탐 없음
- [ ] 배치 미전송: shadow에서 통과 + `behavior_batches_missing` 기록
- [ ] 위조 궤적을 verify에 실어 보냄 → **무시됨** (저장 배치가 없으니 채점 불가)
- [ ] 순번 건너뛰기·receipt 불일치 → 배치 거부
- [ ] 동일 궤적 재투입 → AI가 `exact_replay=True` (프로덕션 스키마에서 이미 실증)
- [ ] 배치를 마지막에 몰아 전송 → `batch_delivery_timing` 탐지
- [ ] 기존 캡차 유형 전부 회귀 통과

### Phase 6 — 점진 활성화

```
shadow 배포  →  실트래픽 수집  →  오탐률 측정  →  active 논의
```

`active`는 오탐률이 목표(3% 이하)를 만족한 뒤에만. 현재 참여자별 최악 26.67%,
54명 중 4명이 3% 초과 상태이므로 **지금 active는 금지.**

---

## 4. 일정·규모

| Phase | 작업 | 소요 | 담당 |
|---|---|---|---|
| 0 | 합의 | 0.5일 | 전원 |
| 1 | 테이블 2개 (alembic) | 0.5일 | DB팀 + 나 |
| 2 | 서버 프로토콜 | 1일 | 나 (김태형 리뷰) |
| 3 | 위젯 | 0.5일 | 나 (김태형 리뷰) |
| 4 | AI 연결 | 0.5일 | 나 |
| 5 | 로컬 검증 | 0.5일 | 나 |
| 6 | shadow 배포 | 0.5일 | 김태형 + 나 |

**합계 약 4일.** Phase 2~5는 제가 하고 김태형님 리뷰만 받으면 됩니다.

## 5. 되돌리기

- Phase 2~4는 전부 **기능 플래그** 뒤에 둔다 (`BEHAVIOR_EVENT_TRANSPORT=off|shadow`)
- `off`면 기존 동작과 100% 동일 — 위젯도 배치 전송을 하지 않는다
- 테이블 2개는 남아도 무해 (다른 코드가 참조하지 않음)

## 6. 미결정 / 확인 필요

1. **`/captcha/v1/*`가 외부 공개 API인가** — `site_key`/`secret_key` 구조가 SaaS형이라
   외부 고객이 있을 수 있다. 있다면 필드 추가만 허용되고 위젯 배포 조율이 필요하다
2. **클릭 유형 처리** — Phase 3의 (i)/(ii) 선택
3. **김민서 캡차(경로 A)는 어떻게 할지** — 같은 방어를 두 코드베이스에 유지하는 것은 피해야 한다.
   `sw-captcha` 브랜치를 보존만 하고 배포는 안 하는 쪽을 권고
4. **인강 체크포인트에 언제 적용할지** — 캡차 전체에 켤지, 체크포인트만 먼저 켤지

## 7. 이 작업이 닫는 구멍

2026-07-28 분석에서 확인된 것:

| 공격 | 현재 백엔드 캡차 | 이식 후 |
|---|---|---|
| 궤적 위조 (오프라인 합성) | ❌ 통과 | ✅ 저장 배치만 채점 |
| 사람 궤적 재생 | ❌ 통과 | ✅ receipt chain + AI DTW |
| 궤적 없이 정답만 제출 | ❌ 통과 | ✅ `behavior_batches_missing` |
| 마지막에 몰아 전송 | ❌ 탐지 못 함 | ✅ 전송 타이밍 괴리 |
| 인강 체크포인트 자동 통과 | ❌ 정답만 검사 | ✅ 행동 검증 추가 |

**막지 못하는 것 (정직하게)**: 사람 대행, 완벽 페이싱 재생(±400ms 이내),
시험 자동화(별도 통제 필요 — `docs` 별건), 영상 유출·계정 공유(범위 밖).
