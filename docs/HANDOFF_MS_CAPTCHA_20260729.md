# 김민서님께 — `sw-captcha` 병합 결과 및 배포 요청 (2026-07-29)

보낸 사람: 조성원 (행동 기반 봇 판별 AI)

---

## 0. 요약

행동 신뢰 프로토콜 + AI 판별을 **민서님 최신 캡차에 병합해서 `origin/sw-captcha`에 올려뒀습니다.**

- 민서님 최신 커밋 5개(`edfa26e`~`32d086c`) **전부 포함**. 기능 손실 없습니다
- 테스트 23개 통과, 실제 기동해서 배치 흐름까지 확인
- PR: https://github.com/catchap-captcha/ai-service/pull/new/sw-captcha

**부탁**: 배포 + 환경변수 1개 변경 (3절). 그리고 **제가 민서님 코드를 두 군데 판단해서 바꿨는데,
그 판단이 맞는지 확인 부탁드립니다** (2절).

---

## 0-1. PoW 병합 완료 — 배포만 남았습니다 (2026-07-29 17:20)

`origin/ms`(09c5351) 8커밋을 `sw-captcha`에 병합해 푸시했습니다. **`523e555`**

```
54a33d7  merge: ms PoW + 방어 스택을 행동 신뢰 프로토콜과 통합
523e555  test: verify() 통합 경로 회귀 테스트 3건
```

테스트 **26 통과** (기존 23 + 신규 3).

### 판단이 필요했던 곳 — 확인 부탁드립니다

민서님 `09c5351`의 그림자 채점이 `payload.events`, 즉 **클라이언트가 보낸 궤적**을
모델에 넘기고 있었습니다. 그러면 봇이 그럴듯한 궤적을 만들어 보내는 것만으로 사람
점수를 받습니다(레드팀에서 사람 궤적 재생 97.5% 확인한 그 경로입니다). 그래서
**채점 소스는 서버 저장 배치로 두고 그 부분만 버렸습니다.** `record_shadow_outcome`은
제 쪽에 이미 있습니다.

**민서님의 규칙 게이트는 전부 살렸습니다** — `automation_score`, 행동지문,
클러스터 차단. 토큰 발급 직전에 그대로 둡니다. 행동 AI 게이트만
`resolve_final_verdict`와 중복이라 제외했습니다.

PoW 게이트는 verify 앞쪽에 뒀는데 **중복선택 422 검사보다는 뒤**입니다. 12개 리스트
중복 검사가 sha256보다 싸고, 그건 PoW와 무관하게 422인 요청이라서입니다.

`create_challenge`는 합집합이라 이제 **`behavior_nonce`와 `pow`가 한 응답에 함께**
실립니다. 병합 전에는 PoW 챌린지에 nonce가 없어서 프런트가 배치를 아예 못 보내는
상태였습니다.

### 배포는 제가 못 합니다

`/srv/codex-workspaces/ms` 가 `drwx------ ms ms` 이고 제 계정은 sudo에 비번이
필요합니다. **`sw-captcha` 배포는 민서님이 해주셔야 합니다.**

---

## (참고) 16:34 재배포 때 확인된 것

오늘 낮에는 `:8000` 에 행동 수집이 **들어가 있었습니다.** 제가 07:32(UTC) 에 실제로
배치 4건 전송 → 전부 `200` + 영수증 정상, `verify` `200` 까지 확인했습니다.

그런데 **16:34:08 에 `:8000` 이 재시작되면서 그 코드가 사라졌습니다.**

```
behavior-batches 라우트   openapi.json 에 0건        (제 코드에선 무조건 등록됩니다)
프론트 번들               "behavior-batches" 0회, "batch_seq" 0회
/health/ready             {status, approved_questions} — behavior_* 필드 전무
```

`BEHAVIOR_EVENT_TRANSPORT=off` 로는 이 증상이 안 나옵니다. 라우트 등록은 설정과 무관하고,
`/health/ready` 도 `off` 여도 `behavior_event_transport: "off"` 를 반환합니다.
**코드 자체가 없는 상태** 입니다 — `sw-captcha` 가 빠진 브랜치로 재배포된 것으로 보입니다.

**16:34 에 무엇을 배포하셨는지 확인 부탁드립니다.** 그 빌드에는 PoW(`pow.seed`/`bits:17`)가
있는데 `sw-captcha` 에는 PoW 코드가 없습니다. 두 작업이 서로 다른 브랜치에 있는 것 같습니다.
**PoW 쪽으로 `sw-captcha` 를 병합**하는 게 맞다면 제가 하겠습니다 — 어느 브랜치인지만
알려주세요.

---

## 1. 무엇이 추가되나

캡차를 푸는 동안의 포인터 궤적을 **서버가 신뢰할 수 있는 형태로** 수집해서 AI가 봇인지 판별합니다.

지금 구조의 문제는 궤적이 verify 시점에 브라우저에서 한 번에 오는 것입니다. 봇이 그럴듯한
궤적을 만들어 넣으면 그대로 통과합니다(레드팀에서 사람 궤적 재생 97.5% 확인).

병합본은 이렇게 바뀝니다:

```
지금:  풀기 → verify 에 events 한꺼번에 제출 → 서버가 그걸 채점
병합:  풀면서 200ms마다 배치 전송(서버 저장) → verify 는 저장된 배치만 채점
                                              → 클라이언트가 보낸 events 는 무시
```

배치마다 nonce + HMAC 영수증 체인 + 순번을 검증하므로 사후 조립·순서 조작이 불가능합니다.
추가로 좌표-객체 결속(press가 정답 객체에서 시작했는지), 전송 타이밍 괴리(긴 궤적을 마지막에
몰아 올리는 패턴)를 봅니다.

**민서님 쪽 기존 동작은 그대로입니다.** 전부 `BEHAVIOR_EVENT_TRANSPORT` 플래그 뒤에 있고,
`off`면 지금과 100% 동일합니다.

---

## 2. ⚠️ 민서님 코드를 두 군데 판단해서 바꿨습니다 — 확인 부탁드립니다

### 2-1. `src/main.jsx` `load()`의 `setEvents([...])` 제거

민서님 코드:
```javascript
setEvents([{ type: "challenge_loaded", object_id: null, x: null, y: null, timestamp_ms: Date.now() }]);
```

**제거했습니다.** 이유 두 가지:

1. 병합본에는 `events` 상태(`useState`) 정의가 없습니다. 그대로 두면 **ReferenceError**로 화면이 깨집니다
2. verify가 더 이상 클라이언트 `events`를 제출하지 않습니다(서버 저장 배치를 채점하므로).
   그 상태 배열을 쓰는 곳이 없습니다

대신 `record("challenge_loaded", null, null)`이 같은 이벤트를 배치 큐에 넣습니다. 기록은 유지됩니다.

### 2-2. `onPointerCancel` — 민서님 인라인 핸들러 대신 `cancelDrag` 유지

민서님 코드:
```javascript
onPointerCancel={()=>{setDragging(null);setDragPoint(null);}}
```

병합본:
```javascript
onPointerCancel={cancelDrag}
```

**이유**: 인라인 핸들러는 드래그 상태만 정리하고 **`pointer_cancel` 이벤트를 기록하지 않습니다.**

`pointer_cancel`은 실제 브라우저에서만 발생하는 이벤트입니다(터치가 시스템에 가로채질 때 등).
합성 궤적(봇)에서는 거의 나오지 않아서 **사람다움을 나타내는 신호 중 판별력이 높은 축**입니다.
기록이 빠지면 그 신호가 조용히 사라집니다.

`cancelDrag`는 드래그 상태 정리 + `pointer_cancel` 기록을 둘 다 합니다. UI 동작은 동일합니다.

**둘 다 제 판단이니, 의도와 다르면 알려주세요.**

---

## 3. 배포 요청

### 3-1. 배포

`origin/sw-captcha`를 배포해주세요. 지금 배포본(`ms`)과의 차이는 위 1절 + 판단 2건뿐입니다.

### 3-2. 환경변수 (이것만 바꾸면 됩니다)

```diff
- BEHAVIOR_EVENT_TRANSPORT=off
+ BEHAVIOR_EVENT_TRANSPORT=shadow
```

**나머지는 바꾸지 마세요:**

| 변수 | 값 | 비고 |
|---|---|---|
| `BEHAVIOR_AI_URL` | `http://127.0.0.1:8010` | 기본값. AI가 같은 GPU 서버에 있습니다 |
| `BEHAVIOR_AI_BACKEND_KEY` | ⚠️ **AI 쪽 값과 일치시켜야 합니다** | 아래 3-2b 참조 |
| `BEHAVIOR_POLICY_MODE` | `shadow` | 유지 |

### 3-2b. ⚠️ 정정 (2026-07-29 16:40) — 백엔드 키는 "그대로 두면" 안 됩니다

위 표에서 처음에 **"현재 값 그대로, 변경 불필요"** 라고 적었습니다. **제가 틀렸습니다.**
AI가 기존 키를 승계했다고 가정했는데, 실제로는 AI가 자기 `.env`의 `CAPTCHA_BACKEND_API_KEY`
(64자)를 쓰고 있고 캡차가 보내는 값과 다릅니다. 그래서 캡차→AI 호출이 전부
**401 Unauthorized** 로 거부됩니다.

401이 나면 `behavior_client` 가 fail-open 으로 "점수 없음"을 반환하므로 **캡차는 정상
통과되고 에러도 안 보입니다.** 대신 AI 기록이 한 건도 남지 않습니다. 실제로
`ai_behavior_attempts` 에 캡차에서 온 행이 0건입니다.

```bash
# GPU 서버에서 — AI 가 기대하는 값
grep '^CAPTCHA_BACKEND_API_KEY=' /home/sw/catchap-behavior/.env

# 캡차 .env 의 BEHAVIOR_AI_BACKEND_KEY 를 위 값과 똑같이 맞춰주세요
```

확인 방법 (풀이 1회 후):

```bash
grep 'behavior/predict' /home/sw/catchap-behavior/logs/behavior-ai.log | tail -3
#   200 OK  → 정상 연동
#   401     → 아직 키가 다릅니다
```

키를 제 쪽에서 캡차 값에 맞춰도 됩니다. 어느 방향이든 알려주시면 제가 맞추겠습니다.

`shadow`는 **AI 점수를 기록만 하고 통과/실패 판정에 반영하지 않습니다.** 사용자 영향 0입니다.
`active`는 오탐률 측정 전까지 켜지 마세요 (참여자 54명 중 4명이 3% 초과 상태).

### 3-3. 배포 후 확인

```bash
# 1. 라우트가 생겼는지
curl -s http://127.0.0.1:8000/openapi.json | grep -c behavior-batches   # 1 이상

# 2. AI 가 살아있는지
curl -s http://127.0.0.1:8010/health
#   → status:ok, mysql_connected:true, model_loaded:true 여야 정상
```

AI는 제가 이미 준비해뒀습니다 — 최신 코드 + DB 연결 + 재생 탐지 동작 확인 완료.

---

## 4. 별건 — `postMessage` targetOrigin 지적

`src/main.jsx`:
```javascript
window.parent.postMessage({ type:"catchap-verified", token: ..., lecture_id: ... }, "*")
```

`"*"`는 **어느 부모창이든 이 토큰을 받을 수 있다**는 뜻입니다. 캡차 iframe이 악성 페이지에
끼워지면 그 페이지가 토큰을 가져갑니다.

실제 위험은 낮습니다 — 토큰이 `lecture_id`에 묶여 있고, 검증에 사이트 시크릿이 필요하고,
1회용입니다. 다만 정석은 부모 origin을 명시하는 것입니다.

**제안**: `ALLOWED_ORIGINS` 설정이 이미 있으니 그걸 재사용하면 됩니다.

1. `/api/config` 응답에 허용 embed origin을 포함 (지금은 `siteKey`만 반환)
2. 프론트가 `"*"` 대신 그 값을 targetOrigin으로 사용
3. 여러 개면 embed URL에 어느 것인지 실어 보내고 **서버가 허용목록과 대조**
   (클라이언트가 준 값을 그대로 믿으면 안 됩니다)

제가 고쳐도 되지만 민서님 영역이라 먼저 말씀드립니다. 원하시면 제가 PR로 올리겠습니다.

---

## 5. 인강 연동 관련

`1abcebe`(임베드 위젯 + `verify-token` + 강의 바인딩) 잘 봤습니다. `lecture_id`를 챌린지·토큰에
박고 `verify_token`에서 일치를 확인하는 구조, 그리고 `FOR UPDATE` + `consumed_at IS NULL`로
동시 재사용을 막는 부분 정확합니다.

저희가 만들려는 건 **"이상행동이 감지되면 이 위젯을 띄운다"**입니다. 감지·승급은 백엔드(김태형님)
쪽 작업이고, 캡차는 지금 상태로 충분합니다. **민서님이 추가로 하실 일은 없습니다.**

---

문의: 조성원 (wwdhogo@gmail.com)
