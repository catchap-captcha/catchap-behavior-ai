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
| `BEHAVIOR_AI_BACKEND_KEY` | **현재 값 그대로** | AI 쪽이 기존 키를 승계했습니다. 변경 불필요 |
| `BEHAVIOR_POLICY_MODE` | `shadow` | 유지 |

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
