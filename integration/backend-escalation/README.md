# 백엔드 승급 연동 — 적용 패치 (2026-07-29)

작성: 최성우 / 대상: `catchap-backend` (`210.109.52.124:/home/ubuntu/catchap-backend`)

**배포는 하지 않았습니다.** 코드만 준비했습니다. 적용은 백엔드 담당자가 검토 후 하시면 됩니다.

## 무엇을 하는가

```
평소 인강 시청 — 캡차 없음
      ↓  이상행동 감지 (하트비트마다)
메인 캡차(김민서 드래그) 승급
      ↓  푸는 동안 궤적 수집 → 행동 AI 판별
통과 → 의심도 리셋, 시청 계속
```

인강 체크포인트 캡차("이 대목 봤는가")는 **건드리지 않습니다.** 뜨는 이유가 다른 별개 장치입니다.

## 파일

| 파일 | 적용 |
|---|---|
| `config.patch` | `app/core/config.py` (+14) |
| `lecture.patch` | `app/models/lecture.py` (+5) |
| `lecture_service.patch` | `app/services/lecture_service.py` (+101/-1) |
| `lectures.patch` | `app/api/v1/endpoints/lectures.py` (+59) |
| `main_captcha_client.py` | **신규** → `app/clients/main_captcha_client.py` |
| `lecture_botsusp_01_bot_suspicion.py` | **신규** → `alembic/versions/` |
| `test_escalation.py` | 로직 검증 (DB·FastAPI 불필요) |

```bash
cd /home/ubuntu/catchap-backend
patch -p1 < config.patch
patch -p1 < lecture.patch
patch -p1 < lecture_service.patch
patch -p1 < lectures.patch
cp main_captcha_client.py app/clients/
cp lecture_botsusp_01_bot_suspicion.py alembic/versions/
alembic upgrade head
```

원본이 **CRLF** 라 그대로 유지했습니다. 4개 패치 모두 순수 추가이고 삭제는 1줄
(`advance()`의 `return {` → `state = {`)뿐입니다.

## 신호 — 새로 만들지 않았습니다

이미 계산되고 **버려지던** 값 셋을 씁니다.

| 신호 | 기존 처리 | 가중 |
|---|---|---|
| position 자기신고가 wall-clock 허용치 초과 | `advance()`의 `min()`이 조용히 클램프 | +3 |
| 동시접속 충돌 | `claim_session`이 409만 던지고 끝 | +5 |
| 체크포인트 연속 오답 상한 도달 | 되감기만 하고 끝 | +4 |
| 정상 전진 하트비트 | — | **-1 (감쇠)** |

상한 30. 임계 기본 10.

**감쇠가 왜 필요한가**: 없으면 장시간 시청에서 누적만 되어 결국 전원이 캡차를 봅니다.
오탐이 시간의 함수가 되어버립니다. 단 **전진이 있는 비트에서만** 감쇠합니다 —
일시정지까지 감쇠에 넣으면 재생을 멈춰두고 의심도를 씻어낼 수 있습니다.

## 3단 모드 — `off`로 시작합니다

```
off      기존과 100% 동일. 누적도 판정도 안 함                    ← 기본값
record   누적·판정하되 화면에 아무것도 띄우지 않음. 로그만
enforce  임계 초과 시 응답에 captcha_required 를 실어 보냄
```

`MAIN_CAPTCHA_URL` 또는 `MAIN_CAPTCHA_SITE_SECRET`이 비면 **자동으로 `off`로 강등**됩니다.
설정을 빼먹은 채 켜지는 상태를 막습니다.

**`record`를 권합니다.** 프런트 작업 없이 "지금 트래픽에서 몇 명이 걸리는가"를 관측해
임계값을 교정할 수 있습니다. 사용자 영향 0입니다. 아래 기본값 10은 **근거 없는 출발점**입니다.

## 컬럼 이름을 `bot_suspicion`으로 한 이유

0717(`lecture_pin_02`)에 드롭된 `suspicion`과 이름을 달리했습니다. 그건 체크포인트
**간격을 좁히는** 감시 장치였고 고정 핀 전환으로 쓸 곳이 없어졌습니다. 이건 **별개 캡차를
띄우는** 트리거이고 핀 예약에는 손대지 않습니다. 같은 이름을 재사용하면 드롭 이력과 섞여
읽는 사람이 어느 의미인지 알 수 없게 됩니다.

## fail-closed 지점

`POST /lectures/{lecture_id}/bot-check` — 프런트가 캡차 토큰을 제출하는 엔드포인트입니다.

토큰을 **캡차 서버에 서버-투-서버로 검증**하고(사이트 시크릿 필요) 확인된 것만 인정합니다.
브라우저가 "통과했다"고 말하는 것은 믿지 않습니다. 검증 실패는 **403**입니다 —
통과를 확인하지 못했으면 통과가 아닙니다.

(캡차 안에서 AI를 부르는 쪽은 fail-open입니다. "점수를 못 얻었다"와 "사람임을 확인했다"는
다른 문제이기 때문입니다.)

`claim_session`의 409 경로에서는 **명시적으로 커밋**합니다. 예외로 빠져나가면 호출자가
커밋하지 않아 신호가 롤백되어 사라집니다.

## 검증

```
$ python3 test_escalation.py
가산·상한 ✅  임계 판정 ✅  off 모드 무동작 ✅
설정 누락 시 off 강등 ✅  알 수 없는 모드 값 → off ✅  리셋 ✅
모두 통과
```

DB·FastAPI 없이 순수 로직만 확인합니다. `python3 -m py_compile`로 4개 파일 문법도 확인했습니다.

**검증하지 못한 것**: 실제 하트비트 경로에서의 동작. 백엔드를 띄울 권한이 없어
`advance()` 안에서의 실측은 못 했습니다. 적용 후 `record` 모드 로그로 확인이 필요합니다.

## 프런트 (나중, 소규모)

`LecturePlayer`가 이미 하트비트를 호출하므로 **응답 처리에 분기 하나**만 추가하면 됩니다.
새 화면은 필요 없습니다.

```javascript
if (res.captcha_required) {
  window.catchapOnVerified = (token) =>
    post(`/lectures/${lectureId}/bot-check`, { captcha_token: token });
  loadScript(CAPTCHA_HOST + "/widget.js", { "data-lecture-id": res.captcha_lecture_id });
}
```

`widget.js`(김민서님 작성)가 iframe 생성·origin 확인·postMessage 수신을 다 처리합니다.

## 되돌리기

`BOT_ESCALATION_MODE=off` — 코드 경로가 전부 죽습니다. 컬럼은 남아도 무해합니다.

## 확인 부탁드릴 것

1. 컬럼 이름 `bot_suspicion` 괜찮은지 (0717 결정과의 관계는 위 참조)
2. 가중치·임계값 — `record` 관측 후 정하는 게 맞다고 봅니다
3. `claim_session` 409 경로의 명시적 커밋 — 트랜잭션 규약상 괜찮은지
4. 승급을 어디까지 걸지 (시청 중 / 완료 직전 / 이수증 발급 전)
