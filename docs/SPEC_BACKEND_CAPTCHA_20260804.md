# 백엔드 ↔ 캡차 연동 규약

조성원 · 2026-08-04 · 캡차 코드(`ai-service-ms-behavior`)에서 읽어낸 것

---

## 한 장 요약

```
①  백엔드가 사용자를 캡차 페이지로 보낸다     (session_id · lecture_id · purpose 를 실어서)
②  사용자가 캡차를 푼다                      (프론트 ↔ 캡차, 백엔드 관여 없음)
③  캡차가 captcha_token 을 프론트에 준다
④  프론트가 그 토큰을 백엔드로 보낸다
⑤  백엔드가 캡차 서버에 토큰을 검증한다        ← 여기가 백엔드가 할 일의 전부
```

**백엔드가 구현할 것은 ⑤ 하나다.** 호출 한 번이다.

---

## ⑤ 토큰 검증 — 서버 대 서버

```
POST  {CAPTCHA_BASE}/api/verify-token
Header  X-Captcha-Site-Secret: <CAPTCHA_SITE_SECRET>
```

요청 본문 (`VerifyTokenRequest`)

| 필드 | 타입 | 필수 | 비고 |
|---|---|---|---|
| `token` | str (32~256) | ✅ | 프론트가 받은 `captcha_token` |
| `session_id` | str (8~128) | ✅ | 발급 때와 **같아야** 한다 |
| `lecture_id` | str \| null | | 주면 발급 때 값과 **일치해야** 통과 |
| `purpose` | `signup` \| `login` \| `recovery` \| `lecture` | | 기본 `lecture` |

응답

```json
성공   {"success": true,  "lecture_id": "...", "challenge_id": "..."}
실패   {"success": false, "error": "invalid_or_used_token"}
```

**시크릿이 틀리면 401 이다.** 실패는 `success:false` 로 오지 예외로 오지 않으므로,
`success` 를 반드시 확인해야 한다.

### 반드시 지킬 것 세 가지

**① 토큰은 1회용이다.** `verify_token` 이 검증과 동시에 소비한다(`FOR UPDATE` +
`consumed_at`). **두 번 호출하면 두 번째는 실패한다.** 재시도 로직이 같은 토큰을
다시 보내면 정상 사용자가 막힌다.

**② `session_id` 가 발급 때와 같아야 한다.** 다르면 토큰이 유효해도 실패한다.
백엔드가 세션을 재발급하는 경로가 있으면 그 사이에 토큰이 죽는다.

**③ `lecture_id` 를 보낼 거면 발급 때도 보내야 한다.** 발급 때 없이 검증 때만 주면
불일치로 실패한다.

---

## ① 캡차로 보내기

```
{CAPTCHA_BASE}/?participant=<코드>          수집용
{CAPTCHA_BASE}/                             일반
```

챌린지 생성은 프론트가 한다. 백엔드는 관여하지 않는다.

```
POST /api/captcha/challenges
Header  X-Captcha-Site-Key: <CAPTCHA_SITE_KEY>
```

**site key 는 공개값**이고 프론트에 들어간다. **site secret 은 백엔드에만** 둔다.
`/api/config` 가 site key 를 그대로 내려주므로 그 둘을 헷갈리면 안 된다.

---

## 백엔드에 필요한 설정

```
MAIN_CAPTCHA_URL           캡차 베이스 URL
MAIN_CAPTCHA_SITE_SECRET   CAPTCHA_SITE_SECRET 과 같은 값
BOT_ESCALATION_MODE        record → enforce   (record 는 캡차를 부르지 않는다)
```

7/30 에 하지영님이 확인해 주신 대로 `record` 에서는 캡차 설정이 필요 없고, `enforce`
로 올릴 때 받으면 된다.

---

## 확인된 것 / 확인 못 한 것

**확인됨** — 위 규약은 `app/main.py:926` 과 `app/db.py:595` 를 직접 읽어 정리했다.
토큰 1회 소비, `session_id`·`lecture_id` 일치 요구, 시크릿 헤더 이름 전부 코드 기준이다.

**확인 못 함** — 실제 호출은 해보지 않았다. 캡차 `.env` 의 `CAPTCHA_SITE_SECRET` 을
읽을 권한이 없다. 민서가 값을 알려주면 서버-서버 왕복을 한 번 돌려 확인할 수 있다.

---

## 순서 제안

연동 자체는 호출 하나지만, **켜는 순서**가 중요하다.

```
1  MAIN_CAPTCHA_URL · SECRET 설정만 하고 enforce 는 아직 끈다
2  캡차 위젯이 플랫폼에 실제로 뜨는지 확인          ← 지금 #captcha-mount 가 비어 있다
3  토큰 검증 왕복을 한 번 성공시킨다 (수동)
4  enforce 로 올린다
```

**2 번이 아직 안 됐다.** `www.catchap5.com/captcha` 를 열어보면 "API 캡챠 위젯 자리"
라는 자리표시자 문구가 그대로 떠 있고, 드래그 캡차 요소도 캡차 서버 호출도 없다.
백엔드가 `enforce` 로 올려도 **사용자에게는 빈 화면이 보인다.**

---

## 그리고 하나 — 지금 켜면 오탐이 사용자에게 간다

행동 모델은 `shadow` 다. 캡차 자체는 문항 풀이·허니팟·PoW 로 판정하므로 켜도 된다.
다만 **백엔드 의심도 가중치가 아직 근거 없는 값**이라는 문제는 그대로다(D-2, 관측
트래픽 0). `enforce` 로 올리기 전에 가중치를 정하는 게 순서다.

---
문의: 조성원 (wwdhogo@gmail.com)
