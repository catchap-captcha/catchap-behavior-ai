# 지금 수집 가능한가 — 실측 점검

조성원 · 2026-08-06

---

## 결론

**됩니다. 오늘 시작할 수 있습니다.**

```
https://captcha.catchap5.com/?participant=<참여자코드>
```

이 링크로 들어가서 풀면 참여자 코드·원시 궤적·스테이지 크기가 전부 저장됩니다.
**위젯 부착을 기다릴 필요가 없습니다.**

다만 **위젯 경로(iframe)로는 아직 안 됩니다.** widget.js 가 참여자 코드를
iframe 주소에 안 실어 보냅니다. 민서에게 한 줄 요청했습니다.

---

## 앞선 판단 정정

이 문서 초판에 "참여자를 저장할 컬럼이 없어서 수집 불가"라고 적었습니다.
**틀렸습니다.** ORM 이 파이썬 속성명을 다른 DB 컬럼명에 매핑하고 있는데, 제가
속성명으로 SQL 을 던져서 "컬럼 없음" 이 나온 것이었습니다.

```python
# app/database/mysql_models.py
anonymous_participant_id  →  "participant_id"
captcha_width             →  "viewport_width"
captcha_height            →  "viewport_height"
presented_at              →  "started_at"
```

7/28 에 적어둔 "프로덕션 DB 는 읽기·쓰기 둘 다 실패한다"도 낡은 정보입니다.

---

## 1. 연동은 이미 돌고 있다

```
/health
{"status":"ok", "mysql_connected":true, "model_loaded":true,
 "model_version":"revalidation_two_view_participant_safe_20260722",
 "feature_schema_version":"2.3", "policy_mode":"shadow"}

로그
POST /api/v1/behavior/predict          685
POST /api/v1/behavior/shadow/outcomes  641
POST /api/v1/behavior/collect            0
저장 실패                                1
```

`/collect` 는 한 번도 안 불렸지만 **`/predict` 가 원시 데이터를 통째로 저장**합니다.

```python
repo.save_attempt_bundle(attempt={...}, events=events, interaction=...)
repo.save_features(...)
repo.save_security_features(...)
```

---

## 2. 실제로 쌓인 것

```
ai_behavior_attempts    619건
ai_pointer_events    19,388건
ai_attempt_features     619건

품질   valid 603 · pending 9 · invalid 7
기간   7/31 ~ 8/05
```

### 참여자 라벨이 살아 있다

```
sw-mouse           103
sw-mouse-v2         95
데모-mouse           63
integration-test    14
pwbot-*             ...
(없음)                7
```

### 스테이지 크기도 남고 있다

```
500×375   159
500×333   139
640×360    42
375×500    36
333×500    28
500×500    11
```

**화면마다 크기가 다릅니다.** 이 값이 남기 때문에 나중에 정규화로 차이를 없앨 수
있습니다. 7/31 에 옛 수집 화면 0.11% 가 메인 캡차에서 33.3% 로 튄 것이 이 차이였습니다.

---

## 3. 남은 것 하나 — 위젯이 참여자 코드를 안 넘긴다

캡차 서버는 받을 준비가 돼 있습니다.

```python
# ai-service-ms-behavior/app/main.py:93, 806
participant_id: str | None = Field(default=None, max_length=64)
anonymous_participant_id=payload.participant_id
```

프론트도 읽습니다.

```js
he.get("participant")  →  sessionStorage["catchap-participant"]  →  participant_id
```

**그런데 위젯이 만드는 iframe 주소에 없습니다.**

```js
var params = new URLSearchParams({ embed: "1", lecture: lecture });
```

sessionStorage 폴백이 있지만 **탭 단위**라 수집 절차로 쓸 수 없습니다.
민서에게 `data-participant` 전달을 요청했습니다. 한 줄입니다.

---

## 4. 세 화면에서 수집하기 — 어디까지 되나

수집처를 세 곳으로 늘리고 싶다는 요구가 있었습니다.

```
민서 드래그 캡차     드래그 궤적 있음        ← 지금 모델의 원천
LLM 캡차            radio / checkbox        ← 클릭
문제은행 문제 풀이    radio / checkbox        ← 클릭
```

플랫폼 번들을 열어보니 `pointerdown`/`dragstart` 가 2개뿐이고(하나는 비밀번호 보기
버튼) 문항 UI 는 `radio`·`checkbox` 입니다. **문제 푸는 화면에는 드래그가 없습니다.**

### 그래서 문제 풀이 행동 자체는 지금 모델로 못 씁니다

특징 스키마 `trajectory-only 2.3` 은 드래그 궤적에서 뽑습니다.

```python
MIN_MOVES_PER_DRAG = 2      # 누름과 뗌 사이에 이동 2회 이상
```

클릭은 그 사이 이동이 0~1 회라 전부 걸러지고, 결과는 이렇게 됩니다.

```python
return {"human_score": 0.0, ..., "reason": "every_drag_below_move_floor"}
```

**사람이든 봇이든 전부 0.0 이 나옵니다.** 학습에도 채점에도 못 씁니다.

### 할 수 있는 것 — 세 화면에 드래그 캡차를 띄운다

문제 푸는 행동을 재는 게 아니라, **그 화면에서 캡차를 한 번 풀게** 하는 것입니다.
같은 위젯, 같은 파이프라인입니다. 수집 기회가 3배가 됩니다.

**화면 구분은 참여자 코드 접미사로 합니다.** 스키마를 바꿀 필요가 없습니다.

```
sw-mouse-player      시청 화면
sw-mouse-bank        문제은행
sw-trackpad-llm      LLM 캡차
```

`person_of()` 가 `-` 앞만 잘라 쓰므로 사람 단위 분리는 그대로 동작합니다.

### 문제 풀이 행동 모델은 새 과제다

클릭·스크롤·타이핑으로 사람을 가리는 건 다른 모델입니다. 새 특징 스키마 + 새 학습
데이터 + 새 검증이 필요합니다. **발표까지 13일**이고 지금 모델도 승급 전이라,
지금 시작하면 둘 다 못 끝냅니다. 나중 과제로 둡니다.

---

## 5. 지금 할 일

```
지금 바로   직접 링크로 수집 시작 — 위젯을 안 기다린다
            https://captcha.catchap5.com/?participant=sw-mouse

민서        widget.js 에 data-participant 전달 (한 줄)
지영        세 화면에 위젯 부착 + 참여자 코드 전달
```

DB 변경은 **필요 없습니다.** 컬럼이 이미 다 있고 `metadata` JSON 도 있습니다.
프로덕션 스키마는 건드리지 않습니다.

### 필요 수량

```
8명 × 2개 입력장치(마우스·트랙패드) × 40세션 = 640세션
```

`tools/collection_split.py` 의 분리 규칙(사람 단위, 소금값 고정)은 이미 박아뒀습니다.

---

## 6. 정리

```
어제까지 생각한 병목    위젯이 안 붙어서 수집을 못 한다
실제                   직접 링크로 오늘부터 가능
                       위젯 경로만 민서 한 줄이 남았다
```

---
문의: 조성원 (wwdhogo@gmail.com)
