# 궤적 밖 신호 도입 계획

조성원 · 2026-07-31

---

## 왜 필요한가 — 오늘 데이터가 근거다

사람 166건 / 봇 298건으로 동작점을 훑은 결과, **어느 임계값에서도 두 기준을 동시에
만족하지 못한다.**

| threshold | 사람 FRR | 봇 차단 |
|---:|---:|---:|
| 0.99995 (현재) | 22.2% | 80.5% |
| 0.20 | 2.5% | 51.0% |
| 0.05 | 1.2% | 48.3% |

분포가 겹쳐 있기 때문이다.

```
사람  최소 0.065 · 5% 0.943 · 중앙 1.000000
봇    중앙 0.123 · 75% 0.9999 · 최대 1.000000
```

**봇 상위 25%가 사람 구간에 완전히 들어와 있다.** 임계값 조정은 맞바꿈이지
개선이 아니고, 같은 특징으로 표본을 늘려도 겹침은 그대로다.

근본 원인은 하나다. **`(x, y, 시각)`은 전부 공격자가 원하는 대로 만들 수 있는
값이다.** 거기서 계산한 특징은 더 좋은 생성기가 나오면 원리적으로 따라잡힌다.
오늘 자로 그은 직선이 `human_probability 1.0000` 을 받은 것이 그 실증이다.

그래서 **공격자가 싸게 만들 수 없는 값**이 필요하다.

---

## 무엇을 수집하나

7/17 계획서(`docs/행동데이터앞으로할것.md` §4.2)가 이미 규정해 둔 것이 절반이다.
거기에 두 개를 더한다.

### 1군 — 계획서에 있던 것 (미구현)

| 필드 | 무엇을 잡나 |
|---|---|
| `pointer_type` | mouse / pen / touch. 장치와 궤적의 정합성 |
| `pressure` | 마우스는 눌린 동안 0.5 고정, 터치는 변동. 주입은 기본값 |
| `width` · `height` | 접촉 면적. 터치는 변하고 마우스는 1×1 |
| `buttons` | 눌린 버튼 비트마스크. 드래그 중 일관성 |
| `is_primary` · `pointer_id` | 멀티터치 구분 |

### 2군 — 새로 제안 (여기가 핵심)

| 필드 | 왜 강한가 |
|---|---|
| **`isTrusted`** | JS 로 만든 합성 이벤트는 `false`. 한 필드로 순진한 주입을 거른다 |
| **`getCoalescedEvents().length`** | 진짜 포인터는 프레임 사이 원시 샘플을 여러 개 들고 있다. **실제 입력 장치가 실제 브라우저를 움직여야만 생기는 값** |
| `event.timeStamp` | 브라우저가 찍는 고해상도 시각. `Date.now()` 와 별개로 대조 가능 |

**`getCoalescedEvents()` 가 이 계획의 중심이다.**

```
실제 마우스   1000Hz 입력 / 60Hz 화면  →  pointermove 하나에 원시 샘플 여러 개
CDP 주입      dispatchMouseEvent 1회   →  항상 1개
HTTP 스크립트 이벤트를 만들지도 않음    →  값 자체가 없음
```

시각 인식은 이제 싸다. 하지만 **실제 브라우저를 띄우고 실제 입력 장치 수준의
이벤트를 만드는 것은 여전히 비싸다.** 그 비용을 강제하는 층이 지금 하나도 없다.

### 개인정보

전부 상호작용 특성이고 식별자가 아니다. 이름·이메일·계정은 여전히 안 받는다.
`pointer_type` 이 장치 종류를 드러내지만 그건 지금도 `viewport` 로 대략 드러난다.

---

## 어디를 고치나 — 4개 계층

### ① 프론트 (김민서)

`record()` 가 `PointerEvent` 를 이미 받고 있는데 `clientX/Y` 만 쓰고 버린다.

```javascript
const record = (type, objectId, event) => {
  // ...기존 좌표 계산...
  pendingEventsRef.current.push({
    seq, type, object_id, x, y, timestamp_ms: now,
    // 궤적 밖 신호. 브라우저가 안 주면 null 로 둔다 —
    // 없는 값을 0 으로 채우면 실제 0 과 결측이 섞인다(7/17 계획서 4.2 주의사항).
    pointer_type: event?.pointerType ?? null,
    pressure: event?.pressure ?? null,
    width: event?.width ?? null,
    height: event?.height ?? null,
    buttons: event?.buttons ?? null,
    is_primary: event?.isPrimary ?? null,
    is_trusted: event?.isTrusted ?? null,
    coalesced: event?.getCoalescedEvents?.().length ?? null,
    event_ts: event?.timeStamp ?? null,
  });
};
```

`?.` 와 `?? null` 로 감싸는 이유: 구형 브라우저나 `getCoalescedEvents` 미지원
환경에서 예외로 캡차 전체가 죽으면 안 된다. **없으면 null 이지 실패가 아니다.**

### ② 캡차 서버 (김민서)

`BehaviorBatchEvent` 에 필드를 열고, `build_predict_payload` 가 그대로 넘긴다.
전부 `| None = None` 이라 프론트가 안 보내도 기존 동작 그대로다.

### ③ AI 수신 (조성원)

`PointerEventIn` 이 `extra="forbid"` 라 명시적으로 열어야 한다.

```python
class PointerEventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seq: int; event_type: EventType; t_ms: int; x: float; y: float
    # 신규 — 전부 nullable. 결측과 0 을 구분한다
    pointer_type: str | None = None
    pressure: float | None = None
    ...
```

**저장 쪽은 이미 준비돼 있다.** `ai_pointer_events` 에
`pointer_type`·`pressure`·`buttons_mask`·`movement_x/y` 컬럼이 이미 있고 지금은
전부 NULL 이다. 마이그레이션이 필요한 건 `coalesced`·`is_trusted` 정도다.

### ④ 특징·재학습 (조성원)

`trajectory-only 2.3` → 장치 뷰를 더한 새 스키마.

```
coalesced_mean / coalesced_zero_ratio    ← 가장 기대되는 축
is_trusted_ratio
pressure_variance                         터치는 변동, 주입은 고정
pointer_type_consistency                  선언한 장치와 궤적이 맞는가
```

---

## ⚠️ 검증을 잘못하면 스스로를 속인다

**지금 내 레드팀 도구로는 이 개선을 검증할 수 없다.** HTTP 로 직접 이벤트를
만들기 때문에 `coalesced`·`isTrusted` 가 아예 없거나 null 이다. 그러면 새 특징이
100% 분리해내고, 우리는 "해결됐다" 고 착각한다.

**제대로 된 적수는 실제 브라우저를 띄우는 봇이다.** 다행히 이미 있다.

```
ai-service-ms-behavior/tools/capture_playwright_bots.mjs
data/interim/playwright_bots_300 · playwright_ease_burst_*
```

검증 절차:

```
1. 사람          실제 브라우저 (수집 세션)
2. HTTP 스크립트 봇   내 도구 — 하한 확인용
3. Playwright 봇     ← 진짜 적수. 이게 갈라지는지가 전부다
```

**3번이 안 갈라지면 이 계획은 실패다.** 그 경우 남는 건 문항 품질과 비용
(PoW·레이트리밋)이고, 행동 모델은 보조 필터로 고정된다. 그것도 정직한 결론이다.

---

## 못 막는 것 — 미리 적는다

| | |
|---|---|
| 사람 대행 | 진짜 사람이 푼다. 이 계층으로는 원리적으로 불가 |
| 실제 입력 장치 주입 | 하드웨어 수준 자동화. 비용은 크지만 가능 |
| 오래된 브라우저 사용자 | `getCoalescedEvents` 미지원 → 결측. **결측을 봇 신호로 쓰면 안 된다** |

마지막 항목이 중요하다. 결측률이 높은 브라우저군이 통째로 오탐되면 지금보다
나빠진다. **결측은 별도 범주로 두고, 있는 경우에만 판별에 쓴다.**

---

## 순서와 분담

| | 작업 | 담당 | 비용 |
|---|---|---|---|
| 1 | 프론트 필드 수집 | 김민서 | 몇 줄 |
| 2 | 캡차 스키마 + 전달 | 김민서 | 몇 줄 |
| 3 | AI 수신·저장 | 조성원 | 반나절 |
| 4 | Playwright 봇 세트 확보 | 조성원 | 반나절 |
| 5 | 수집 (사람 + Playwright 봇) | 팀 | 1시간 |
| 6 | 새 특징 분리력 확인 | 조성원 | 반나절 |
| 7 | 갈라지면 재학습 · 아니면 보조 필터로 고정 | 조성원 | 1~2일 |

**1·2 가 없으면 나머지가 전부 막힌다.** 프론트 몇 줄이 이 계획의 임계 경로다.

발표(8/19)까지 3주면 6번까지는 확실히 되고, 7번은 6번 결과에 달렸다.

---

## 이것과 별개로 병행할 것

행동 모델이 어떻게 되든 **인증의 바닥은 문항 품질**이다.

7/29 집계로 보기 2개 문항이 438개였다(재확인 필요). 찍어도 50% 통과다.
시각 인식 봇은 허니팟도 자연스럽게 피한다 — 빈 영역에 놓이므로 이미지를 보고
고르면 안 집는다. **허니팟은 JSON 을 긁는 봇만 잡는다.**

그래서 시각 봇에 대한 실질 방어는 문항 난이도와 이 계획의 장치 신호 둘뿐이다.

---
문의: 조성원 (wwdhogo@gmail.com)
