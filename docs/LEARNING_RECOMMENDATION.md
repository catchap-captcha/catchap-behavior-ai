# 취약문제 추천 (Weak-Problem Recommendation) — 구조 & 설명

> 대화에서 정리한 설계를 규칙 기반 코드로 구현한 것입니다. **ML도 데이터도 없이
> 문제 하나만 풀려도 작동**하며, 데이터가 쌓이면 그 위에 통계·ML(XGBoost →
> BKT/IRT → DKT)을 얹어 발전시킵니다. 위치: `ai-service/learning/`.

---

## 0. 큰 그림 — 캡차 = 학습 문제

catchap은 **초등 저학년용 덧셈 문제를 드래그로 푸는 CAPTCHA**입니다. 아이가
"3+2=?"에서 숫자 타일(예: 5)을 답 슬롯으로 끌어다 놓으면, 그 드래그 한 번이:

1. **캡차 통과(사람 인증)** — 봇/매크로 차단
2. **학습 기록** — 이 개념을 아는지 → 취약점 분석 → 문제 추천

즉 **수집은 한 번, 용도는 둘**입니다. 드래그에서 원본 행동데이터를 한 벌만 저장하고,
Feature만 용도별로 뽑습니다:

- **봇탐지 = HOW(어떻게 끌었나)**: 속도·곡률·떨림·타이밍 → `ai-service`의 Human/Bot 모델
- **추천 = WHAT(무엇을 골랐나)**: 고른 답·정답 여부·개념·시간·조작 실수 → **이 패키지**

---

## 1. 파이프라인

```
원본 드래그 시도(RawAttempt)
   │  operation_error : 조작실수 vs 개념오답 판정 + 재시도 확정
   ▼
개념 레벨 결과(CountedOutcome)        ← 조작실수는 여기서 제외됨
   │  mastery : 찍기·소표본 보정 숙련도
   ▼
개념별 숙련도(MasteryResult)
   │  weakness : 취약도 계산 + Top-K
   ▼
취약 개념(WeakConcept)
   │  recommender : 숙련도에 맞는 난이도 문제 선택
   ▼
추천 문제(Recommendation)
```

닫힌 루프(설계 6단계)는 **추천 문제를 푼 뒤 `diagnose()`를 다시 호출**하면 성립합니다.
숙련도가 더 큰 시도 집합으로 재계산되므로, 추천의 `mastery_before`와 새 숙련도를
비교하면 전/후 변화가 나옵니다.

---

## 2. 모듈 구조 (`learning/`)

| 파일 | 역할 |
|------|------|
| `config.py` | 모든 튜닝 상수 (가중치·임계값·보정값). 시작 기본값이며 데이터로 조정 |
| `models.py` | 데이터 모델 (`RawAttempt`, `Question`, `Judgment`, `CountedOutcome`, `MasteryResult`, `WeakConcept`, `Recommendation`, `DiagnoseResult`) + `Outcome` enum |
| `operation_error.py` | 조작실수 판정 + 재시도 확정 (`classify_attempt`, `resolve_presentation`, `resolve_all`) |
| `mastery.py` | 찍기·소표본 보정 숙련도 (`concept_mastery`, `mastery_by_concept`) |
| `weakness.py` | 취약도 공식 + Top-K 선정 (`weakness_for_concept`, `top_weak_concepts`) |
| `recommender.py` | 난이도 매칭 문제 추천 (`recommend`, `difficulty_band`, `target_band`) |
| `service.py` | 전체 오케스트레이션 (`diagnose`) — **공개 진입점** |

순수 파이썬(표준 라이브러리)만 사용 → DB·프레임워크 없이 단독 테스트 가능.
테스트: `tests/test_learning.py` (19개).

---

## 3. 조작실수 판정 (operation_error)

아이는 **라벨된 타일**을 끌기 때문에, **집은 타일 = 의도한 답**을 알 수 있습니다.
여기에 놓은 위치를 합치면 대부분 규칙으로 갈립니다.

### 1차 판정 (진리표)

| 상황 | 판정 | 숙련도 |
|------|------|:---:|
| 시스템 오류 | `SYSTEM_ERROR` | 제외 |
| pointercancel / 아무것도 안 잡음 / 유효영역 밖 드롭(null) | `OPERATION_ERROR` | 제외 |
| 정답 타일 → 슬롯에 정상 드롭 | `CORRECT` | 반영(정답) |
| 오답 타일 → 슬롯에 정상 드롭 | `CONCEPT_ERROR` | 반영(오답) |
| 그 외(엉뚱한 위치) | `AMBIGUOUS` | 보류·재시도 |

### 재시도 확정 (게이밍 방지) ⭐

- 조작실수/시스템/애매 시도는 **완전 제외**
- 한 문제 제시당 **첫 개념 레벨(정답·개념오답) 결과 1개만** 반영
- 최대 재시도 **2회**(`RETRY_MAX`); 초과 시 그 문제는 미반영
- 결과:
  - "실수로 실패 → 다시 정답" = **정답 반영** (첫 개념레벨=정답)
  - "틀림 → 다시 정답" = **오답 반영** (첫 개념레벨=오답, 재시도 정답 무시)
  → 재시도로 숙련도를 부풀릴 수 없음

`operation_error_probability()`는 `regrab/failed_drop/cancel/final_drop_error`로 만든
보조 신호이며, **애매할 때만** 참고합니다(재시도 증거가 우선).

---

## 4. 숙련도 계산 (mastery)

두 보정을 겹칩니다 (서로 다른 문제를 해결):

```
1) 소표본 보정 :  observed = (정답 + 2) / (유효풀이 + 4)
2) 찍기 보정   :  mastery  = (observed − guess) / (1 − guess),  0 미만은 0
```

- `guess`(찍기 기준선) = 문제별 `1 / 선택지수`의 평균.
  2지선다=0.5, 3지선다≈0.33, 4지선다=0.25, **주관식=0**.
- 유효 풀이 < 3개면 `diagnostic_needed=True` (취약으로 확정하지 않음).

### 예시

- 3지선다 정답률 50% → `(0.5 − 0.33) / 0.67 ≈ 25%` (찍기 감안하면 낮음)
- 2지선다 vs 4지선다 같은 50% → 2지선다 숙련도가 더 낮게 나옴 (찍기 확률이 높으니까)

> 나중에 BKT의 guess 파라미터 / IRT의 추측 모수가 이 규칙을 정밀하게 대체합니다.

---

## 5. 취약도 (weakness)

```
weakness = 0.45·(1 − mastery)      # 못하는 정도 (주)
         + 0.30·recent_wrong        # 최근 5문제 오답 비율
         + 0.15·review_urgency      # 마지막 학습 후 경과 (min(일수/30, 1))
         + 0.10·hard_fail           # 난이도≥0.7 문제 실패 비율
```

- 각 항목 0~1 정규화, 가중치 합=1.
- **"복습 필요(경과시간)"를 작은 가중치로 분리** → 잘 아는데 오래된 개념을 "취약"과
  혼동하지 않음. `reason`은 가장 기여가 큰 항목으로 자동 생성.
- Top-K(기본 3) 선정. 유효 풀이 < 3개 개념은 순위에서 빼고 **"진단 필요"**로 반환.

---

## 6. 문제 추천 (recommender)

학생 숙련도에 맞는 난이도(근접 발달 영역)를 고릅니다:

```
숙련도 < 40%   → easy
40 ~ 70%       → medium
≥ 70%          → hard (또는 다음 개념)
```

추가 규칙:
- 최근에 푼 문제 제외 (`recently_solved`)
- 목표 밴드를 벗어난 문제는 뒤로 (한 번에 hard로 점프 안 함)
- **연속으로 hard만 주지 않음**
- 한 세트 3~5문제 (`DEFAULT_N_RECOMMEND`)
- `next_concept_ready()`: 숙련도 ≥ 85%면 다음 개념으로

---

## 7. 사용법

```python
from datetime import datetime
from learning import diagnose, RawAttempt, Question

result = diagnose(
    student_id="stu_001",
    attempts=[...],          # list[RawAttempt] — 이 학생의 모든 원본 시도
    question_bank=[...],     # list[Question]   — 추천 후보 문제
    now=datetime.utcnow(),
)
result.mastery            # {concept_id: MasteryResult}
result.weak_concepts      # [WeakConcept] Top-K
result.diagnostic_needed  # [concept_id] 데이터 부족
result.recommendations    # [Recommendation]
```

`diagnose()`는 DB·ML 없이 즉시 동작합니다. 실제 저장(learning_* 테이블)은 CAPTCHA/학습
백엔드에서 `RawAttempt`를 채워 넘기는 방식으로 연결하면 됩니다.

---

## 8. 단계별 발전 로드맵

| 단계 | 지금 (데이터 적음) | 추후 (데이터 축적) |
|------|-------------------|-------------------|
| 진단 | **규칙 기반 보정 정답률** (이 패키지) | BKT·IRT로 숙련 확률 추정 |
| 취약 | 취약도 공식 Top-3 | 미래 오답 확률 예측 |
| 추천 | 난이도 밴드 필터 | XGBoost 정답확률 / DKT 맞춤 |
| 조작실수 | 규칙 + 재시도 | 라벨 쌓이면 별도 판정 모델 |
| 찍기 | `1/선택지수` 보정 | BKT guess / IRT 추측모수 |

DKT는 순차 데이터가 많이 필요하므로 **맨 마지막**. 어떤 단계든 **조작실수 시도는
학습 시퀀스에서 제외**합니다.

---

## 9. 확정된 것 / 아직 열린 것

**확정 (코드에 반영됨)**
- 조작실수 진리표 + 재시도 확정(게이밍 방지)
- 소표본 보정 `(정답+2)/(풀이+4)`
- 찍기 보정 `(관측−1/선택지수)/(1−1/선택지수)`
- 취약도 가중치 `0.45 / 0.30 / 0.15 / 0.10`
- 난이도 밴드 `0.4 / 0.7`, 추천 3~5개

**아직 열림 (기본값으로 시작, 데이터로 튜닝)**
- 취약도 가중치·재시도 5초 창·재시도 2회 한도 등 모든 상수
- 선택지 개수·난이도 부여 주체(콘텐츠 팀), 개념 태그·선행관계
- 한 문제=다개념일 때 책임 분배(현재는 문제=단일 개념 가정)
- 추천 "성공"의 정의(효과 측정 지표)

---

## 10. 전제 (다른 팀/콘텐츠 필요)

- **문제 메타데이터**: `difficulty`(있음), `answer_options_count`, `concept_id`,
  `correct_answer_id` — 콘텐츠에 준비돼야 함
- **개념 태그 + 선행관계** 그래프 — 콘텐츠/운영 작업
- **learning_* MySQL 테이블** — DB 팀 (원본 시도·개념·문제·숙련도·추천·추천결과)
- **어린이 데이터**: 익명 ID·`age_group`·`consent_version`, 보호자 동의
