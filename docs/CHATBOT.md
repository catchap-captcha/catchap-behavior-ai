# 챗봇 (LLM) — 구조 & 설명

> LLM(Claude)로 대화하는 챗봇. **대화 능력은 데이터 없이도 지금 작동**하고,
> 개인화("이 개념이 부족해요")는 학습 진단 데이터가 있을 때 강해집니다.
> 위치: `ai-service/chat/`.

## 0. 핵심 원리

- **대화 = LLM(Claude)이 담당** → 규칙기반 아님, 지금도 진짜 AI로 작동
- **개인화 = 우리 데이터 주입** → `learning.diagnose()` 결과를 프롬프트에 넣음(직접 주입, RAG 아님)
- **얼굴 둘, 엔진 하나** → 냥냥이(아이)/상담사(학부모)는 시스템 프롬프트만 다름

## 1. 구조 (`chat/`)

| 파일 | 역할 | 공유? |
|------|------|:---:|
| `engine.py` | Claude 호출 (유일한 API 호출 지점) | 🟢 공유 |
| `context.py` | `DiagnoseResult` → "학습 상태" 사실 블록 | 🟢 공유 |
| `prompts.py` | 역할별 시스템 프롬프트(냥냥이/상담사) | 🔴 분리 |
| `service.py` | 오케스트레이션 (`reply` / `kid_reply` / `parent_reply`) | 🟢 공유 |
| `config.py` | 모델·상수 (`DEFAULT_MODEL` 등) | — |

## 2. 흐름

```
reply(role, user_message, history, learning_result)
  ① context.py  : learning_result → "학습 상태" 텍스트 (없으면 일반 모드)
  ② prompts.py  : 역할 페르소나 + 학습 상태 = system
  ③ engine.py   : Claude 호출 (system + history + user_message)
  ④ 반환        : 답변 텍스트
```
LLM은 기억이 없으니 `history`(이전 대화)를 매번 함께 보냅니다.

## 3. 두 역할

| | 냥냥이 (kid) | 상담사 (parent) |
|---|---|---|
| 말투 | 반말·이모지·격려 | 존댓말·정확 |
| 데이터 노출 | 수치 숨김, 응원 위주 | 숙련도·이유 자세히 |
| 정답 | 직접 안 주고 힌트 | (해당 없음) |
| 안전 | 학습 유지·주제 이탈 차단·보호자 안내 | 심리 진단 금지 |

## 4. 사용법

```python
from chat import kid_reply, parent_reply
from learning import diagnose

result = diagnose(student_id, attempts, question_bank, now)  # 학습 진단(선택)

# 아이용
kid_reply("그림 찾기가 어려워요", history=prev, learning_result=result)
# 학부모용
parent_reply("이번 주 우리 아이 어땠나요?", history=prev, learning_result=result)
```

- `learning_result=None`이면(데이터 없음) → 개인화 없이 일반적으로 도와줌
- 실제 호출엔 `ANTHROPIC_API_KEY`(.env) 필요. 없으면 코드·테스트는 되지만 실호출은 안 됨.

## 5. 지금 있는 것 / 남은 것

- ✅ 챗봇 로직(엔진·프롬프트·컨텍스트·서비스), 테스트 8개 (키 없이 통과)
- ✅ `learning.diagnose()` 연동
- ❌ **API 엔드포인트** (`POST /chat/kid`·`/chat/parent`) — 어디 둘지(ai-service vs catchap-backend) 결정 후 추가
- ❌ **대화 저장(history)** — DB 테이블 (기억 유지)
- ❌ **실제 API 키** — 발급 후 `.env`
- 🔵 채팅 UI — 프론트팀 (목업 있음)

## 6. 발전 방향

- **비용 최적화**: 아이용 `claude-haiku-4-5`, 학부모용 `claude-sonnet-5`로 모델 분리 가능(`config.DEFAULT_MODEL` 또는 `ChatEngine(model=...)`); 프롬프트 캐싱
- **스트리밍 응답**: 글자 단위 실시간 표시 (UX)
- **RAG**: "가정 학습 팁" 같은 문서가 많아지면 `context.py`에 검색 결과도 합쳐 주입
- **안전 필터**: 아이 자유 입력용 추가 가드
