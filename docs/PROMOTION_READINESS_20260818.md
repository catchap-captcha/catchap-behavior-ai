# 모델 승격 준비 점검 — 2026-08-18

조성원 · 목적은 "차단을 끄고, 다양한 사람 데이터를 모아, 승격을 제대로 하는 것" 이다.
이 문서는 **그 준비가 됐는지**를 코드와 실서비스 응답으로 확인한 결과다.

## 0. 한 줄 요약

절차와 도구는 갖춰져 있었다. 그런데 **실서비스가 shadow 가 아니라 active 였고**,
그 때문에 승격 근거가 될 shadow 관측이 한 건도 쌓이지 않고 있었다.

## 1. 실서비스 상태 (`/health/ready`, 버전 8c7f37c)

```
behavior_policy_mode        active     ← 캡차 서비스
behavior_ai.policy_mode     active     ← AI 서비스
behavior_ai_policy_matches  true       ← 시행 조건 충족
behavior_event_transport    shadow
```

우리 예시 파일은 셋 다 다른 값을 적고 있다 — **배포값이 문서에서 갈라져 있다.**

```
ai-service/deploy/behavior-ai.env.example   RISK_POLICY_MODE=shadow
ai-service-ms-behavior/.env.example         BEHAVIOR_POLICY_MODE=shadow
```

### 이 상태가 무엇을 뜻하나

`resolve_final_verdict` 는 양쪽이 다 `active` 이고 추천 조치가 `step_up` 계열이면
**정답을 맞힌 시도를 `failed` 로 뒤집는다**(`app/behavior_client.py:444`).
지금 그 조건이 성립한다.

그리고 `record_shadow_outcome` 은 `policy_mode == "shadow"` 일 때만 호출된다
(`app/main.py:1082`). AI 쪽도 shadow 가 아니면 409 로 거부한다
(`app/api/shadow.py:33`). 즉 **`ai_shadow_outcomes` 는 비어 있을 것이다.**

`behavior_event_transport=shadow` 는 **보조 서버 신호**(텔레메트리 누락, stop/go,
배치 전달 타이밍)만 가린다. 궤적 모델의 판정은 가리지 않는다. 이름이 비슷해
헷갈리기 쉬운 두 스위치다.

## 2. 오늘 고친 것

### 2.1 봉인 홀드아웃 해시가 한 번도 대조되지 않고 있었다

`tools/lockbox_audit.py` 는 스스로 세 가지 질문에 답한다고 적어 두었고 그 셋째가
"디스크의 데이터가 봉인 당시 그대로인가(sha256)" 인데, 대조가 `lockbox_manifest`
키가 있을 때만 돌았다. 그 형식을 쓰는 것은 **이미 소진된 세 벌뿐**이고, 아직
봉인된 9벌은 자기 매니페스트의 `output.sha256` 에 해시를 적는 형식이다.

결과: 감사는 9벌 전부를 `해시미확인` 으로 내보내면서 `--strict` 에서 정상 종료했다.
위험은 "해시가 틀렸다" 가 아니라 **"틀렸는지 아무도 모른다"** 였다.

고친 뒤 다시 돌린 값:

```
봉인 유지 표시 9건 · 실제로 쓸 수 있는 것 9건 · 해시 검증된 것 9건
불일치 0 · 파일없음 0
```

**봉인 데이터 12벌 전부 봉인 당시 그대로다.** 변조 없음.

지키는 시험: `tests/test_lockbox_hash_verified.py` — 감사 보고서에 `해시미확인` 이
하나라도 남으면 실패한다.

### 2.2 승격 기준이 네 파일에 흩어져 있었다

```
tools/consolidate_formal_validation.py        FRR 3% · 아는 5% · 미지 10%
tools/evaluate_candidate_external_holdout.py  MAX_BOT_ASR 0.05 · MAX_HUMAN_FRR 0.03
tools/aim_lofo.py                             FRR_BUDGET 0.05 · 기준 ≤10%
docs/PROMOTION_STATUS_20260810.md             표에 손으로 적은 값
```

`training/promotion_gate.py` 로 모았다. **기준을 새로 정하지 않았다** — 기존 두
도구가 이미 같은 값을 쓰고 있었고 그 값을 그대로 옮겼다.

★핵심은 **재지 않은 항목을 불합격으로 떨어뜨리는 것**이다. `None` 을 통과로 처리하면
"안 재봤다" 가 보고서에 "문제 없다" 로 찍힌다. 승격에서 그 방향의 실수는 봇을
들여보낸 뒤에야 드러난다.

전제조건 둘을 더 건다.

- `sealed_holdout_verified` — 봉인 해시가 대조됐나. 아니면 점수가 근거가 못 된다.
- `shadow_outcomes_observed` — 실사용자 관측이 최소치를 넘었나. 오프라인 점수만으로
  켜면 실사용자 오탐을 한 번도 안 보고 켜는 셈이다.

지키는 시험: `tests/test_promotion_gate.py` (7개). 마지막 하나는 **기준 표류 감시**다 —
다른 도구의 상수가 갈라지면 거기서 걸린다.

전체 시험 263개 통과.

## 3. 남은 것 — 내가 못 하는 것

### 3.1 파드 환경값 변경 (인프라 담당자님)

내 손이 닿지 않는다(kubeconfig 없음). 요청문은 §5.

### 3.2 개인정보 처리 방침과 끄는 스위치 (팀 결정)

실사용자 궤적 수집을 늘리는 것이 목적인데 방침 고지와 거부 수단(backend#50)이
보류 상태다. **기술이 아니라 순서 문제이고, 내가 혼자 정할 일이 아니다.**
수집을 늘리기 전에 이것이 먼저 나가야 한다.

## 4. 권하는 순서

```
1. 개인정보 방침·끄는 스위치 배포        ← 수집의 전제. 팀 결정
2. 양쪽 policy_mode 를 shadow 로          ← 여기서부터 승격 근거가 쌓인다
3. 매일 /api/v1/admin/shadow/summary 확인
4. 2~4주 뒤 promotion_gate 로 심사
```

## 5. 인프라 담당자님께 보낼 요청문

> 안녕하세요, 조성원입니다.
>
> 행동 판별 층을 **shadow(관측만) 로 내리고 싶습니다.** 지금 실서비스가 active 라
> 궤적 판정이 실제 캡차 결과를 바꾸고 있는데, 아직 실사용자 오탐을 충분히 재지
> 못한 상태라 관측 모드로 두고 데이터를 모으는 편이 맞다고 판단했습니다.
>
> 두 파드의 환경값을 바꿔 주시면 됩니다. **둘 다** 바뀌어야 합니다.
>
> ```
> 캡차 서비스   BEHAVIOR_POLICY_MODE   active → shadow
> AI  서비스    RISK_POLICY_MODE       active → shadow
> ```
>
> `BEHAVIOR_EVENT_TRANSPORT` 는 지금의 `shadow` 그대로 두시면 됩니다 — 궤적 수집은
> 계속돼야 합니다.
>
> 한쪽만 바뀌면 두 서비스 정책이 어긋나 시행은 꺼지지만, "AI가 사용자 결과를 바꾸지
> 않았다" 를 증명하는 기록이 남지 않습니다. 그래서 둘 다 부탁드립니다.
>
> 반영 뒤 아래로 확인됩니다.
>
> ```
> curl -s https://captcha.catchap5.com/health/ready
> → behavior_policy_mode: "shadow", behavior_ai.policy_mode: "shadow"
> ```
