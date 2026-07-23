# CAPTCHA Shadow Mode 실행 절차

CAPTCHA UI가 완성된 뒤, AI 권고가 실제 사용자를 과도하게 방해하지 않는지 확인하는
운영 절차다. 이 단계에서는 **AI 권고로 통과·실패를 바꾸지 않는다.**

## 1. 사전 준비

1. DB 관리자가 `db/migrations/20260723_shadow_mode.sql`을 기존 AI DB에 한 번 적용한다.
2. shadow 환경의 `.env`에 다음을 설정한다.

```dotenv
RISK_POLICY_MODE=shadow
PRODUCTION_MODEL_DIR=models/candidate/revalidation_two_view_participant_safe_20260722
CAPTCHA_BACKEND_API_KEY=<backend-only-secret>
ADMIN_API_KEY=<admin-only-secret>
```

`PRODUCTION_MODEL_DIR`은 shadow 환경에서만 candidate bundle을 읽기 위한 경로다. 이 설정은
candidate를 production 승격하는 행위가 아니다.

3. 서비스 재시작 후 `GET /health`에서 다음을 확인한다.

```json
{
  "model_loaded": true,
  "policy_mode": "shadow"
}
```

## 2. CAPTCHA backend 처리 순서

1. 기존 방식으로 main CAPTCHA를 발급하고 답안을 검증한다.
2. 같은 시도의 pointer events를 포함해 backend가 `/api/v1/behavior/predict`를 호출한다.
3. 응답의 `policy_mode`가 `shadow`이면 `recommended_action`을 화면이나 최종 verdict에 적용하지 않는다.
4. main CAPTCHA verdict를 기존 방식 그대로 사용자에게 반환한다.
5. 처리 후 backend가 `/api/v1/behavior/shadow/outcomes`에 한 번 기록한다.

```json
{
  "attempt_id": "predict와 같은 attempt id",
  "main_captcha_verdict": "passed",
  "final_verdict": "passed"
}
```

`main_captcha_verdict`와 `final_verdict`가 다르면 API가 거부한다. 이는 shadow 중에 AI가
사용자 결과를 바꿨는지 즉시 발견하기 위한 보호 장치다.

## 3. 매일 확인할 지표

관리자 키로 `GET /api/v1/admin/shadow/summary`를 조회한다.

- 전체 shadow 결과 수
- `would_step_up_rate`: AI가 추가 인증을 권고했을 비율
- action별 main CAPTCHA 통과·실패 수와 통과율
- `step_up` 또는 `step_up_and_rate_limit`을 권고받은 사용자의 실제 main CAPTCHA 통과율

main CAPTCHA 오답은 봇 라벨이 아니다. 이 지표는 문제 난이도·사용성·AI 마찰을 함께
살피는 용도이며, shadow 결과만으로 모델을 재학습하거나 threshold를 조정하지 않는다.

## 4. 최소 통합 테스트

1. 자연스러운 정상 궤적: `/predict`의 `allow`, 일반 CAPTCHA 결과 유지, outcome 저장
2. 애매한 궤적: `step_up`, 일반 CAPTCHA 결과 유지, outcome의 `would_have_action=step_up`
3. 반복 궤적: `step_up_and_rate_limit`, 일반 CAPTCHA 결과 유지, outcome 저장
4. 같은 outcome 재전송: `idempotent=true`, 중복 행 없음
5. backend key 없이 outcome 요청: `401`
6. `RISK_POLICY_MODE=active` 상태에서 shadow outcome 요청: `409 shadow_mode_disabled`

## 5. 종료 판단

shadow mode가 끝나도 바로 차단 모드로 전환하지 않는다. CAPTCHA 정답률, would-step-up 비율,
추가 인증이 필요했던 실제 세션의 분포를 팀이 검토한 뒤 별도 승인으로만
`RISK_POLICY_MODE=active`를 고려한다.
