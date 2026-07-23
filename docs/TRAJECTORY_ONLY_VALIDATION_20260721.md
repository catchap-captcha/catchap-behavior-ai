# 포인터 궤적 전용 모델 검증 - 2026-07-21

## 결론

`FAIL / production 승격 금지`

이 후보는 모델 입력으로 단일 포인터 궤적의 `(x, y, t)`만 사용한다. 화면 상호작용
요약값인 `regrab_count`, `retry_count`, `pointercancel_count`, `empty_click_count`,
`failed_drop_count`는 모델 입력에서 제외했다.

일반 규칙형 봇과 일부 어려운 외부 봇에는 효과가 있었지만, 사람의 실제 궤적을 약하게
변형한 외부 `replay_warp`를 막지 못했다. 따라서 이 결과를 production 모델로 올리거나
CAPTCHA를 차단하는 데 사용하면 안 된다.

## 입력과 데이터 분리

- 피처: Feature v2 중 39개. 좌표, 시간, 속도, 가속도, 곡률, 정지, 엔트로피 등 한 번의
  포인터 궤적에서 계산되는 값만 사용한다.
- 사람: 20,066건, 연결 참여자 46명. 참여자 단위로 train/validation/test를 분리했다.
- 학습용 봇: 13,000건
  - 규칙형 10 family 10,000건
  - PCA+GMM 개발용 합성 봇 1,000건
  - 개발용 적대적 replay 2,000건
- 외부 평가용 봇: 각 1,000건. 학습과 임계값 보정에 사용하지 않았다.
  - PCA+GMM 외부 holdout
  - 회전·교대 time-warp 적대적 replay holdout
  - `replay_warp` 외부 holdout

## 일반 사람/봇 Test

| 모델 | 정상 사용자 FRR | 알려진 Bot ASR | 판정 |
|---|---:|---:|---|
| ExtraTrees | 0.72% | 7.53% | Bot ASR 5% 기준 실패 |
| LightGBM | 0.18% | 7.30% | Bot ASR 5% 기준 실패 |
| RandomForest | 0.36% | 7.21% | Bot ASR 5% 기준 실패 |
| XGBoost | 0.23% | 7.30% | Bot ASR 5% 기준 실패 |

사람 오탐률은 일반 test에서 모두 3% 이하였지만, 알려진 봇 통과율조차 5% 이하라는
기준에는 도달하지 못했다.

## 외부 봇 Holdout ASR

ASR은 봇이 사람으로 통과한 비율이며 낮을수록 좋다. 목표는 5% 이하이다.

| 모델 | 외부 ML 봇 | 외부 적대적 replay | 외부 `replay_warp` |
|---|---:|---:|---:|
| ExtraTrees | 13.7% | 3.0% | 98.8% |
| LightGBM | 0.8% | 1.6% | 92.0% |
| RandomForest | 0.9% | 1.7% | 96.0% |
| XGBoost | 0.6% | 1.6% | 91.1% |

`replay_warp`는 가장 나은 XGBoost도 91.1%가 통과했다. 이는 단일 궤적만 보고서는
원본 사람 궤적과 매우 가까운 재생·변형 궤적을 안정적으로 분리하지 못한다는 증거다.

## 정확한 해석

1. 모델은 규칙적으로 생성한 자동화 궤적과 일부 독립적인 ML/적대적 변형을 탐지할 수 있다.
2. 공격자가 실제 사람 궤적을 저장한 뒤 충분히 사람답게 재생·변형하면, 현재 궤적 하나만으로
   사람과 확실히 구별할 수 없다. 원본과 사실상 같은 `(x, y, t)`라면 어떤 단일-궤적 분류기도
   구별할 정보가 없다.
3. 그래서 이 모델은 `생성된 자동화 궤적 탐지` 후보로 계속 개선할 수 있지만, replay 방어 전체를
   대체하지는 못한다. exact replay·세션 재사용 등은 별도의 서버 방어 계층에서 다뤄야 한다.

## 다음 실험 순서

1. 이번 외부 `replay_warp` holdout은 더 이상 학습·튜닝에 사용하지 않고 보존한다.
2. 개발용 replay 생성기를 더 넓은 재표본화, 미세 곡률 변화, 국소 속도 변화 조합으로 확장한다.
3. 새로 만든 변형 조합과 새 참여자의 궤적을 다음 외부 holdout으로 분리한다.
4. 사람 100명 이상, 마우스·터치·브라우저별 데이터로 Human FRR을 다시 확인한다.
5. 새 외부 holdout에서 Bot ASR 5% 이하와 Human FRR 3% 이하를 동시에 만족할 때만 shadow mode를
   검토한다.

## 재현 경로

- 학습 결과: `reports/local_h20066_b13000_trajectory_only_20260721/training_summary.json`
- 후보 모델: `models/candidate/local_h20066_b13000_trajectory_only_20260721/`
- 데이터 분리 기록: `data/processed/local_h20066_b13000_trajectory_only_20260721/manifest.json`

모든 후보는 `candidate_only`이며 production 디렉터리에 복사하지 않았다.
