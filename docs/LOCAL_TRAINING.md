# 로컬 Human/Bot 학습 구조

체크포인트별 발표용 비교 이력은
[`docs/EXPERIMENT_HISTORY.md`](EXPERIMENT_HISTORY.md)에 기록한다.

## 현재 데이터

| 클래스 | 건수 | 출처 |
|---|---:|---|
| Human | 20,066 | 2026-07-21 읽기 전용 DB 스냅샷, 품질 정제 완료 |
| 합성 Bot | 10,000 | 10개 family 각 1,000건 |
| 브라우저 Bot | 90 | Chrome+Playwright 3종, 외부 holdout 전용 |

원본 21,678건 중 1,612건은 궤적 누락, 포인트 부족 또는 명시적 제외 상태로 학습에서
제외했다. 익명 Human 889건은 train-only로 배치한다. 사용 가능한 연결 참여자는 46명이라
100명 production 다양성 게이트를 아직 만족하지 못한다.

## 전체 흐름

```text
Human pointer trace 20,066      Synthetic Bot trace 10,000
             |                              |
             +-------- 29 Feature 계산 -----+
                            |
                  데이터 준비 상태 검사
                            |
           train / validation / test 그룹 분리
                            |
    RandomForest / ExtraTrees / XGBoost / LightGBM 학습
                            |
       사용자·봇 그룹 5-fold OOF threshold 보정
                            |
           개발 데이터 전체 재학습 후 test 1회 평가
                            |
              FRR -> Bot Recall -> F1 -> 속도 순 선택
                            |
       unseen family + Playwright 외부 holdout 검사
```

## 분할 구조

- 연결된 Human: 같은 참여자의 모든 시도를 한 split에만 둔다.
- 익명 Human 889건: 참여자 누수를 검사할 수 없으므로 train에만 두며 OOF 임계값
  보정에서는 제외한다.
- Rule Bot: family별 독립 배치 그룹을 만들고 각 family가 train/validation/test에
  모두 들어가게 분리한다.
- 기본 비율은 그룹 기준 약 70/15/15이며, 행 수는 참여자별 시도 수 때문에 정확히
  70/15/15가 아닐 수 있다.
- 선택 모델은 straight/accel/jitter 중 하나를 학습에서 통째로 제외한 3회 추가
  테스트를 받는다. 이 결과는 처음 보는 규칙형 공격으로의 일반화를 확인한다.

## 검증에서 검사하는 것

1. 데이터 무결성: 라벨 누락, Feature 스키마 버전, NULL/NaN/Infinity, ID 중복.
2. 데이터 누수: 같은 연결 참여자 또는 같은 Bot 배치 그룹이 여러 split에 섞이지 않는지.
3. Human FRR: 실제 사람을 Bot으로 막은 비율. 서비스 보호를 위해 가장 중요하며 3% 이하가 목표다.
4. Bot Recall: 실제 Bot 중 차단한 비율. FRR 목표를 만족하는 모델끼리 비교한다.
5. Human Precision/Recall/F1: Human 판정의 균형을 확인한다.
6. ROC-AUC/PR-AUC: threshold 하나에 의존하지 않은 분류력 참고 지표다.
7. Confusion Matrix: Human/Bot의 네 가지 판정 건수를 직접 확인한다.
8. 추론 시간: 실시간 CAPTCHA 응답에 사용할 수 있는지 확인한다.
9. Feature Importance: 모델이 정답 여부 같은 금지 신호가 아니라 행동 Feature를 사용하는지 본다.
10. Bot family holdout: 학습에서 보지 못한 규칙형 Bot에도 탐지가 유지되는지 본다.
11. 결합 판정: ML, Fingerprint, DTW, 세션 요청 빈도를 합친 최종 FRR·ASR을 본다.

후보 추천은 일반 test 점수만 보지 않고 알려진 Bot ASR, 미지 family 최악 ASR,
`replay_warp` ASR, Playwright holdout ASR을 함께 본다. 하나라도 기준을 실패하면
`deployment_eligible=false`다.

2026-07-21 1차 결합 평가에서는 알려진 Bot 전체 ASR이 0.84%로 낮아졌지만,
`replay_warp` 최악 ASR이 8.75%로 5% 기준을 실패했다. 이후 9개 replay 쌍
신호를 추가한 오프라인 실험은 보지 않은 참가자 기반 새 seed holdout에서
Replay ASR 0/1,000, Human FRR 0/2,219를 기록했다. 다만 같은 warp family에
대한 결과이며 앱·API에는 적용하지 않는다. 상세 수치는
[`docs/CHECKPOINT_20260721.md`](CHECKPOINT_20260721.md)에 기록한다.

## 가장 중요한 해석 주의점

- 높은 정확도 하나만 보면 안 된다. Human FRR과 Bot Recall을 반드시 함께 본다.
- 현재 Bot은 규칙형 합성 데이터라서 높은 점수는 baseline 통과를 뜻할 뿐, 실제 공격 방어가
  완성됐다는 뜻은 아니다.
- Human 참여자 다양성은 궤적 기준 46명이다. 배포 전 더 많은 참여자와 기기·브라우저를
  확보해야 한다.
- 선택 모델은 candidate로만 저장한다. 실제 서비스 production 승격은 별도 검증 뒤 수행한다.

## 실행

프로젝트 루트 `ai-service`에서 실행한다.

```bash
.venv/bin/python -m training.run_local_training
```

산출물:

```text
data/processed/local_h4786_b3000_20260713/
  bot_features.jsonl
  combined_features.jsonl
  manifest.json

reports/local_h4786_b3000_20260713/
  data_readiness.json
  split_manifest.json
  model_comparison.csv
  training_summary.json
  confusion_matrix_*.png
  feature_importance_*.csv

models/candidate/local_h4786_b3000_20260713/
  random_forest.joblib
  xgboost.joblib
  lightgbm.joblib
```

## 2026-07-13 실제 학습 결과

데이터 분할:

| Split | Human | Bot | 합계 |
|---|---:|---:|---:|
| Train | 3,527 | 2,012 | 5,539 |
| Validation | 658 | 508 | 1,166 |
| Test | 601 | 480 | 1,081 |

일반 test 결과:

| 모델 | Accuracy | Human FRR | Bot Recall | Human F1 |
|---|---:|---:|---:|---:|
| RandomForest | 100.00% | 0.00% | 100.00% | 100.00% |
| XGBoost | 100.00% | 0.00% | 100.00% | 100.00% |
| LightGBM | 98.80% | 2.16% | 100.00% | 98.91% |

일반 test 점수는 같은 세 가지 규칙형 Bot 분포를 나눠 평가한 결과라 매우 높다. 이
숫자만으로 배포하면 안 된다.

Bot family holdout 결과:

| 모델 | Accel | Jitter | Straight | 최악 | 평균 |
|---|---:|---:|---:|---:|---:|
| RandomForest | 100.0% | 2.1% | 100.0% | 2.1% | 67.4% |
| XGBoost | 31.4% | 66.2% | 100.0% | 31.4% | 65.9% |
| LightGBM | 31.4% | 83.1% | 100.0% | 31.4% | 71.5% |

상대적으로 가장 나은 후보는 LightGBM이지만 최악의 unseen-family Bot Recall이
31.4%로 80% 게이트보다 낮다. 따라서 모델 파일은 candidate로 보관하되
`deployment_eligible=false`, production 미승격으로 판정한다.

다음 학습에서 중요한 순서:

1. 동일한 세 종류의 수량만 늘리기보다 새로운 자동화·재생·곡선·구간 이동 패턴을 추가한다.
2. generator/version 하나 이상을 학습에서 완전히 제외해 최종 공격 테스트로 유지한다.
3. 실제 브라우저 자동화에서 수집된 Bot 궤적을 규칙형 합성 데이터와 섞는다.
4. Human은 참여자·마우스·터치·브라우저별 오탐률을 따로 확인한다.
5. 최악의 family-holdout Bot Recall 80% 이상과 Human FRR 3% 이하를 동시에 만족한 뒤 배포를 검토한다.
